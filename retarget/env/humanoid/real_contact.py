"""Differentiable ComFree prediction with collision refreshed at every step.

IPOPT evaluates the actual robot trajectory. Each current Torch state is copied
to private MJWarp scratch data to read real collision rows, their Jacobians and
impedance masses. No desired contact point is inserted into those rows. The
piecewise local derivatives of each refreshed stage pass through the current
contact Jacobians, actuator model, ComFree response and state integration;
discrete changes of contact topology are reevaluated at the next objective call.

The engine snapshots are detached. These derivatives are therefore local
partials; they omit state derivatives of the refreshed mass, bias and contact
data, and are not the full derivative of the refreshed multi-step objective.
The actual humanoid finite-difference diagnostic currently fails; see
docs/humanoid_scsp.md before treating this as a complete IPOPT gradient model.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch
import warp as wp

from retarget.dynamics.comfree.response import _contact_response_unpadded
from retarget.task.humanoid.cost_contact import (
    ContactCostOptions, _body_subtree, _object_joint, _refresh, _snapshot, _stream,
)

from .frozen_contact import (
    FrozenContactRollout, _actuators, _difference, _frame, _geometry, _integrate,
)


@dataclass(frozen=True)
class RealContactResult:
    """Four logical points: two palms, then two soles; no sampled foot points.

    States contain the initial node. Sites, reactions, physical gaps and masks
    describe the H pre-integration stages. Hand reactions are forces on the
    object from actual hand-object collisions; sole reactions are forces on the
    robot from actual ground collisions. Missing contacts have exactly zero
    reaction and gap and a false mask. Physical contact manifolds are summed
    into these four entries without modifying engine constraint rows.
    """

    qpos: torch.Tensor  # (1, H+1, nq)
    qvel: torch.Tensor  # (1, H+1, nv)
    site_xpos: torch.Tensor  # (1, H, 4, 3), fixed logical site locations
    site_jacobian: torch.Tensor  # (1, H, 4, 3, nv), refreshed actual geometry
    contact_force: torch.Tensor  # (1, H, 4, 3), world frame
    contact_gap: torch.Tensor  # (1, H, 4), minimum physical manifold gap
    active_mask: torch.Tensor  # (1, H, 4), real active constraint contacts
    actual_contact_count: torch.Tensor  # (1, H), active engine manifolds
    constraint_count: torch.Tensor  # (1, H), actual engine rows including limits
    timestep: torch.Tensor  # (H,)


@dataclass(frozen=True)
class _ContactMap:
    """Detached mapping from real constraint rows to four logical reactions."""

    row_indices: torch.Tensor  # (manifolds, max_pyramid_rows), invalid slots=0
    row_valid: torch.Tensor  # matching Boolean mask
    force_rays: torch.Tensor  # (manifolds, max_pyramid_rows, 3), on geom 2
    signs: torch.Tensor  # (manifolds, 4), sign of force on logical recipient
    gap: torch.Tensor  # (manifolds,), physical distances
    gap_jacobian: torch.Tensor  # (manifolds, nv)
    active_count: torch.Tensor  # scalar, all real active manifolds

    def evaluate(self, row_force, delta):
        """Decode pyramidal forces with differentiable ComFree row reactions."""
        batch = len(row_force)
        if self.row_indices.shape[0] == 0:
            zeros = row_force.new_zeros((batch, 4, 3))
            return zeros, row_force.new_zeros((batch, 4)), torch.zeros((batch, 4), device=row_force.device, dtype=torch.bool)
        selected = row_force[:, self.row_indices]
        selected = torch.where(self.row_valid[None], selected, 0.0)
        force_on_second = torch.einsum("ncr,crd->ncd", selected, self.force_rays)
        force = torch.einsum("ck,ncd->nkd", self.signs, force_on_second)
        active = self.signs != 0
        distances = self.gap[None] + torch.einsum("cv,nv->nc", self.gap_jacobian, delta)
        gaps = torch.where(active[None], distances[..., None], torch.inf).amin(dim=1)
        mask = active.any(dim=0)[None].expand(batch, -1)
        return force, torch.where(mask, gaps, 0.0), mask


def _contact_map(env, frame, site_bodies, object_bodies):
    """Read globally packed real contacts and their exact per-world row IDs.

    The engine normal points from geom 1 to geom 2. Pyramidal row r contributes
    ``n +/- mu_i*t_i`` to force on geom 2; rotational friction edges contribute
    normal force plus a torque, so only their normal component appears here.
    """
    d, cpu = env.data_wp, env.model_cpu
    device, dtype = frame.gap.device, frame.gap.dtype
    ncon = int(_snapshot(d.nacon)[0])
    rows = len(frame.gap)
    if ncon == 0 or rows == 0:
        return _ContactMap(
            torch.empty((0, 1), device=device, dtype=torch.long),
            torch.empty((0, 1), device=device, dtype=torch.bool),
            torch.empty((0, 1, 3), device=device, dtype=dtype),
            torch.empty((0, 4), device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=dtype),
            torch.empty((0, cpu.nv), device=device, dtype=dtype),
            torch.zeros((), device=device, dtype=torch.long),
        )
    geom = _snapshot(d.contact.geom)[:ncon].long()
    world = _snapshot(d.contact.worldid)[:ncon].long()
    good = (world == 0) & ((geom >= 0) & (geom < cpu.ngeom)).all(-1)
    ids = torch.nonzero(good).flatten()
    geom = geom[ids]
    addresses = _snapshot(d.contact.efc_address)[ids].long()
    dimensions = _snapshot(d.contact.dim)[ids].long()
    frames = _snapshot(d.contact.frame)[ids]
    friction = _snapshot(d.contact.friction)[ids]
    physical_gap = _snapshot(d.contact.dist)[ids]
    slots = torch.arange(addresses.shape[1], device=device)
    count = torch.where(dimensions == 1, 1, 2 * (dimensions - 1))
    valid = (slots[None] < count[:, None]) & (addresses >= 0) & (addresses < rows)
    active = valid.any(-1)
    safe_rows = addresses.clamp(0, rows - 1)

    tangent_index = (slots // 2 + 1).clamp(max=2)
    friction_index = (slots // 2).clamp(max=friction.shape[1]-1)
    direction = torch.where(slots % 2 == 0, 1.0, -1.0).to(dtype)
    tangent = frames[:, tangent_index] * friction[:, friction_index, None] * direction[None, :, None]
    tangent = torch.where(((dimensions[:, None] > 1) & (slots[None] < 4))[..., None], tangent, 0.0)
    rays = frames[:, :1].expand(-1, len(slots), -1) + tangent

    body_table = torch.as_tensor(cpu.geom_bodyid.copy(), device=device, dtype=torch.long)
    weld_table = torch.as_tensor(cpu.body_weldid.copy(), device=device, dtype=torch.long)
    bodies = body_table[geom]
    weld = weld_table[bodies]
    sites_weld = weld_table[site_bodies]
    first_matches = weld[:, :1] == sites_weld[None]
    second_matches = weld[:, 1:] == sites_weld[None]
    on_object = object_bodies[bodies]
    on_ground = weld == 0
    # Palms only export reactions delivered to the object; hand-floor or self
    # contacts still affect dynamics, but cannot masquerade as object force.
    hand_contact = (first_matches[:, :2] & on_object[:, 1:]) | (second_matches[:, :2] & on_object[:, :1])
    hand_sign = torch.where(on_object[:, 1:], 1.0, -1.0).to(dtype) * hand_contact
    # Ground reactions are forces on the sole body, hence the opposite sign
    # when that logical foot is the first engine geom.
    foot_contact = (first_matches[:, 2:] & on_ground[:, 1:]) | (second_matches[:, 2:] & on_ground[:, :1])
    foot_sign = torch.where(second_matches[:, 2:], 1.0, -1.0).to(dtype) * foot_contact
    signs = torch.cat((hand_sign, foot_sign), -1) * active[:, None]
    first_row = torch.where(valid, addresses, rows).amin(-1).clamp(max=rows-1)
    gap_jacobian = torch.where(active[:, None], frame.normal_jacobian[first_row], 0.0)
    return _ContactMap(safe_rows, valid, rays, signs, physical_gap, gap_jacobian, active.sum())


class RealContactRollout:
    """Recompute actual contacts throughout every single-action rollout.

    Only batch size one is accepted: IPOPT follows one robot action trajectory.
    Upper-layer contact-combination populations use their separate object model
    and cannot create contacts or Jacobian rows in this backend.
    """

    def __init__(self, config, env, *, options=None):
        self.config = config
        self.options = options or ContactCostOptions()
        ids = tuple(config.contact_site_ids)
        if len(ids) != 4 or len(set(ids)) != 4:
            raise ValueError("real-contact rollout requires two palm and two sole site IDs")
        template, self.scratch = FrozenContactRollout._initialize(config, env, self.options)
        self.device, self.dtype = template.device, template.dtype
        self.nq, self.nv, self.nu = template.nq, template.nv, template.nu
        self._layout = template._layout
        self.object_qpos_adr, self.object_dof_adr = template.object_qpos_adr, template.object_dof_adr
        self.initial_qpos = _snapshot(env.data_wp.qpos)[0]
        self.initial_qvel = _snapshot(env.data_wp.qvel)[0]
        cpu = env.model_cpu
        if any(site < 0 or site >= cpu.nsite for site in ids):
            raise ValueError("real-contact rollout contains an invalid logical site")
        self.site_bodies = torch.as_tensor(cpu.site_bodyid[list(ids)].copy(), device=self.device, dtype=torch.long)
        welds = cpu.body_weldid[cpu.site_bodyid[list(ids)]]
        if len(set(map(int, welds))) != 4 or (welds == 0).any():
            raise ValueError("real-contact palm/sole sites require four distinct moving body groups")
        joint = _object_joint(config, env, self.options)
        if joint is None:
            raise ValueError("real-contact rollout requires a separate object free joint")
        self.object_bodies = _body_subtree(cpu, int(cpu.jnt_bodyid[joint]), self.device)
        self._actuator = None

    def _refresh_stage(self, qpos, qvel, control):
        scratch = self.scratch
        with wp.ScopedDevice(scratch.data_wp.qpos.device), _stream(scratch):
            for name, value in (("qpos", qpos), ("qvel", qvel), ("ctrl", control)):
                target = getattr(scratch.data_wp, name)
                wp.copy(target, wp.from_torch(value.detach().contiguous(), dtype=target.dtype))
        _refresh(scratch, dynamics=True)
        if self._actuator is None:
            self._actuator = _actuators(scratch)
        return _frame(self.config, scratch, self.options), _geometry(self.config, scratch)

    def rollout(self, controls):
        if not isinstance(controls, torch.Tensor) or not controls.is_floating_point():
            raise TypeError("real-contact controls must be a floating Torch tensor")
        if controls.ndim != 3 or controls.shape[0] != 1 or controls.shape[-1] != self.nu or controls.shape[1] == 0:
            raise ValueError("real-contact controls must have shape (1, H, nu), H > 0")
        controls = controls.to(device=self.device, dtype=self.dtype)
        qpos, qvel = self.initial_qpos[None], self.initial_qvel[None]
        qpositions, qvelocities = [qpos], [qvel]
        sites, site_jacobians, forces, gaps, masks, counts, row_counts, timesteps = [], [], [], [], [], [], [], []
        for t in range(controls.shape[1]):
            control = controls[:, t]
            frame, geometry = self._refresh_stage(qpos, qvel, control)
            delta = _difference(qpos, frame.nominal_qpos, self._layout)
            site = geometry.site_xpos + torch.einsum("kcv,nv->nkc", geometry.site_jacobian, delta)
            moment = frame.actuator_moment
            length = frame.actuator_length + torch.einsum("uv,nv->nu", moment, delta)
            rate = torch.einsum("uv,nv->nu", moment, qvel)
            actuator_force = self._actuator.force(control, length, rate)
            smooth = frame.smooth_residual + torch.einsum("uv,nu->nv", moment, actuator_force)
            acceleration = torch.einsum("ij,nj->ni", frame.mass_inverse, smooth)
            distance = frame.gap + torch.einsum("rv,nv->nr", frame.normal_jacobian, delta)
            total, row_force, _ = _contact_response_unpadded(
                frame.timestep, qvel, acceleration, frame.jacobian[None], distance,
                frame.impedance_mass[None], smooth,
                stiffness=self.options.comfree_stiffness, damping=self.options.comfree_damping,
            )
            mapping = _contact_map(self.scratch, frame, self.site_bodies, self.object_bodies)
            force, gap, mask = mapping.evaluate(row_force, delta)
            sites.append(site)
            site_jacobians.append(geometry.site_jacobian[None])
            forces.append(force)
            gaps.append(gap)
            masks.append(mask)
            counts.append(mapping.active_count)
            row_counts.append(torch.full((), len(frame.gap), device=self.device, dtype=torch.long))
            timesteps.append(frame.timestep)
            qvel = qvel + frame.timestep * torch.einsum("ij,nj->ni", frame.mass_inverse, total)
            qpos = _integrate(qpos, qvel, frame.timestep, self._layout)
            qpositions.append(qpos)
            qvelocities.append(qvel)
        return RealContactResult(
            qpos=torch.stack(qpositions, 1), qvel=torch.stack(qvelocities, 1),
            site_xpos=torch.stack(sites, 1), site_jacobian=torch.stack(site_jacobians, 1),
            contact_force=torch.stack(forces, 1),
            contact_gap=torch.stack(gaps, 1), active_mask=torch.stack(masks, 1),
            actual_contact_count=torch.stack(counts)[None], constraint_count=torch.stack(row_counts)[None],
            timestep=torch.stack(timesteps),
        )
