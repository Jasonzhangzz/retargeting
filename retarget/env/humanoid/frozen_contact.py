"""Differentiable trajectory rollout with a reusable MJWarp contact cache.

MJWarp is used only to construct a nominal trajectory cache. Candidate controls
are evaluated entirely in Torch: neither collision detection nor simulator state
is revisited when MPPI changes its controls. The cached model is a local model;
contacts absent from the nominal trajectory cannot be discovered by a rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from retarget.dynamics.comfree import contact_response_torch
from retarget.task.humanoid.cost_contact import (
    ContactCostOptions,
    _actuator_force,
    _body_subtree,
    _dense_dynamics,
    _object_joint,
    _refresh,
    _snapshot,
    _stream,
)


@dataclass(frozen=True)
class RolloutResult:
    """States include the initial node; all other arrays describe H steps."""

    qpos: torch.Tensor
    qvel: torch.Tensor
    site_xpos: torch.Tensor
    penetration_distance: tuple[torch.Tensor, ...]
    object_displacement: torch.Tensor
    timestep: torch.Tensor
    object_qpos_adr: int | None
    object_dof_adr: int | None


@dataclass(frozen=True)
class ContactFrame:
    """One detached nominal dynamics snapshot, with unpadded constraint rows.

    ``normal_jacobian`` maps a configuration perturbation to each constraint's
    normal gap. Unlike ``jacobian``, contact rows have no friction tangents.
    ``contact_*`` contains physical robot-object contacts only, and distances
    include no solver margin. Both Jacobians use MuJoCo's nv tangent coordinates.
    """

    nominal_qpos: torch.Tensor
    nominal_qvel: torch.Tensor
    mass_cholesky: torch.Tensor
    jacobian: torch.Tensor
    normal_jacobian: torch.Tensor
    gap: torch.Tensor
    impedance_mass: torch.Tensor
    contact_jacobian: torch.Tensor
    contact_distance: torch.Tensor
    actuator_moment: torch.Tensor
    actuator_length: torch.Tensor
    smooth_residual: torch.Tensor
    timestep: torch.Tensor


@dataclass(frozen=True)
class _Geometry:
    qpos: torch.Tensor
    site_xpos: torch.Tensor
    site_jacobian: torch.Tensor


@dataclass(frozen=True)
class _Actuators:
    gain: torch.Tensor
    bias: torch.Tensor
    affine_gain: torch.Tensor
    affine_bias: torch.Tensor
    ctrl_limited: torch.Tensor
    ctrl_range: torch.Tensor
    force_limited: torch.Tensor
    force_range: torch.Tensor
    clamp_control: bool

    def force(self, control, length, velocity):
        if self.clamp_control:
            bounded = torch.maximum(
                torch.minimum(control, self.ctrl_range[:, 1]), self.ctrl_range[:, 0]
            )
            control = torch.where(self.ctrl_limited, bounded, control)
        gain = self.gain[:, 0] + self.affine_gain * (
            self.gain[:, 1] * length + self.gain[:, 2] * velocity
        )
        bias = self.bias[:, 0] + self.bias[:, 1] * length + self.bias[:, 2] * velocity
        force = gain * control + torch.where(self.affine_bias, bias, 0.0)
        bounded = torch.maximum(
            torch.minimum(force, self.force_range[:, 1]), self.force_range[:, 0]
        )
        return torch.where(self.force_limited, bounded, force)


def _quaternion_product(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )


def _unit_quaternion(q):
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def _rotation_vector(a, b):
    """Local tangent taking quaternion b to a, finite at the identity."""
    b = _unit_quaternion(b)
    inverse = torch.cat((b[..., :1], -b[..., 1:]), -1)
    relative = _quaternion_product(inverse, _unit_quaternion(a))
    relative = torch.where(relative[..., :1] < 0, -relative, relative)
    xyz = relative[..., 1:]
    norm = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, relative[..., :1])
    scale = torch.where(norm > 1e-7, angle / norm.clamp_min(1e-7), 2.0)
    return xyz * scale


def _advance_quaternion(q, velocity, dt):
    rotation = velocity * dt
    angle = torch.linalg.vector_norm(rotation, dim=-1, keepdim=True)
    # sin(angle/2)/angle has an analytic continuation at zero. sinc avoids
    # zero-norm divisions and retains the correct derivative at identity.
    xyz = 0.5 * torch.sinc(angle / (2.0 * torch.pi)) * rotation
    delta = torch.cat((torch.cos(0.5 * angle), xyz), -1)
    return _unit_quaternion(_quaternion_product(_unit_quaternion(q), delta))


def _joint_layout(cpu):
    return tuple(
        (int(cpu.jnt_type[j]), int(cpu.jnt_qposadr[j]), int(cpu.jnt_dofadr[j]))
        for j in range(cpu.njnt)
    )


def _difference(qpos, reference, layout):
    parts = []
    for kind, qadr, _ in layout:
        if kind == int(mujoco.mjtJoint.mjJNT_FREE):
            parts.extend(
                (
                    qpos[..., qadr : qadr + 3] - reference[..., qadr : qadr + 3],
                    _rotation_vector(
                        qpos[..., qadr + 3 : qadr + 7],
                        reference[..., qadr + 3 : qadr + 7],
                    ),
                )
            )
        elif kind == int(mujoco.mjtJoint.mjJNT_BALL):
            parts.append(
                _rotation_vector(
                    qpos[..., qadr : qadr + 4], reference[..., qadr : qadr + 4]
                )
            )
        else:
            parts.append(qpos[..., qadr : qadr + 1] - reference[..., qadr : qadr + 1])
    return torch.cat(parts, -1)


def _integrate(qpos, velocity, dt, layout):
    parts = []
    for kind, qadr, vadr in layout:
        if kind == int(mujoco.mjtJoint.mjJNT_FREE):
            parts.extend(
                (
                    qpos[..., qadr : qadr + 3] + dt * velocity[..., vadr : vadr + 3],
                    _advance_quaternion(
                        qpos[..., qadr + 3 : qadr + 7],
                        velocity[..., vadr + 3 : vadr + 6],
                        dt,
                    ),
                )
            )
        elif kind == int(mujoco.mjtJoint.mjJNT_BALL):
            parts.append(
                _advance_quaternion(
                    qpos[..., qadr : qadr + 4], velocity[..., vadr : vadr + 3], dt
                )
            )
        else:
            parts.append(
                qpos[..., qadr : qadr + 1] + dt * velocity[..., vadr : vadr + 1]
            )
    return torch.cat(parts, -1)


def _point_jacobians(env, points, bodies):
    """Read MJWarp's current spatial DOFs, without another engine evaluation."""
    cpu, m, d = env.model_cpu, env.model_wp, env.data_wp
    nv = cpu.nv
    cdof = _snapshot(d.cdof)[0, :nv]
    subtree = _snapshot(d.subtree_com)[0]
    root = _snapshot(m.body_rootid).long()[bodies]
    ancestor = _snapshot(m.body_isdofancestor).bool()[bodies, :nv]
    offset = points[:, None, :] - subtree[root, None, :]
    jacobian = cdof[None, :, 3:] + torch.linalg.cross(
        cdof[None, :, :3].expand(len(points), -1, -1), offset.expand(-1, nv, -1), dim=-1
    )
    return torch.where(ancestor[..., None], jacobian, 0.0).transpose(-1, -2)


def _geometry(config, env):
    qpos = _snapshot(env.data_wp.qpos)[0]
    ids = torch.as_tensor(config.contact_site_ids, device=qpos.device, dtype=torch.long)
    if ids.numel() and bool(torch.any((ids < 0) | (ids >= env.model_cpu.nsite))):
        raise ValueError("contact_site_ids contains an invalid site")
    sites = _snapshot(env.data_wp.site_xpos)[0, ids]
    bodies = torch.as_tensor(
        env.model_cpu.site_bodyid.copy(), device=qpos.device, dtype=torch.long
    )[ids]
    return _Geometry(qpos, sites, _point_jacobians(env, sites, bodies))


def _actuators(env):
    # Keep the existing helper's explicit checks for unsupported callbacks,
    # stateful/delayed actuators, force limits and actuator disable flags.
    _actuator_force(env, _snapshot(env.data_wp.ctrl))
    m, cpu = env.model_wp, env.model_cpu

    def first_model(array):
        return _snapshot(array)[0]

    return _Actuators(
        first_model(m.actuator_gainprm),
        first_model(m.actuator_biasprm),
        _snapshot(m.actuator_gaintype) == int(mujoco.mjtGain.mjGAIN_AFFINE),
        _snapshot(m.actuator_biastype) == int(mujoco.mjtBias.mjBIAS_AFFINE),
        _snapshot(m.actuator_ctrllimited).bool(),
        first_model(m.actuator_ctrlrange),
        _snapshot(m.actuator_forcelimited).bool(),
        first_model(m.actuator_forcerange),
        not bool(cpu.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)),
    )


def _contacts(config, env, options, jacobian, row_count):
    """Freeze normal geometry, including physical gaps outside constraint margin."""
    d, cpu = env.data_wp, env.model_cpu
    normal_rows = jacobian.clone()
    ncon = int(_snapshot(d.nacon)[0])
    device, dtype = jacobian.device, jacobian.dtype
    empty_j = torch.empty((0, cpu.nv), dtype=dtype, device=device)
    empty_d = torch.empty(0, dtype=dtype, device=device)
    if ncon == 0:
        return normal_rows, empty_j, empty_d
    geom = _snapshot(d.contact.geom)[:ncon].long()
    world = _snapshot(d.contact.worldid)[:ncon].long()
    valid = (world == 0) & ((geom >= 0) & (geom < cpu.ngeom)).all(-1)
    indices = torch.nonzero(valid).flatten()
    if not len(indices):
        return normal_rows, empty_j, empty_d
    geom = geom[indices]
    body_table = torch.as_tensor(
        cpu.geom_bodyid.copy(), device=device, dtype=torch.long
    )
    bodies = body_table[geom]
    points = _snapshot(d.contact.pos)[indices]
    normals = _snapshot(d.contact.frame)[indices, 0]
    first = _point_jacobians(env, points, bodies[:, 0])
    second = _point_jacobians(env, points, bodies[:, 1])
    normal_jacobian = torch.einsum("pc,pcv->pv", normals, second - first)

    addresses = _snapshot(d.contact.efc_address)[indices].long()
    dimensions = _snapshot(d.contact.dim)[indices].long()
    # Contacts are globally packed, while constraint row addresses are local
    # to a world. Respect condim so unused address columns cannot leak in.
    for p in range(len(indices)):
        rays = 1 if int(dimensions[p]) == 1 else 2 * (int(dimensions[p]) - 1)
        rows = addresses[p, :rays]
        rows = rows[(rows >= 0) & (rows < row_count)]
        normal_rows[rows] = normal_jacobian[p]

    joint = _object_joint(config, env, options)
    if joint is None:
        return normal_rows, empty_j, empty_d
    free = [j for j in range(cpu.njnt) if cpu.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    robot_body = int(cpu.jnt_bodyid[free[0]])
    object_body = int(cpu.jnt_bodyid[joint])
    if robot_body == object_body:
        raise ValueError("robot and object must have separate body roots")
    robots = _body_subtree(cpu, robot_body, device)
    objects = _body_subtree(cpu, object_body, device)
    selected = (robots[bodies[:, 0]] & objects[bodies[:, 1]]) | (
        objects[bodies[:, 0]] & robots[bodies[:, 1]]
    )
    distance = _snapshot(d.contact.dist)[indices]
    return normal_rows, normal_jacobian[selected].clone(), distance[selected].clone()


def _frame(config, env, options):
    d = env.data_wp
    mass, jacobian, moment = _dense_dynamics(env)
    count = int(_snapshot(d.nefc)[0])
    if count > jacobian.shape[1]:
        raise ValueError("MJWarp constraint capacity overflow while caching contacts")
    jacobian = jacobian[0, :count, : env.model_cpu.nv].clone()
    normal, contact_jacobian, contact_distance = _contacts(
        config, env, options, jacobian, count
    )
    force = _snapshot(d.actuator_force)[0]
    moment = moment[0].clone()
    residual = _snapshot(d.qfrc_smooth)[0] - torch.einsum("uv,u->v", moment, force)
    return ContactFrame(
        nominal_qpos=_snapshot(d.qpos)[0],
        nominal_qvel=_snapshot(d.qvel)[0],
        mass_cholesky=torch.linalg.cholesky(mass[0]),
        jacobian=jacobian,
        normal_jacobian=normal,
        gap=_snapshot(d.efc.pos)[0, :count].clone(),
        impedance_mass=_snapshot(d.efc.D)[0, :count].clone(),
        contact_jacobian=contact_jacobian,
        contact_distance=contact_distance,
        actuator_moment=moment,
        actuator_length=_snapshot(d.actuator_length)[0],
        smooth_residual=residual,
        timestep=_snapshot(env.model_wp.opt.timestep)[0],
    )


def _scratch(env):
    """Copy initial first-world inputs; all derived arrays belong to this cache."""
    cpu, source = env.model_cpu, env.data_wp
    device = source.qpos.device
    with wp.ScopedDevice(device), _stream(env):
        data = mjw.make_data(
            cpu,
            nworld=1,
            nconmax=max(1, (source.naconmax + source.nworld - 1) // source.nworld),
            nccdmax=max(1, (source.naccdmax + source.nworld - 1) // source.nworld),
            njmax=source.njmax,
        )
        for name in (
            "qpos",
            "qvel",
            "qacc",
            "ctrl",
            "act",
            "act_dot",
            "time",
            "qacc_warmstart",
            "qfrc_applied",
            "xfrc_applied",
            "mocap_pos",
            "mocap_quat",
        ):
            src, dst = getattr(source, name), getattr(data, name)
            if dst.size:
                value = _snapshot(src)[:1].contiguous()
                wp.copy(dst, wp.from_torch(value, dtype=dst.dtype))
    return SimpleNamespace(
        model_cpu=cpu,
        model_wp=env.model_wp,
        data_wp=data,
        device=str(device),
        num_worlds=1,
    )


class FrozenContactRollout:
    """A nominal trajectory's detached local dynamics, reusable across controls.

    ``from_env`` discovers collisions once along a nominal trajectory. Every
    later ``rollout`` is a pure Torch computation. Rebuild the cache when the
    real initial state or model changes, ordinarily once per MPC window.
    """

    @classmethod
    def from_env(cls, config, env, nominal_controls, *, options=None):
        options = options or ContactCostOptions()
        if not isinstance(nominal_controls, torch.Tensor):
            raise TypeError("nominal_controls must be a Torch tensor")
        if (
            nominal_controls.ndim != 2
            or nominal_controls.shape[0] == 0
            or nominal_controls.shape[1] != env.model_cpu.nu
        ):
            raise ValueError("nominal_controls must have shape (H, nu), with H > 0")
        instance, scratch = cls._initialize(config, env, options)
        nominal = nominal_controls.detach().to(
            device=instance.device, dtype=instance.dtype
        )
        frames, geometry = [], []
        with torch.no_grad():
            for control in nominal:
                with wp.ScopedDevice(scratch.data_wp.qpos.device), _stream(scratch):
                    wp.copy(
                        scratch.data_wp.ctrl, wp.from_torch(control[None].contiguous())
                    )
                _refresh(scratch, dynamics=True)
                if instance._actuator is None:
                    instance._actuator = _actuators(scratch)
                frame = _frame(config, scratch, options)
                frames.append(frame)
                geometry.append(_geometry(config, scratch))
                qpos, qvel = instance._step(
                    frame,
                    frame.nominal_qpos[None],
                    frame.nominal_qvel[None],
                    control[None],
                )
                with wp.ScopedDevice(scratch.data_wp.qpos.device), _stream(scratch):
                    wp.copy(scratch.data_wp.qpos, wp.from_torch(qpos.contiguous()))
                    wp.copy(scratch.data_wp.qvel, wp.from_torch(qvel.contiguous()))
            # The terminal geometry needs FK, not another collision query.
            with wp.ScopedDevice(scratch.data_wp.qpos.device), _stream(scratch):
                mjw.fwd_kinematics(scratch.model_wp, scratch.data_wp)
            geometry.append(_geometry(config, scratch))
        instance.frames, instance._geometry = tuple(frames), tuple(geometry)
        instance.horizon = len(frames)
        instance.timestep = torch.stack([frame.timestep for frame in frames])
        return instance

    @classmethod
    def from_current(cls, config, env, horizon, *, options=None):
        """Use one current-state collision snapshot over the entire horizon.

        This stricter local approximation cannot acquire future contacts. It
        still integrates state recursively and updates each frozen normal gap.
        """
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        options = options or ContactCostOptions()
        instance, scratch = cls._initialize(config, env, options)
        with torch.no_grad():
            _refresh(scratch, dynamics=True)
            instance._actuator = _actuators(scratch)
            frame = _frame(config, scratch, options)
            geometry = _geometry(config, scratch)
        instance.frames = (frame,) * horizon
        instance._geometry = (geometry,) * (horizon + 1)
        instance.horizon = horizon
        instance.timestep = frame.timestep.repeat(horizon)
        return instance

    @classmethod
    def _initialize(cls, config, env, options):
        cpu = env.model_cpu
        if (
            cpu.neq
            or (cpu.dof_frictionloss != 0).any()
            or (cpu.tendon_frictionloss != 0).any()
        ):
            raise ValueError(
                "unilateral ComFree does not support equality/frictionloss rows"
            )
        if cpu.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL:
            raise ValueError("ComFree requires pyramidal friction cones")
        if not env.model_wp.opt.run_collision_detection:
            raise ValueError(
                "building a contact cache requires enabled collision detection"
            )
        instance = cls()
        instance.options = options
        instance.nq, instance.nv, instance.nu = cpu.nq, cpu.nv, cpu.nu
        initial_qpos = _snapshot(env.data_wp.qpos)[0]
        instance.device, instance.dtype = initial_qpos.device, initial_qpos.dtype
        instance._layout = _joint_layout(cpu)
        instance._actuator = None
        joint = _object_joint(config, env, options)
        instance.object_qpos_adr = (
            None if joint is None else int(cpu.jnt_qposadr[joint])
        )
        instance.object_dof_adr = None if joint is None else int(cpu.jnt_dofadr[joint])
        return instance, _scratch(env)

    def _step(self, frame, qpos, qvel, control):
        delta = _difference(qpos, frame.nominal_qpos, self._layout)
        moment = frame.actuator_moment
        length = frame.actuator_length + torch.einsum("uv,nv->nu", moment, delta)
        velocity = torch.einsum("uv,nv->nu", moment, qvel)
        force = self._actuator.force(control, length, velocity)
        smooth = frame.smooth_residual + torch.einsum("uv,nu->nv", moment, force)
        factor = frame.mass_cholesky.expand(len(qpos), -1, -1)
        acceleration = torch.cholesky_solve(smooth[..., None], factor).squeeze(-1)
        gap = frame.gap + torch.einsum("rv,nv->nr", frame.normal_jacobian, delta)
        rows = frame.jacobian.shape[0]
        total, _, _ = contact_response_torch(
            frame.timestep,
            qvel,
            acceleration,
            frame.jacobian.expand(len(qpos), -1, -1),
            gap,
            frame.impedance_mass.expand(len(qpos), -1),
            torch.full((len(qpos),), rows, device=qpos.device, dtype=torch.int32),
            smooth,
            stiffness=self.options.comfree_stiffness,
            damping=self.options.comfree_damping,
        )
        acceleration = torch.cholesky_solve(total[..., None], factor).squeeze(-1)
        next_velocity = qvel + frame.timestep * acceleration
        return _integrate(
            qpos, next_velocity, frame.timestep, self._layout
        ), next_velocity

    def _sites(self, qpos, geometry):
        delta = _difference(qpos, geometry.qpos, self._layout)
        return geometry.site_xpos + torch.einsum(
            "kcv,nv->nkc", geometry.site_jacobian, delta
        )

    def rollout(self, actions):
        """Evaluate controls (N, H, nu), preserving gradients through all H steps.

        No Warp/MJWarp access, mutable simulator views or simulator restoration
        occur here. Batch size is independent of the environment used to build
        the cache. Controls are raw MuJoCo controls, without knot scaling.
        """
        if not isinstance(actions, torch.Tensor):
            raise TypeError("actions must be a Torch tensor")
        if (
            actions.ndim != 3
            or tuple(actions.shape[1:]) != (self.horizon, self.nu)
            or len(actions) == 0
        ):
            raise ValueError("actions must have shape (N, H, nu), with N > 0")
        actions = actions.to(device=self.device, dtype=self.dtype)
        qpos = self.frames[0].nominal_qpos.expand(len(actions), -1)
        qvel = self.frames[0].nominal_qvel.expand(len(actions), -1)
        qpositions, qvelocities = [qpos], [qvel]
        sites = [self._sites(qpos, self._geometry[0])]
        distances, displacements = [], []
        for t, frame in enumerate(self.frames):
            previous_qpos = qpos
            qpos, qvel = self._step(frame, qpos, qvel, actions[:, t])
            delta = _difference(qpos, frame.nominal_qpos, self._layout)
            distances.append(
                frame.contact_distance
                + torch.einsum("pv,nv->np", frame.contact_jacobian, delta)
            )
            if self.object_qpos_adr is None:
                displacements.append(qpos.new_zeros((len(actions), 3)))
            else:
                adr = self.object_qpos_adr
                displacements.append(
                    qpos[:, adr : adr + 3] - previous_qpos[:, adr : adr + 3]
                )
            qpositions.append(qpos)
            qvelocities.append(qvel)
            sites.append(self._sites(qpos, self._geometry[t + 1]))
        return RolloutResult(
            qpos=torch.stack(qpositions, 1),
            qvel=torch.stack(qvelocities, 1),
            site_xpos=torch.stack(sites, 1),
            penetration_distance=tuple(distances),
            object_displacement=torch.stack(displacements, 1),
            timestep=self.timestep,
            object_qpos_adr=self.object_qpos_adr,
            object_dof_adr=self.object_dof_adr,
        )
