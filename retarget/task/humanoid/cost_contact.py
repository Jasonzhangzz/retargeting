"""Contact-aware copy of the SPIDER humanoid reward.

``cost.py`` remains the original objective. This opt-in module adds point
correspondence, two penetration barriers, and differentiable ComFree dynamics.
``get_trajectory_cost`` scores a complete frozen-contact dynamics rollout without
reading Warp data or rerunning collision. The single-step helpers remain useful
for diagnostics; MPC uses the trajectory cache in env.humanoid.frozen_contact.

Derived from SPIDER commit 44717007de41cbef7565dff7ff9f4453557a2d3d.
Copyright Meta Platforms, Inc.; CC BY-NC 4.0, see licenses/SPIDER.txt.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retarget.env.humanoid.config import PhysicsConfig as Config
    from retarget.env.humanoid.physics import MJWPEnv
import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from retarget.dynamics.comfree import contact_response_torch

from .math import quat_sub


@dataclass(frozen=True)
class ContactCostOptions:
    """Weights and penetration tolerances, in metres and seconds.

    The penetration cost sums squared excess depths over robot-object contacts.
    ``object_body_id`` names the root of the object subtree. By default it is the
    last free-joint body for a humanoid_object embodiment; the first free-joint
    body is the robot. Point targets are expressed in that object's joint frame.
    """

    contact_weight: float = 1.0
    soft_penetration_weight: float = 15.0
    penetration_weight: float = 75.0
    soft_penetration_limit: float = 0.002
    penetration_limit: float = 0.01
    object_motion_weight: float = 1.0
    comfree_stiffness: float = 0.2
    comfree_damping: float = 0.001
    object_body_id: int | None = None

    def __post_init__(self):
        for name, value in vars(self).items():
            if name != "object_body_id" and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.penetration_limit < self.soft_penetration_limit:
            raise ValueError("penetration_limit must be >= soft_penetration_limit")


def _stream(env):
    device = wp.get_device(env.data_wp.qpos.device)
    if device.is_cuda:
        torch_stream = torch.cuda.current_stream(str(device))
        warp_stream = wp.get_stream(device)
        # Runtime uses a Torch ExternalStream alias of Warp's own stream.
        # Wrapping that handle a second time would unregister the original
        # stream when the temporary wrapper is collected, breaking graph capture.
        if warp_stream.cuda_stream != torch_stream.cuda_stream:
            warp_stream = wp.stream_from_torch(torch_stream)
        return wp.ScopedStream(warp_stream)
    return nullcontext()


def _snapshot(array):
    # A caller may step the simulator before backward. Never retain a mutable
    # zero-copy simulation view in the action's autograd graph.
    return wp.to_torch(array).detach().clone()


def _refresh(env, *, dynamics=False):
    with wp.ScopedDevice(env.data_wp.qpos.device), _stream(env):
        # step() caches contacts before integration, so refresh at the current q.
        mjw.fwd_position(env.model_wp, env.data_wp)
        if dynamics:
            mjw.fwd_velocity(env.model_wp, env.data_wp)
            mjw.fwd_actuation(env.model_wp, env.data_wp)
            mjw.fwd_acceleration(env.model_wp, env.data_wp)


def _object_joint(config, env, options):
    cpu = env.model_cpu
    free = [j for j in range(cpu.njnt) if cpu.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if options.object_body_id is None:
        if config.embodiment_type != "humanoid_object":
            return None
        if len(free) < 2:
            raise ValueError("humanoid_object requires an object free joint")
        return free[-1]
    body = options.object_body_id
    if not isinstance(body, int) or not 0 < body < cpu.nbody:
        raise ValueError("object_body_id must identify a non-world body")
    joints = [j for j in free if cpu.jnt_bodyid[j] == body]
    if not joints:
        raise ValueError("object_body_id must be a free-joint body")
    return joints[0]


def _body_subtree(cpu, root, device):
    members = [False] * cpu.nbody
    members[root] = True
    for body in range(root + 1, cpu.nbody):
        members[body] = members[int(cpu.body_parentid[body])]
    return torch.tensor(members, device=device)


def _rotate(quat, points):
    """Rotate vectors by MuJoCo's wxyz quaternions, with broadcasting."""
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
    xyz = quat[..., 1:]
    cross = 2 * torch.linalg.cross(xyz.expand_as(points), points, dim=-1)
    return (
        points
        + quat[..., :1] * cross
        + torch.linalg.cross(xyz.expand_as(points), cross, dim=-1)
    )


def get_contact_cost(config, env, ref, *, options=None, object_local_points=None):
    """Sum masked site-to-corresponding-point distances (metres).

    ``config.contact_site_ids`` selects the robot sites, and ``ref[3]`` is their
    contact mask. ``object_local_points`` has shape (sites, 3) or (N, sites, 3).
    If omitted, convert world targets ``ref[4]`` into the reference object's
    local frame, then move them with the current object pose. Without an object,
    the supplied targets remain world-space points, e.g. foot-floor contacts.
    Call after a position refresh, or use ``get_reward`` which refreshes for you.
    """
    options = options or ContactCostOptions()
    qpos = _snapshot(env.data_wp.qpos)
    zero = qpos.new_zeros(qpos.shape[0])
    ids = config.contact_site_ids
    if not ids:
        return zero, {"contact_cost": zero}
    sites = _snapshot(env.data_wp.site_xpos)[:, ids]
    mask = torch.as_tensor(ref[3], device=sites.device, dtype=sites.dtype)
    if mask.shape not in ((len(ids),), (len(qpos), len(ids))):
        raise ValueError("contact mask must match contact_site_ids")
    joint = _object_joint(config, env, options)
    target = torch.as_tensor(ref[4], device=sites.device, dtype=sites.dtype)
    if target.shape not in ((len(ids), 3), tuple(sites.shape)):
        raise ValueError("contact point targets must match contact_site_ids")
    if joint is not None:
        adr = int(env.model_cpu.jnt_qposadr[joint])
        if object_local_points is None:
            qref = torch.as_tensor(ref[0], device=qpos.device, dtype=qpos.dtype)
            ref_pose = qref[..., adr : adr + 7]
            inverse = torch.cat((ref_pose[..., 3:4], -ref_pose[..., 4:7]), dim=-1)
            local = _rotate(inverse.unsqueeze(-2), target - ref_pose[..., None, :3])
        else:
            local = torch.as_tensor(
                object_local_points, device=sites.device, dtype=sites.dtype
            )
            if local.shape not in ((len(ids), 3), tuple(sites.shape)):
                raise ValueError("object_local_points must match contact_site_ids")
        target = qpos[:, None, adr : adr + 3] + _rotate(
            qpos[:, None, adr + 3 : adr + 7], local.expand_as(sites)
        )
    elif object_local_points is not None:
        raise ValueError("object_local_points requires an object free joint")
    # Ignore unset targets of inactive contacts (which may contain NaNs).
    error = torch.where(mask[..., None] != 0, sites - target, 0.0)
    cost = options.contact_weight * (
        torch.linalg.vector_norm(error, dim=-1) * mask
    ).sum(-1)
    return cost, {"contact_cost": cost}


def get_penetration_cost(config, env, *, options=None, refresh=True):
    """Read live globally packed Warp collisions, excluding floor/self contacts."""
    options = options or ContactCostOptions()
    if refresh:
        _refresh(env)
    d, cpu = env.data_wp, env.model_cpu
    qpos = _snapshot(d.qpos)
    zero = qpos.new_zeros(qpos.shape[0])
    joint = _object_joint(config, env, options)
    if joint is None:
        return zero, {
            "soft_penetration_cost": zero,
            "deep_penetration_cost": zero,
            "penetration_cost": zero,
        }
    geom = _snapshot(d.contact.geom).long()
    world = _snapshot(d.contact.worldid).long()
    distance = _snapshot(d.contact.dist)
    count = _snapshot(d.nacon)[0]
    valid = torch.arange(len(distance), device=qpos.device) < count
    valid &= ((geom >= 0) & (geom < cpu.ngeom)).all(-1)
    valid &= (world >= 0) & (world < len(qpos))
    free = [j for j in range(cpu.njnt) if cpu.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    robot_body = int(cpu.jnt_bodyid[free[0]])
    object_body = int(cpu.jnt_bodyid[joint])
    if robot_body == object_body:
        raise ValueError("robot and object must have separate body roots")
    objects = _body_subtree(cpu, object_body, qpos.device)
    robots = _body_subtree(cpu, robot_body, qpos.device)
    body = torch.as_tensor(
        cpu.geom_bodyid.copy(), device=qpos.device, dtype=torch.long
    )[geom.clamp(0, cpu.ngeom - 1)]
    valid &= (robots[body[:, 0]] & objects[body[:, 1]]) | (
        objects[body[:, 0]] & robots[body[:, 1]]
    )
    depth = torch.where(valid, -distance, 0.0)
    soft = (
        options.soft_penetration_weight
        * torch.relu(depth - options.soft_penetration_limit).square()
    )
    deep = (
        options.penetration_weight
        * torch.relu(depth - options.penetration_limit).square()
    )
    index = world.clamp(0, len(qpos) - 1)
    soft_cost = zero.scatter_add(0, index, soft)
    deep_cost = zero.scatter_add(0, index, deep)
    cost = soft_cost + deep_cost
    return cost, {
        "soft_penetration_cost": soft_cost,
        "deep_penetration_cost": deep_cost,
        "penetration_cost": cost,
    }


@wp.kernel(enable_backward=False)
def _unpack_rows(
    count: wp.array(dtype=int),
    rownnz: wp.array2d(dtype=int),
    rowadr: wp.array2d(dtype=int),
    columns: wp.array3d(dtype=int),
    values: wp.array3d(dtype=float),
    dense: wp.array3d(dtype=float),
):
    world, row = wp.tid()
    if row < count[world]:
        for k in range(rownnz[world, row]):
            adr = rowadr[world, row] + k
            dense[world, row, columns[world, 0, adr]] = values[world, 0, adr]


@wp.kernel(enable_backward=False)
def _unpack_moments(
    rownnz: wp.array2d(dtype=int),
    rowadr: wp.array2d(dtype=int),
    columns: wp.array2d(dtype=int),
    values: wp.array2d(dtype=float),
    dense: wp.array3d(dtype=float),
):
    world, actuator = wp.tid()
    for k in range(rownnz[world, actuator]):
        adr = rowadr[world, actuator] + k
        dense[world, actuator, columns[world, adr]] = values[world, adr]


def _dense_dynamics(env):
    m, d, cpu = env.model_wp, env.data_wp, env.model_cpu
    index = _snapshot(m.M_elemid).long()
    values = _snapshot(d.M)[:, index.clamp_min(0)]
    lower = torch.where(index >= 0, values, 0.0)
    mass = (
        lower
        + lower.transpose(-1, -2)
        - torch.diag_embed(lower.diagonal(dim1=-2, dim2=-1))
    )
    rows = d.efc.pos.shape[1]
    with wp.ScopedDevice(d.qpos.device), _stream(env):
        if m.is_sparse:
            dense = wp.zeros((d.nworld, rows, cpu.nv), device=d.qpos.device)
            wp.launch(
                _unpack_rows,
                (d.nworld, rows),
                inputs=[
                    d.nefc,
                    d.efc.J_rownnz,
                    d.efc.J_rowadr,
                    d.efc.J_colind,
                    d.efc.J,
                    dense,
                ],
            )
            jacobian = _snapshot(dense)
        else:
            jacobian = _snapshot(d.efc.J)[:, :rows, : cpu.nv]
        moment = wp.zeros((d.nworld, cpu.nu, cpu.nv), device=d.qpos.device)
        wp.launch(
            _unpack_moments,
            (d.nworld, cpu.nu),
            inputs=[
                d.moment_rownnz,
                d.moment_rowadr,
                d.moment_colind,
                d.actuator_moment,
                moment,
            ],
        )
        moment = _snapshot(moment)
    return mass, jacobian, moment


def _actuator_force(env, action):
    """Stateless fixed/affine MuJoCo actuators, including saturation."""
    cpu, m, d = env.model_cpu, env.model_wp, env.data_wp
    if any(
        getattr(m.callback, name) is not None
        for name in ("control", "act_gain", "act_bias")
    ):
        raise ValueError("custom control/gain/bias callbacks are unsupported")
    if (cpu.actuator_dyntype != mujoco.mjtDyn.mjDYN_NONE).any():
        raise ValueError(
            "differentiable contact prediction requires stateless actuators"
        )
    if not all(
        x in (mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtGain.mjGAIN_AFFINE)
        for x in cpu.actuator_gaintype
    ):
        raise ValueError(
            "differentiable contact prediction requires fixed/affine actuator gain"
        )
    if not all(
        x in (mujoco.mjtBias.mjBIAS_NONE, mujoco.mjtBias.mjBIAS_AFFINE)
        for x in cpu.actuator_biastype
    ):
        raise ValueError(
            "differentiable contact prediction requires none/affine actuator bias"
        )
    if (
        cpu.opt.disableactuator
        or (cpu.jnt_actfrclimited != 0).any()
        or (cpu.tendon_actfrclimited != 0).any()
    ):
        raise ValueError(
            "disabled actuator groups and joint/tendon force limits are unsupported"
        )
    if cpu.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION):
        raise ValueError("differentiable contact prediction requires enabled actuation")
    if hasattr(cpu, "actuator_delay") and cpu.actuator_delay.any():
        raise ValueError("delayed actuators are unsupported")
    length, velocity = _snapshot(d.actuator_length), _snapshot(d.actuator_velocity)
    gainprm, biasprm = _snapshot(m.actuator_gainprm), _snapshot(m.actuator_biasprm)
    # Model parameters may be broadcast or randomized across worlds.
    worlds = torch.arange(len(action), device=action.device)
    gainprm = gainprm[worlds % len(gainprm)]
    biasprm = biasprm[worlds % len(biasprm)]
    affine_gain = _snapshot(m.actuator_gaintype) == int(mujoco.mjtGain.mjGAIN_AFFINE)
    gain = gainprm[..., 0] + affine_gain * (
        gainprm[..., 1] * length + gainprm[..., 2] * velocity
    )
    bias = biasprm[..., 0] + biasprm[..., 1] * length + biasprm[..., 2] * velocity
    bias = torch.where(
        _snapshot(m.actuator_biastype) == int(mujoco.mjtBias.mjBIAS_AFFINE), bias, 0.0
    )
    if not cpu.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL):
        limits = _snapshot(m.actuator_ctrlrange)
        limits = limits[worlds % len(limits)]
        clamped = torch.maximum(torch.minimum(action, limits[..., 1]), limits[..., 0])
        action = torch.where(_snapshot(m.actuator_ctrllimited), clamped, action)
    force = gain * action + bias
    limits = _snapshot(m.actuator_forcerange)
    limits = limits[worlds % len(limits)]
    clamped = torch.maximum(torch.minimum(force, limits[..., 1]), limits[..., 0])
    return torch.where(_snapshot(m.actuator_forcelimited), clamped, force)


def object_displacement_cost(
    config, env, action, target_displacement, *, options=None, refresh=True
):
    """Differentiable one-step object translation error, in squared metres.

    ``action`` is a Torch MuJoCo control vector (nu,) or (N, nu), with no knot
    scaling applied. The target is a world-space displacement (3,) or (N, 3).
    Returns per-world cost and ``predicted_displacement``. Collision topology,
    J, M and passive forces are frozen at the refreshed state. Semi-implicit
    Euler uses the simulator timestep, not the MPC control interval.
    """
    options = options or ContactCostOptions()
    joint = _object_joint(config, env, options)
    if joint is None:
        raise ValueError("object displacement prediction requires an object free joint")
    cpu, d, m = env.model_cpu, env.data_wp, env.model_wp
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
    if refresh:
        _refresh(env, dynamics=True)
    qvel = _snapshot(d.qvel)
    if not isinstance(action, torch.Tensor):
        raise TypeError("action must be a Torch tensor")
    action = action.to(device=qvel.device, dtype=qvel.dtype)
    if action.shape == (cpu.nu,):
        action = action.expand(len(qvel), -1)
    if action.shape != (len(qvel), cpu.nu):
        raise ValueError("action must have shape (nu,) or (N, nu)")
    target = torch.as_tensor(target_displacement, device=qvel.device, dtype=qvel.dtype)
    if target.shape not in ((3,), (len(qvel), 3)):
        raise ValueError("target_displacement must have shape (3,) or (N, 3)")
    mass, jacobian, moment = _dense_dynamics(env)
    force = _actuator_force(env, action)
    # Replace actuator forces by their delta, preserving actuator-level gravity
    # compensation as well as gravity, passive forces and externally applied forces.
    smooth = _snapshot(d.qfrc_smooth) + torch.einsum(
        "wuv,wu->wv", moment, force - _snapshot(d.actuator_force)
    )
    acceleration = torch.linalg.solve(mass, smooth.unsqueeze(-1)).squeeze(-1)
    dt = _snapshot(m.opt.timestep)
    total, row_force, constraint = contact_response_torch(
        dt,
        qvel,
        acceleration,
        jacobian,
        _snapshot(d.efc.pos),
        _snapshot(d.efc.D)[:, : jacobian.shape[1]],
        _snapshot(d.nefc),
        smooth,
        stiffness=options.comfree_stiffness,
        damping=options.comfree_damping,
    )
    acceleration = torch.linalg.solve(mass, total.unsqueeze(-1)).squeeze(-1)
    dt = dt[torch.arange(len(qvel), device=qvel.device) % len(dt), None]
    adr = int(cpu.jnt_dofadr[joint])
    displacement = dt * (qvel[:, adr : adr + 3] + dt * acceleration[:, adr : adr + 3])
    cost = options.object_motion_weight * (displacement - target).square().sum(-1)
    return cost, {
        "object_motion_cost": cost,
        "predicted_displacement": displacement,
        "comfree_row_force": row_force,
        "comfree_constraint_force": constraint,
    }


def get_trajectory_cost(
    config,
    trajectory,
    ref,
    *,
    options=None,
    object_local_points=None,
    target_displacement=None,
):
    """Score a whole dynamics-only rollout without any simulator calls.

    ``trajectory`` is the result of ``FrozenContactRollout.rollout``. References
    keep the original five-tuple layout with a leading horizon dimension H.
    Costs propagate gradients through all earlier actions, including contact
    point and penetration costs. Terminal scaling and horizon averaging follow
    the original SPIDER convention. Component diagnostics have shape (N, H).
    """
    options = options or ContactCostOptions()
    qpos, qvel = trajectory.qpos[:, 1:], trajectory.qvel[:, 1:]
    if qpos.ndim != 3 or qvel.ndim != 3 or qpos.shape[:2] != qvel.shape[:2]:
        raise ValueError("trajectory states must have shape (N, H+1, nq/nv)")
    worlds, horizon, nq = qpos.shape
    if not worlds or not horizon:
        raise ValueError("trajectory requires at least one world and one step")
    refs = tuple(torch.as_tensor(r, device=qpos.device, dtype=qpos.dtype) for r in ref)
    if len(refs) != 5:
        raise ValueError("reference must be the original five-tuple")
    qref, vref, _, mask, points = refs
    if qref.shape != (horizon, nq) or vref.shape != (horizon, qvel.shape[-1]):
        raise ValueError("reference state shapes must match the trajectory horizon")
    delta = _diff_qpos(
        config,
        qpos.reshape(worlds * horizon, nq),
        qref.unsqueeze(0).expand(worlds, -1, -1).reshape(worlds * horizon, nq),
    ).reshape(worlds, horizon, config.nv)
    qpos_dist = torch.linalg.vector_norm(delta * _weight_diff_qpos(config), dim=-1)
    qvel_dist = torch.linalg.vector_norm(qvel - vref, dim=-1)
    contact = qpos.new_zeros((worlds, horizon))
    ids = config.contact_site_ids
    object_adr = trajectory.object_qpos_adr
    if ids and options.contact_weight > 0:
        sites = trajectory.site_xpos[:, 1:]
        if sites.shape != (worlds, horizon, len(ids), 3):
            raise ValueError("trajectory sites must match contact_site_ids")
        if mask.shape != (horizon, len(ids)) or points.shape != (horizon, len(ids), 3):
            raise ValueError("reference contact shapes must match contact_site_ids")
        active = mask[..., None] != 0
        # Mask unset/NaN targets before the quaternion transforms: masking only
        # their final distances would still introduce NaNs in rotation gradients.
        target = torch.where(active, points, 0.0)
        if object_adr is not None:
            if object_local_points is None:
                pose = qref[:, object_adr : object_adr + 7]
                inverse = torch.cat((pose[:, 3:4], -pose[:, 4:7]), dim=-1)
                local = _rotate(inverse[:, None], target - pose[:, None, :3])
            else:
                local = torch.as_tensor(
                    object_local_points, device=qpos.device, dtype=qpos.dtype
                )
                if local.shape not in (
                    (len(ids), 3),
                    (horizon, len(ids), 3),
                    (worlds, horizon, len(ids), 3),
                ):
                    raise ValueError("object_local_points must match sites and horizon")
                local = torch.where(active, local, 0.0)
            current = qpos[..., object_adr : object_adr + 7]
            target = current[..., None, :3] + _rotate(
                current[..., None, 3:], local.expand_as(sites)
            )
        elif object_local_points is not None:
            raise ValueError("object_local_points requires an object free joint")
        error = torch.where(active, sites - target, 0.0)
        contact = options.contact_weight * (
            torch.linalg.vector_norm(error, dim=-1) * mask
        ).sum(-1)

    if len(trajectory.penetration_distance) != horizon:
        raise ValueError("cached penetration distances must cover the entire horizon")
    soft, deep = [], []
    for distance in trajectory.penetration_distance:
        if distance.ndim != 2 or distance.shape[0] != worlds:
            raise ValueError(
                "cached penetration distances must have shape (N, contacts)"
            )
        soft.append(
            options.soft_penetration_weight
            * torch.relu(-distance - options.soft_penetration_limit).square().sum(-1)
        )
        deep.append(
            options.penetration_weight
            * torch.relu(-distance - options.penetration_limit).square().sum(-1)
        )
    soft, deep = torch.stack(soft, dim=1), torch.stack(deep, dim=1)
    penetration = soft + deep
    displacement = trajectory.object_displacement
    if displacement.shape != (worlds, horizon, 3):
        raise ValueError("object displacements must have shape (N, H, 3)")
    motion = qpos.new_zeros((worlds, horizon))
    if object_adr is not None and options.object_motion_weight > 0:
        if target_displacement is None:
            # Humanoid object layout uses a final free joint (nq=nv+2).
            # Velocity address can differ from its qpos address by earlier quats.
            # The engine carries the exact object velocity address as metadata.
            vadr = trajectory.object_dof_adr
            target = vref[:, vadr : vadr + 3] * trajectory.timestep[:, None]
        else:
            target = torch.as_tensor(
                target_displacement, device=qpos.device, dtype=qpos.dtype
            )
            if target.shape not in ((horizon, 3), (worlds, horizon, 3)):
                raise ValueError(
                    "target_displacement must have shape (H, 3) or (N, H, 3)"
                )
        motion = options.object_motion_weight * (displacement - target).square().sum(-1)
    elif target_displacement is not None and object_adr is None:
        raise ValueError("target_displacement requires an object free joint")
    step_cost = (
        qpos_dist + config.vel_rew_scale * qvel_dist + contact + penetration + motion
    )
    scale = qpos.new_ones(horizon)
    scale[-1] = config.terminal_rew_scale
    step_cost = step_cost * scale
    total = step_cost.mean(dim=1)
    return total, {
        "cost": total,
        "step_cost": step_cost,
        "qpos_dist": qpos_dist,
        "qvel_dist": qvel_dist,
        "contact_cost": contact,
        "soft_penetration_cost": soft,
        "deep_penetration_cost": deep,
        "penetration_cost": penetration,
        "object_motion_cost": motion,
        "predicted_displacement": displacement,
    }


def _weight_diff_qpos(config: Config) -> torch.Tensor:
    w = torch.ones(config.nv, device=config.device)
    if config.embodiment_type == "bimanual":
        half_dof = int(config.nu // 2)
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        w[6:half_dof] = config.joint_rew_scale
        w[half_dof : half_dof + 3] = config.base_pos_rew_scale
        w[half_dof + 3 : half_dof + 6] = config.base_rot_rew_scale
        w[half_dof + 6 : config.nu] = config.joint_rew_scale
        # object: weights live in nv-space (6-dim per freejoint), regardless of nq_obj
        w[-12:-9] = config.pos_rew_scale
        w[-9:-6] = config.rot_rew_scale
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    elif config.embodiment_type in ["right", "left"]:
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        w[6 : config.nu] = config.joint_rew_scale
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    elif config.embodiment_type in ["humanoid"]:  # humanoid robot
        # robot pos and rot
        w[:3] = config.pos_rew_scale
        w[3:6] = config.rot_rew_scale
        # robot joint
        w[6:] = config.joint_rew_scale
    elif config.embodiment_type in ["humanoid_object"]:
        # robot pos and rot
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        # robot joint
        w[6:-6] = config.joint_rew_scale
        # object pos and rot
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return w


def _diff_qpos(
    config: Config, qpos_sim: torch.Tensor, qpos_ref: torch.Tensor
) -> torch.Tensor:
    """Compute the difference between qpos_sim and qpos_ref
    TODO: replace with mujoco built-in function, not sure how to call warp internal function yet.
    """
    batch_size = qpos_sim.shape[0]
    qpos_diff = torch.zeros((batch_size, config.nv), device=config.device)
    if config.embodiment_type == "bimanual":
        if config.nq_obj == 12:
            qpos_diff[:, :-12] = qpos_sim[:, :-12] - qpos_ref[:, :-12]
            qpos_diff[:, -12:-9] = qpos_sim[:, -12:-9] - qpos_ref[:, -12:-9]
            qpos_diff[:, -9:-6] = qpos_sim[:, -9:-6] - qpos_ref[:, -9:-6]
            qpos_diff[:, -6:-3] = qpos_sim[:, -6:-3] - qpos_ref[:, -6:-3]
            qpos_diff[:, -3:] = qpos_sim[:, -3:] - qpos_ref[:, -3:]
            return qpos_diff
        # joint
        qpos_diff[:, :-12] = qpos_sim[:, :-14] - qpos_ref[:, :-14]
        # position
        qpos_diff[:, -12:-9] = qpos_sim[:, -14:-11] - qpos_ref[:, -14:-11]
        qpos_diff[:, -6:-3] = qpos_sim[:, -7:-4] - qpos_ref[:, -7:-4]
        # rotation
        qpos_diff[:, -9:-6] = quat_sub(qpos_sim[:, -11:-7], qpos_ref[:, -11:-7])
        qpos_diff[:, -3:] = quat_sub(qpos_sim[:, -4:], qpos_ref[:, -4:])
    elif config.embodiment_type in ["right", "left"]:
        if config.nq_obj == 6:
            qpos_diff[:, :-6] = qpos_sim[:, :-6] - qpos_ref[:, :-6]
            qpos_diff[:, -6:-3] = qpos_sim[:, -6:-3] - qpos_ref[:, -6:-3]
            qpos_diff[:, -3:] = qpos_sim[:, -3:] - qpos_ref[:, -3:]
            return qpos_diff
        # joint
        qpos_diff[:, :-6] = qpos_sim[:, :-7] - qpos_ref[:, :-7]
        # position
        qpos_diff[:, -6:-3] = qpos_sim[:, -7:-4] - qpos_ref[:, -7:-4]
        # rotation
        qpos_diff[:, -3:] = quat_sub(qpos_sim[:, -4:], qpos_ref[:, -4:])
    elif config.embodiment_type in ["humanoid"]:
        # joint
        qpos_diff[:, 6:] = qpos_sim[:, 7:] - qpos_ref[:, 7:]
        # position
        qpos_diff[:, :3] = qpos_sim[:, :3] - qpos_ref[:, :3]
        # rotation
        qpos_diff[:, 3:6] = quat_sub(qpos_sim[:, 3:7], qpos_ref[:, 3:7])
    elif config.embodiment_type in ["humanoid_object"]:
        qpos_humanoid = qpos_sim[:, :-7]
        qpos_object = qpos_sim[:, -7:]
        qpos_ref_humanoid = qpos_ref[:, :-7]
        qpos_ref_object = qpos_ref[:, -7:]
        # position
        qpos_diff[:, :3] = qpos_humanoid[:, :3] - qpos_ref_humanoid[:, :3]
        # rotation
        qpos_diff[:, 3:6] = quat_sub(qpos_humanoid[:, 3:7], qpos_ref_humanoid[:, 3:7])
        # joint
        qpos_diff[:, 6:-6] = qpos_humanoid[:, 7:] - qpos_ref_humanoid[:, 7:]
        # object
        qpos_diff[:, -6:-3] = qpos_object[:, :3] - qpos_ref_object[:, :3]
        qpos_diff[:, -3:] = quat_sub(qpos_object[:, 3:7], qpos_ref_object[:, 3:7])
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return qpos_diff


def get_reward(
    config: Config,
    env: MJWPEnv,
    ref: tuple[torch.Tensor, ...],
    *,
    action: torch.Tensor | None = None,
    options: ContactCostOptions | None = None,
    object_local_points: torch.Tensor | None = None,
    target_displacement: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Original tracking reward minus contact, penetration and motion costs.

    The reference tuple is (qpos, qvel, ctrl, contact_mask, contact_points).
    Motion defaults to the reference object linear velocity times sim_dt; pass
    ``target_displacement`` to supply a measured/desired next-frame translation.
    Pass an explicit ``action.requires_grad_(True)`` to differentiate the motion
    term. Omitting action evaluates the current Warp control, without a graph
    through the original simulator. The original contact scale is superseded by
    ``options.contact_weight`` (default 1) in this module.
    """
    options = options or ContactCostOptions()
    joint = _object_joint(config, env, options)
    motion_enabled = joint is not None and options.object_motion_weight > 0
    _refresh(env, dynamics=motion_enabled)
    qpos_ref, qvel_ref, *_ = ref
    qpos_sim = _snapshot(env.data_wp.qpos)
    qvel_sim = _snapshot(env.data_wp.qvel)

    # weighted qpos tracking
    qpos_diff = _diff_qpos(
        config, qpos_sim, qpos_ref.unsqueeze(0).repeat(qpos_sim.shape[0], 1)
    )
    qpos_weight = _weight_diff_qpos(config)
    delta_qpos = qpos_diff * qpos_weight
    qpos_dist = torch.norm(delta_qpos, p=2, dim=1)
    qvel_dist = torch.norm(qvel_sim - qvel_ref, p=2, dim=1)

    qpos_rew = -qpos_dist * 1.0
    qvel_rew = -config.vel_rew_scale * qvel_dist * 1.0

    contact, contact_info = get_contact_cost(
        config, env, ref, options=options, object_local_points=object_local_points
    )
    penetration, penetration_info = get_penetration_cost(
        config, env, options=options, refresh=False
    )
    motion = qpos_sim.new_zeros(qpos_sim.shape[0])
    motion_info = {
        "object_motion_cost": motion,
        "predicted_displacement": qpos_sim.new_zeros((len(qpos_sim), 3)),
    }
    if motion_enabled:
        if action is None:
            action = _snapshot(env.data_wp.ctrl)
        if target_displacement is None:
            dt = _snapshot(env.model_wp.opt.timestep)
            dt = dt[torch.arange(len(qpos_sim), device=qpos_sim.device) % len(dt), None]
            adr = int(env.model_cpu.jnt_dofadr[joint])
            target_displacement = qvel_ref[..., adr : adr + 3] * dt
        motion, prediction = object_displacement_cost(
            config, env, action, target_displacement, options=options, refresh=False
        )
        # Keep reward diagnostics compatible with the original rollout's
        # per-world temporal reduction. Detailed forces remain in the helper.
        motion_info = {k: prediction[k] for k in motion_info}
    reward = qpos_rew + qvel_rew - contact - penetration - motion

    info = {
        "qpos_dist": qpos_dist,
        "qvel_dist": qvel_dist,
        "qpos_rew": qpos_rew,
        "qvel_rew": qvel_rew,
        **contact_info,
        **penetration_info,
        **motion_info,
    }
    return reward, info


def get_terminal_reward(
    config: Config,
    env: MJWPEnv,
    ref_slice: tuple[torch.Tensor, ...],
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """Scale the contact-aware reward using the original terminal convention."""
    rew, info = get_reward(config, env, ref_slice, **kwargs)
    terminal_rew = config.terminal_rew_scale * rew
    return terminal_rew, info


def get_terminate(
    config: Config, env: MJWPEnv, ref_slice: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    # compute object position and orientation error, compare to thereshold
    qpos_sim = wp.to_torch(env.data_wp.qpos)
    qpos_ref, *_ = ref_slice
    if config.embodiment_type == "bimanual":
        if config.nq_obj == 12:
            right_obj_pos = qpos_sim[:, -12:-9]
            right_obj_pos_ref = qpos_ref[-12:-9].unsqueeze(0)
            right_obj_pos_error = torch.norm(
                right_obj_pos - right_obj_pos_ref, p=2, dim=1
            )
            right_obj_rot = qpos_sim[:, -9:-6]
            right_obj_rot_ref = qpos_ref[-9:-6].unsqueeze(0)
            right_obj_rot_error = torch.norm(
                right_obj_rot - right_obj_rot_ref, p=2, dim=1
            )
            left_obj_pos = qpos_sim[:, -6:-3]
            left_obj_pos_ref = qpos_ref[-6:-3].unsqueeze(0)
            left_obj_pos_error = torch.norm(left_obj_pos - left_obj_pos_ref, p=2, dim=1)
            left_obj_rot = qpos_sim[:, -3:]
            left_obj_rot_ref = qpos_ref[-3:].unsqueeze(0)
            left_obj_rot_error = torch.norm(left_obj_rot - left_obj_rot_ref, p=2, dim=1)
            if torch.all(right_obj_pos_ref.abs() < 1e-4):
                right_obj_pos_error *= 0.0
                right_obj_rot_error *= 0.0
            if torch.all(left_obj_pos_ref.abs() < 1e-4):
                left_obj_pos_error *= 0.0
                left_obj_rot_error *= 0.0
            terminate = (
                (left_obj_pos_error > config.object_pos_threshold)
                | (right_obj_pos_error > config.object_pos_threshold)
                | (left_obj_rot_error > config.object_rot_threshold)
                | (right_obj_rot_error > config.object_rot_threshold)
            )
            return terminate
        left_obj_pos = qpos_sim[:, -14:-11]
        left_obj_pos_ref = qpos_ref[-14:-11].unsqueeze(0)
        left_obj_pos_error = torch.norm(left_obj_pos - left_obj_pos_ref, p=2, dim=1)
        left_obj_quat = qpos_sim[:, -11:-7]
        left_obj_quat_ref = qpos_ref[-11:-7].unsqueeze(0)
        left_obj_quat_error = torch.norm(
            quat_sub(left_obj_quat, left_obj_quat_ref.repeat(qpos_sim.shape[0], 1)),
            p=2,
            dim=1,
        )
        right_obj_pos = qpos_sim[:, -7:-4]
        right_obj_pos_ref = qpos_ref[-7:-4].unsqueeze(0)
        right_obj_pos_error = torch.norm(right_obj_pos - right_obj_pos_ref, p=2, dim=1)
        right_obj_quat = qpos_sim[:, -4:]
        right_obj_quat_ref = qpos_ref[-4:].unsqueeze(0)
        right_obj_quat_error = torch.norm(
            quat_sub(right_obj_quat, right_obj_quat_ref.repeat(qpos_sim.shape[0], 1)),
            p=2,
            dim=1,
        )
        # special case: only have left object
        if torch.all(right_obj_pos_ref.abs() < 1e-4):
            right_obj_pos_error *= 0.0
            right_obj_quat_error *= 0.0
        # special case: only have right object
        if torch.all(left_obj_pos_ref.abs() < 1e-4):
            left_obj_pos_error *= 0.0
            left_obj_quat_error *= 0.0
        terminate = (
            (left_obj_pos_error > config.object_pos_threshold)
            | (right_obj_pos_error > config.object_pos_threshold)
            | (left_obj_quat_error > config.object_rot_threshold)
            | (right_obj_quat_error > config.object_rot_threshold)
        )
    elif config.embodiment_type in ["right", "left"]:
        if config.nq_obj == 6:
            obj_pos = qpos_sim[:, -6:-3]
            obj_pos_ref = qpos_ref[-6:-3].unsqueeze(0)
            obj_pos_error = torch.norm(obj_pos - obj_pos_ref, p=2, dim=1)
            obj_rot = qpos_sim[:, -3:]
            obj_rot_ref = qpos_ref[-3:].unsqueeze(0)
            obj_rot_error = torch.norm(obj_rot - obj_rot_ref, p=2, dim=1)
            terminate = (obj_pos_error > config.object_pos_threshold) | (
                obj_rot_error > config.object_rot_threshold
            )
            return terminate
        obj_pos = qpos_sim[:, -7:-4]
        obj_pos_ref = qpos_ref[-7:-4].unsqueeze(0)
        obj_pos_error = torch.norm(obj_pos - obj_pos_ref, p=2, dim=1)
        obj_quat = qpos_sim[:, -4:]
        obj_quat_ref = qpos_ref[-4:].unsqueeze(0)
        obj_quat_error = torch.norm(
            quat_sub(obj_quat, obj_quat_ref.repeat(qpos_sim.shape[0], 1)), p=2, dim=1
        )
        terminate = (obj_pos_error > config.object_pos_threshold) | (
            obj_quat_error > config.object_rot_threshold
        )
    elif config.embodiment_type in ["humanoid", "humanoid_object"]:
        base_pos = qpos_sim[:, :3]
        base_pos_ref = qpos_ref[:3].unsqueeze(0)
        base_pos_error = torch.norm(base_pos - base_pos_ref, p=2, dim=1)
        base_quat = qpos_sim[:, 3:7]
        base_quat_ref = qpos_ref[3:7].unsqueeze(0)
        base_quat_error = torch.norm(
            quat_sub(base_quat, base_quat_ref.repeat(qpos_sim.shape[0], 1)), p=2, dim=1
        )
        terminate = (base_pos_error > config.base_pos_threshold) | (
            base_quat_error > config.base_rot_threshold
        )
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return terminate
