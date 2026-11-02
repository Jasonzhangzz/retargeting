"""SCSP with one palm projection pool and real-contact robot optimization.

The robot has four fixed sites: two palms and two soles. Only the two object
contact choices are discrete (11 x 11); contact forces use autograd optimization.
IPOPT refreshes actual collision geometry and Jacobians at its current rollout.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import mujoco_warp as mjw
import warp as wp

from retarget.dynamics.comfree.response import _contact_response_unpadded
from retarget.solver.scsp import ObjectEvaluation, RobotEvaluation, SCSPProblem
from retarget.task.humanoid.cost_contact import ContactCostOptions, _rotate, _snapshot, _stream
from retarget.task.humanoid.cost_contact1 import get_trajectory_cost, _smooth_norm

from .contact_projection import build_contact_candidates
from .frozen_contact import FrozenContactRollout, _advance_quaternion, _rotation_vector


@dataclass(frozen=True)
class SCSPContactOptions:
    hand_sites: tuple[str, ...] = ("left_palm", "right_palm")
    foot_sites: tuple[str, ...] = ("left_foot", "right_foot")
    neighbors: int = 10
    candidate_radius: float = 0.03
    point_radius: float = 1e-4
    friction: float = 0.8
    max_contact_force: float = 10.0
    force_regularization: float = 1e-4
    force_smoothness: float = 1e-4
    action_regularization: float = 1e-4
    foot_tracking_weight: float = 1.0
    approach_weight: float = 1.0
    operation_intervals: tuple[tuple[float, float], ...] | None = None
    position_scale: float = 0.03
    force_scale: float = 10.0
    action_trust_radius: float = 0.1
    comfree_stiffness: float = 0.2
    comfree_damping: float = 0.001

    def __post_init__(self):
        for name in ("hand_sites", "foot_sites"):
            sites = getattr(self, name)
            if not isinstance(sites, (list, tuple)) or len(sites) != 2 or any(
                not isinstance(s, str) or not s for s in sites
            ):
                raise ValueError(f"scsp_contact.{name} must name exactly two sites")
            object.__setattr__(self, name, tuple(sites))
        if len(set(self.hand_sites + self.foot_sites)) != 4:
            raise ValueError("scsp_contact.hand_sites and foot_sites must name four distinct palm/sole sites")
        if isinstance(self.neighbors, bool) or not isinstance(self.neighbors, int) or self.neighbors != 10:
            raise ValueError("SCSP requires 10 neighbors plus the initial point per hand")
        if self.operation_intervals is not None:
            if not isinstance(self.operation_intervals, (tuple, list)):
                raise ValueError("scsp_contact.operation_intervals must contain [start, end] seconds")
            intervals = []
            for interval in self.operation_intervals:
                if not isinstance(interval, (tuple, list)) or len(interval) != 2:
                    raise ValueError("each operation interval must contain [start, end] seconds")
                start, end = interval
                if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in interval) or not 0 <= start < end:
                    raise ValueError("operation intervals require 0 <= start < end")
                intervals.append((float(start), float(end)))
            object.__setattr__(self, "operation_intervals", tuple(intervals))
        positive = {"candidate_radius", "point_radius", "max_contact_force", "position_scale", "force_scale",
                    "action_trust_radius", "comfree_stiffness"}
        for name, value in vars(self).items():
            if name in ("hand_sites", "foot_sites", "neighbors", "operation_intervals"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"scsp_contact.{name} has an invalid value")


def contact_options(values=None):
    if isinstance(values, SCSPContactOptions):
        return values
    if values is not None and not isinstance(values, dict):
        raise ValueError("environment.scsp_contact must be a mapping")
    try:
        return SCSPContactOptions(**(values or {}))
    except TypeError as exc:
        raise ValueError(f"Invalid environment.scsp_contact: {exc}") from exc


def _basis(normals):
    """Local frame [inward normal, tangent1, tangent2], columns in world/local."""
    axis = torch.zeros_like(normals)
    axis[..., 0] = 1
    alternative = torch.zeros_like(normals)
    alternative[..., 1] = 1
    axis = torch.where((normals[..., :1].abs() > 0.9), alternative, axis)
    t1 = F.normalize(torch.linalg.cross(normals, axis, dim=-1), dim=-1)
    t2 = torch.linalg.cross(normals, t1, dim=-1)
    return torch.stack((-normals, t1, t2), -1)


def _object_points(pose, local):
    """World point and exact freejoint point Jacobian in MuJoCo nv coordinates."""
    lever = _rotate(pose[:, None, 3:], local)
    eye = torch.eye(3, device=pose.device, dtype=pose.dtype)
    axes = _rotate(pose[:, None, 3:], eye.expand(len(pose), -1, -1))
    angular = torch.linalg.cross(axes[:, None], lever[:, :, None], dim=-1).transpose(-1, -2)
    jacobian = torch.cat((eye.expand(len(pose), local.shape[1], -1, -1), angular), -1)
    return pose[:, None, :3] + lever, jacobian


class HumanoidSCSP:
    """One initial pair of palm projections: 121 possible contact combinations."""

    def __init__(self, task, *, options=None, seed=0, operation_mask=None):
        self.task, self.options = task, contact_options(options)
        self.config = c = copy.copy(task.config)
        if c.embodiment_type != "humanoid_object":
            raise ValueError("SCSP requires a humanoid_object embodiment")
        self.hand_ids = tuple(task.env.model_cpu.site(name).id for name in self.options.hand_sites)
        self.foot_ids = tuple(task.env.model_cpu.site(name).id for name in self.options.foot_sites)
        self.contact_ids = self.hand_ids + self.foot_ids
        c.contact_site_ids = list(self.contact_ids)
        start, self.horizon = task.first_step, c.horizon_steps
        self.refs = tuple(r[start + 1:start + self.horizon + 1] for r in task.refs)
        self.candidates = build_contact_candidates(
            c, task.env, task.refs[0][start:start + 1], self.hand_ids,
            radius=self.options.candidate_radius, neighbors=self.options.neighbors, seed=seed,
            use_hand_surface=False,
        )
        self.reference_sites = self._reference_site_positions(task.env, task.refs[0][start:start + self.horizon])
        # Nominal MJWarp states give bounded local frames. No integration-defect
        # correction is added to either SCSP subproblem's ComFree dynamics.
        self.cache = FrozenContactRollout.from_env(
            c, task.env, task.nominal,
            options=ContactCostOptions(comfree_stiffness=self.options.comfree_stiffness,
                                       comfree_damping=self.options.comfree_damping),
            anchor_to_mjwarp=True,
        )
        self.qadr = self.candidates.object_qpos_adr
        self.vadr = self.candidates.object_dof_adr
        if self.qadr != c.nq - 7 or self.vadr != c.nv - 6:
            raise ValueError("SCSP requires the final free joint to be the object")
        self.hands = len(self.hand_ids)
        self.knots = task.knots
        self.device, self.dtype = self.cache.device, self.cache.dtype
        from .real_contact import RealContactRollout

        self.real_rollout = RealContactRollout(c, task.env, options=self.cache.options)
        self.operation_mask = self._operation_mask(operation_mask)
        self.contact_position_mask = torch.zeros((self.horizon, self.hands), device=self.device, dtype=torch.bool)
        self.consensus_mask = torch.ones((self.horizon, self.hands, 6), device=self.device, dtype=torch.bool)
        self.consensus_mask[..., :3] = False
        self.object_rows = []
        self.object_inverses = []
        for frame in self.cache.frames:
            object_part = frame.jacobian[:, self.vadr:self.vadr+6].abs().sum(-1) > 1e-9
            robot_part = frame.jacobian[:, :self.vadr].abs().sum(-1) > 1e-9
            self.object_rows.append(object_part & ~robot_part)
            self.object_inverses.append(frame.mass_inverse[self.vadr:self.vadr+6, self.vadr:self.vadr+6])
        cpu = task.env.model_cpu
        if any(bool((frame.actuator_moment[:, self.vadr:].abs() > 1e-9).any()) for frame in self.cache.frames):
            raise ValueError("SCSP object must be unactuated")

    def _operation_mask(self, mask):
        if mask is not None:
            mask = torch.as_tensor(mask, device=self.device)
            if mask.dtype != torch.bool or mask.shape not in ((self.horizon,), (self.horizon, self.hands)):
                raise ValueError("operation_mask must be Boolean (H,) or (H, 2)")
            if mask.ndim == 1:
                mask = mask[:, None].expand(-1, self.hands)
            return mask.detach().clone()
        intervals = self.options.operation_intervals
        if intervals is None:
            return torch.ones((self.horizon, self.hands), device=self.device, dtype=torch.bool)
        dt = self.cache.timestep
        times = self.task.first_step * dt[0] + dt.cumsum(0) - dt
        active = torch.zeros(self.horizon, device=self.device, dtype=torch.bool)
        for start, end in intervals:
            active |= (times >= start) & (times < end)
        return active[:, None].expand(-1, self.hands).clone()

    def prepare_iteration(self, actions):
        """Fix this ADMM round's contact mode before either subproblem runs.

        Contact equality applies only to an operating hand with a real contact.
        Inactive position duals stay zero. Force agreement remains enabled: an
        absent real contact must never acquire an upper-layer virtual force.
        """
        with torch.no_grad():
            actual = self.real_rollout.rollout(self.task.controls(actions[None]))
            consensus = self._consensus(actual.site_xpos[:, :, :2], actual.contact_force[:, :, :2])[0]
            self.contact_position_mask = actual.active_mask[0, :, :2] & self.operation_mask
            self.consensus_mask = torch.ones_like(consensus, dtype=torch.bool)
            self.consensus_mask[..., :3] = self.contact_position_mask[..., None]
        return consensus, self.consensus_mask.clone()

    def _reference_site_positions(self, env, qpos):
        with torch.no_grad(), wp.ScopedDevice(env.data_wp.qpos.device), _stream(env):
            data = mjw.make_data(env.model_cpu, nworld=len(qpos), nconmax=1, njmax=1)
            wp.to_torch(data.qpos).copy_(qpos)
            mjw.kinematics(env.model_wp, data)
            return _snapshot(data.site_xpos)[:, list(self.contact_ids)]

    def select(self, indices):
        # There is one fixed object-local candidate pool per hand, seeded only
        # once at the beginning of this MPC window, not once per time step.
        n = len(indices)
        gather = indices[:, None, :, None, None].expand(n, self.horizon, self.hands, 1, 3)
        points = self.candidates.local_points[None].expand(n, self.horizon, -1, -1, -1).gather(3, gather).squeeze(3)
        normals = self.candidates.local_normals[None].expand(n, self.horizon, -1, -1, -1).gather(3, gather).squeeze(3)
        return points, normals

    def force_project(self, indices, parameters):
        force = parameters.reshape(-1, self.knots, self.hands, 3)
        normal = force[..., :1].clamp(0, self.options.max_contact_force)
        tangent = force[..., 1:]
        factor = (self.options.friction * normal / tangent.abs().sum(-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
        force = torch.cat((normal, tangent * factor), -1)
        magnitude = torch.linalg.vector_norm(force, dim=-1, keepdim=True).clamp_min(1e-12)
        return (force * (self.options.max_contact_force / magnitude).clamp(max=1)).flatten(1)

    def dense_forces(self, parameters):
        values = parameters.reshape(-1, self.knots, self.hands*3).transpose(1, 2)
        return F.interpolate(values, size=self.horizon, mode="linear", align_corners=True).transpose(1, 2).reshape(-1, self.horizon, self.hands, 3)

    def _world_force(self, pose, local_normals, force):
        local = torch.einsum("nkij,nkj->nki", _basis(local_normals), force)
        return _rotate(pose[:, None, 3:], local)

    def _consensus(self, points, forces):
        return torch.cat((points / self.options.position_scale, forces / self.options.force_scale), -1)

    def _object_step(self, t, pose, velocity, local, normals, force):
        frame, rows = self.cache.frames[t], self.object_rows[t]
        points, point_j = _object_points(pose, local)
        world_force = self._world_force(pose, normals, force)
        smooth = frame.smooth_residual[self.vadr:] + torch.einsum("nkcv,nkc->nv", point_j, world_force)
        inverse = self.object_inverses[t]
        acceleration = torch.einsum("ij,nj->ni", inverse, smooth)
        delta = torch.cat((pose[:, :3] - frame.nominal_qpos[self.qadr:self.qadr+3],
                           _rotation_vector(pose[:, 3:], frame.nominal_qpos[self.qadr+3:])), -1)
        jacobian = frame.jacobian[rows, self.vadr:]
        gap = frame.gap[rows] + torch.einsum("rv,nv->nr", frame.normal_jacobian[rows, self.vadr:], delta)
        total, _, _ = _contact_response_unpadded(
            frame.timestep, velocity, acceleration, jacobian.expand(len(pose), -1, -1), gap,
            frame.impedance_mass[rows].expand(len(pose), -1), smooth,
            stiffness=self.options.comfree_stiffness, damping=self.options.comfree_damping,
        )
        next_velocity = velocity + frame.timestep * torch.einsum("ij,nj->ni", inverse, total)
        next_pose = torch.cat((pose[:, :3] + frame.timestep * next_velocity[:, :3],
                               _advance_quaternion(pose[:, 3:], next_velocity[:, 3:], frame.timestep)), -1)
        return next_pose, next_velocity, points, world_force

    def object_evaluate(self, indices, parameters):
        local, normals = self.select(indices)
        forces = self.dense_forces(parameters)
        frame = self.cache.frames[0]
        pose = frame.nominal_qpos[self.qadr:].expand(len(parameters), -1)
        velocity = frame.nominal_qvel[self.vadr:].expand(len(parameters), -1)
        poses, velocities, consensus = [pose], [velocity], []
        for t in range(self.horizon):
            pose, velocity, points, world_force = self._object_step(t, pose, velocity, local[:, t], normals[:, t], forces[:, t])
            poses.append(pose)
            velocities.append(velocity)
            consensus.append(self._consensus(points, world_force))
        # Fill robot coordinates from the reference; only object coordinates
        # participate in the upper tracking cost.
        reference_q = torch.cat((frame.nominal_qpos[None], self.refs[0]))
        reference_v = torch.cat((frame.nominal_qvel[None], self.refs[1]))
        qpos = torch.cat((reference_q[None, :, :self.qadr].expand(len(parameters), -1, -1), torch.stack(poses, 1)), -1)
        qvel = torch.cat((reference_v[None, :, :self.vadr].expand(len(parameters), -1, -1), torch.stack(velocities, 1)), -1)
        cost, _ = get_trajectory_cost(self.config, qpos, qvel, self.refs, object_only=True)
        cost = cost + self.options.force_regularization * forces.square().sum(-1).mean((1, 2))
        if self.horizon > 1:
            cost = cost + self.options.force_smoothness * forces.diff(dim=1).square().sum(-1).mean((1, 2))
        return ObjectEvaluation(cost, torch.stack(consensus, 1))

    def robot_evaluate(self, actions, selection):
        # CEM targets enter only the ADMM objective in solver.scsp. Collision
        # discovery and physical reaction forces depend on the robot state and
        # action, never on which upper-layer projection point was selected.
        controls = self.task.controls(actions[None])
        prediction = self.real_rollout.rollout(controls)
        if not torch.is_grad_enabled():
            self.last_prediction = prediction
        cost, _ = get_trajectory_cost(self.config, prediction.qpos, prediction.qvel, self.refs)
        tracking_cost = cost
        cost = cost + self.options.action_regularization * (controls-self.task.nominal).square().mean((1, 2))
        feet_error = prediction.site_xpos[:, :, 2:] - self.reference_sites[None, :, 2:]
        cost = cost + self.options.foot_tracking_weight * _smooth_norm(feet_error).mean((1, 2))
        penetration = torch.where(prediction.active_mask, torch.relu(-prediction.contact_gap-0.01), 0.0)
        cost = cost + 75.0 * penetration.square().mean((1, 2))
        # An explicitly requested approach is a soft tracking objective, not a
        # contact equality. Idle hands have neither approach nor position ADMM.
        local, _ = self.select(selection.indices[None])
        pose = prediction.qpos[:, :-1, self.qadr:]
        target = pose[..., None, :3] + _rotate(pose[..., None, 3:], local)
        approaching = self.operation_mask & ~self.contact_position_mask
        error = torch.where(approaching[None, ..., None],
                            (prediction.site_xpos[:, :, :2] - target) / self.options.position_scale, 0.0)
        approach_cost = self.options.approach_weight * _smooth_norm(error).sum(-1).mean(-1)
        cost = cost + approach_cost
        consensus = self._consensus(prediction.site_xpos[:, :, :2], prediction.contact_force[:, :, :2])
        return RobotEvaluation(cost[0], consensus[0], {
            "qpos": prediction.qpos[0], "qvel": prediction.qvel[0],
            "contact_site_xpos": prediction.site_xpos[0],
            "contact_force": prediction.contact_force[0],
            "contact_gap": prediction.contact_gap[0],
            "contact_active": prediction.active_mask[0],
            "real_contact_count": prediction.actual_contact_count[0],
            "base_tracking_cost": tracking_cost[0], "approach_cost": approach_cost[0],
        })

    def problem(self, mean):
        initial_sites = torch.stack([g.site_xpos[:2] for g in self.cache._geometry[:-1]])
        initial = self._consensus(initial_sites, torch.zeros_like(initial_sites))
        force_mean = mean.new_zeros((self.knots, self.hands, 3))
        force_mean[..., 0] = min(1.0, self.options.max_contact_force * 0.1)
        limit = self.options.action_trust_radius / self.task.control_scale
        return SCSPProblem(
            mean=mean, candidate_counts=torch.full((self.hands,), self.options.neighbors+1, device=mean.device, dtype=torch.long),
            force_mean=force_mean.flatten(), initial_consensus=initial,
            object_evaluate=self.object_evaluate, robot_evaluate=self.robot_evaluate, force_project=self.force_project,
            lower=torch.full_like(mean, -limit), upper=torch.full_like(mean, limit),
            consensus_mask=self.consensus_mask.clone(), prepare_iteration=self.prepare_iteration,
        )
