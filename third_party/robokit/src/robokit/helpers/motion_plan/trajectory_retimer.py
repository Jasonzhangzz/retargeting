# pyright: reportArgumentType=false, reportIndexIssue=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportOperatorIssue=false
from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import warp as wp

from robokit.opt.var_values import Var
from robokit.robo.robot import Robot
from robokit.types import ArrayLike
from robokit.utils.warp_utils import warp_max, wp_device_type


# --- configuration ---------------------------------------------------------
@dataclass
class RetimeConfig:
    """How a completed trajectory is retimed under the velocity, acceleration and jerk limits."""

    speed_scale: float = 0.95
    """Fraction of the fastest speed the limits allow: 1 runs at the limits, 0.5 at half speed."""
    minimum_dt: float = 0.02
    """Shortest step (s) a retime returns."""
    max_dt: float = 0.0
    """>0: longest step (s) a retime returns; 0 caps it at the planning step."""
    virtual_hold_frames: int = 0
    """>0: still frames held at the start and goal, with 5-point derivatives, so paths ramp from and to rest."""


# --- kernels ---------------------------------------------------------------
@wp.func
def _compute_retimed_kinematics(
    q: wp.array3d(dtype=wp.float32),
    batch: int,
    velocity_limits: wp.array1d(dtype=wp.float32),
    num_frames: int,
    num_dofs: int,
    dt: float,
    speed_scale: float,
    minimum_dt: float,
    max_dt: float,
    accel_limit: float,
    jerk_limit: float,
    enabled: int,
    hold_frames: int,
    lane: int,  # the lanes of a CUDA warp share the (frame, dof) pairs; one lane on CPU
    lanes: int,
) -> wp.vec2:
    velocity_dt = float(0.0)
    acceleration = float(0.0)
    jerk = float(0.0)
    if hold_frames > 0:
        # every frame, reading held copies past both ends, with 5-point derivatives
        for pair in range(lane, num_frames * num_dofs, lanes):
            frame, dof = pair // num_dofs, pair % num_dofs
            qm2 = q[batch, wp.max(frame - 2, 0), dof]
            qm1 = q[batch, wp.max(frame - 1, 0), dof]
            qp1 = q[batch, wp.min(frame + 1, num_frames - 1), dof]
            qp2 = q[batch, wp.min(frame + 2, num_frames - 1), dof]
            speed = wp.abs(qm2 - 8.0 * qm1 + 8.0 * qp1 - qp2) / 12.0
            velocity_dt = wp.max(velocity_dt, speed / wp.max(velocity_limits[dof], 1.0e-8))
            accel = wp.abs(-qm2 + 16.0 * qm1 - 30.0 * q[batch, frame, dof] + 16.0 * qp1 - qp2) / 12.0
            acceleration = wp.max(acceleration, accel)
            jerk = wp.max(jerk, wp.abs(-0.5 * qm2 + qm1 - qp1 + 0.5 * qp2))
    else:
        for pair in range(lane, (num_frames - 1) * num_dofs, lanes):
            frame, dof = pair // num_dofs, pair % num_dofs
            diff = wp.abs(q[batch, frame + 1, dof] - q[batch, frame, dof])
            velocity_dt = wp.max(velocity_dt, diff / wp.max(velocity_limits[dof], 1.0e-8))
            if frame + 2 < num_frames:
                d2 = (q[batch, frame + 2, dof] - q[batch, frame + 1, dof]) - (
                    q[batch, frame + 1, dof] - q[batch, frame, dof]
                )
                acceleration = wp.max(acceleration, wp.abs(d2))
            if frame + 3 < num_frames:
                d0 = q[batch, frame + 1, dof] - q[batch, frame, dof]
                d1 = q[batch, frame + 2, dof] - q[batch, frame + 1, dof]
                d3 = q[batch, frame + 3, dof] - q[batch, frame + 2, dof]
                jerk = wp.max(jerk, wp.abs(d3 - 2.0 * d1 + d0))
    velocity_dt = warp_max(velocity_dt)
    acceleration = warp_max(acceleration)
    jerk = warp_max(jerk)
    final_dt = dt
    if enabled != 0:
        acceleration_dt = wp.sqrt(acceleration / (accel_limit + 1.0e-8))
        jerk_dt = wp.pow(jerk / (jerk_limit + 1.0e-8), 1.0 / 3.0)
        ceiling = dt
        if max_dt > 0.0:
            ceiling = max_dt
        # 1.0001: stay strictly under the limits despite float rounding
        final_dt = wp.clamp(
            wp.max(wp.max(velocity_dt, acceleration_dt), jerk_dt) / speed_scale * 1.0001, minimum_dt, ceiling
        )
    return wp.vec2(final_dt, jerk / (final_dt * final_dt * final_dt + 1.0e-30))


@wp.kernel
def _retime_kernel(
    q: wp.array3d(dtype=wp.float32),
    velocity_limits: wp.array1d(dtype=wp.float32),
    num_frames: int,
    num_dofs: int,
    dt: float,
    speed_scale: float,
    minimum_dt: float,
    max_dt: float,
    accel_limit: float,
    jerk_limit: float,
    enabled: int,
    hold_frames: int,
    lanes: int,  # threads per trajectory: a CUDA warp's 32, or 1 on CPU
    step: wp.array1d(dtype=wp.float32),
):
    tid = wp.tid()
    batch, lane = tid // lanes, tid % lanes
    kinematics = _compute_retimed_kinematics(
        q,
        batch,
        velocity_limits,
        num_frames,
        num_dofs,
        dt,
        speed_scale,
        minimum_dt,
        max_dt,
        accel_limit,
        jerk_limit,
        enabled,
        hold_frames,
        lane,
        lanes,
    )
    if lane == 0:
        step[batch] = kinematics[0]


@wp.kernel
def _fastest_feasible_step_kernel(
    q: wp.array3d(dtype=wp.float32),
    scores: wp.array1d(dtype=wp.float32),
    velocity_limits: wp.array1d(dtype=wp.float32),
    dt: float,
    speed_scale: float,
    minimum_dt: float,
    max_dt: float,
    accel_limit: float,
    jerk_limit: float,
    hold_frames: int,
    num_candidates: int,
    lanes: int,  # threads per query: a CUDA warp's 32, or 1 on CPU
    out: wp.array1d(dtype=wp.float32),
):
    """Each query's smallest retimed step among its feasible candidates (among all when none is)."""
    tid = wp.tid()
    batch_idx, lane = tid // lanes, tid % lanes
    best = float(1.0e9)
    best_any = float(1.0e9)
    for candidate in range(num_candidates):
        row = batch_idx * num_candidates + candidate
        timing = _compute_retimed_kinematics(
            q,
            row,
            velocity_limits,
            q.shape[1],
            q.shape[2],
            dt,
            speed_scale,
            minimum_dt,
            max_dt,
            accel_limit,
            jerk_limit,
            1,
            hold_frames,
            lane,
            lanes,
        )
        best_any = wp.min(best_any, timing[0])
        if scores[row] < 1.0e6:
            best = wp.min(best, timing[0])
    if lane == 0:
        out[batch_idx] = wp.where(best < 1.0e9, best, best_any)


@wp.kernel
def _scale_step_kernel(
    step: wp.array1d(dtype=wp.float32),
    scale: float,
    minimum_dt: float,
    dt: wp.array1d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    dt[batch_idx] = wp.max(step[batch_idx] * scale, minimum_dt)


@wp.kernel
def _interpolate_trajectory_kernel(
    q: wp.array3d(dtype=wp.float32),
    step: wp.array1d(dtype=wp.float32),
    interpolation_dt: float,  # >0: sample the plan at this step; 0: out.shape[1] - 1 even samples
    anchors: wp.array1d(dtype=wp.int32),  # frames every plan passes as samples: 0, its waypoints, its last frame
    durations: wp.array1d(dtype=wp.float32),  # per segment: >0 its fixed seconds, else its frames times `step`
    out: wp.array3d(dtype=wp.float32),
):
    """Linearly interpolate the plan, segment by segment between anchors.

    Sampling steps:
    1. Spread ceil(segment time / interpolation_dt) samples evenly across each segment, its time its fixed duration or its frames * step.
    2. Repeat the final frame.
    """
    batch, sample, dof = wp.tid()  # pyright: ignore[reportAssignmentType]
    intervals = q.shape[1] - 1
    n = out.shape[1] - 1
    left = int(0)
    right = intervals
    start = int(0)
    count = n
    if interpolation_dt > 0.0:
        n = int(0)
        for k in range(anchors.shape[0] - 1):
            span = float(anchors[k + 1] - anchors[k]) * (step[batch] / interpolation_dt)
            if durations[k] > 0.0:
                span = durations[k] / interpolation_dt
            steps = wp.min(int(wp.ceil(span)), out.shape[1] - 1 - n)
            if sample >= n:
                left = anchors[k]
                right = anchors[k + 1]
                start = n
                count = steps
            n += steps
    if sample >= n:
        out[batch, sample, dof] = q[batch, intervals, dof]
        return
    x = float(left) + float(sample - start) / (float(count) / float(right - left))
    h = wp.max(int(wp.ceil(x)), 1)
    w = x - float(h - 1)
    out[batch, sample, dof] = w * q[batch, h, dof] + (1.0 - w) * q[batch, h - 1, dof]


# --- timing variable -------------------------------------------------------
class TimingVar(Var):
    """Retimer-owned timestep with no optimizer tangent; views, gathers, and integrates share its array."""

    def __init__(self, dt: wp.array):
        self.dt = dt

    @classmethod
    def full(cls, batch_size: int, dt: float, device: wp_device_type) -> "TimingVar":
        return cls(wp.full(batch_size, dt, dtype=wp.float32, device=device))

    def fill(self, dt: float):
        """Every trajectory at step `dt`."""
        self.dt.fill_(dt)

    def scale(self, step: wp.array, scale: float, minimum_dt: float):
        """Every trajectory at `scale` times its `step`, no shorter than `minimum_dt`."""
        wp.launch(
            _scale_step_kernel,
            dim=self.batch_size,
            inputs=[step, scale, minimum_dt],
            outputs=[self.dt],
            device=self.device,
        )

    @property
    def tangent_dim(self) -> int:
        return 0

    @property
    def batch_size(self) -> int:
        return int(self.dt.shape[0])

    @property
    def device(self) -> wp_device_type:
        return self.dt.device

    def clone(self) -> "TimingVar":
        return TimingVar(wp.clone(self.dt))

    def view(self, batch_size: int, offset: int = 0) -> "TimingVar":
        return self

    def integrate(
        self,
        velocity: ArrayLike,
        out: Optional["TimingVar"] = None,
        tangent_mask: Optional[wp.array] = None,
        weight_decay: float = 0.0,
        out_velocity: Optional[ArrayLike] = None,
    ) -> "TimingVar":
        return self if out is None else out

    def gather(self, indices: ArrayLike, out: Optional["TimingVar"] = None) -> "TimingVar":
        return self if out is None else out

    def accept(self, accept_mask: ArrayLike, proposed: "TimingVar") -> "TimingVar":
        return self


# --- retimer ---------------------------------------------------------------
class TrajectoryRetimer:
    """Retime trajectories: the shortest uniform time step their joint derivative limits allow."""

    def __init__(
        self,
        robot: Robot,
        num_frames: int,
        dt: float,
        acceleration_limit: float,
        jerk_limit: float,
        config: Optional[RetimeConfig],
        device: str,
    ):
        self.num_frames = num_frames
        self.dt = dt
        self.config = config
        self.limits = (acceleration_limit, jerk_limit)
        self.device = wp.get_device(device)
        self._velocity_limits = wp.from_numpy(
            robot.spec.actuated_joint_velocity_limits.astype(np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        self._num_dofs = robot.num_actuated_joints

    def retime(self, q: wp.array, timing: TimingVar):
        """Set `timing` to each trajectory's retimed step; without a config, the planning step.

        Args:
            q: Batched trajectories with shape `(batch, frames, dofs)`.
            timing: Their time steps, one per trajectory.
        """
        config = self.config
        wp.launch(
            _retime_kernel,
            dim=q.shape[0] * (32 if self.device.is_cuda else 1),
            inputs=[
                q,
                self._velocity_limits,
                self.num_frames,
                self._num_dofs,
                self.dt,
                config.speed_scale if config is not None else 1.0,
                config.minimum_dt if config is not None else self.dt,
                config.max_dt if config is not None else 0.0,
                *self.limits,
                int(config is not None),
                config.virtual_hold_frames if config is not None else 0,
                32 if self.device.is_cuda else 1,
            ],
            outputs=[timing.dt],
            block_dim=32 if self.device.is_cuda else 256,
            device=self.device,
        )

    def fastest_feasible(self, q: wp.array, scores: wp.array, num_candidates: int, out: wp.array):
        """Set `out` to each query's shortest retimed step among its feasible candidates (among all when none is).

        Args:
            q: Candidate trajectories with shape `(batch * num_candidates, frames, dofs)`.
            scores: Candidate scores; below 1e6 is feasible.
            num_candidates: Candidates per query.
            out: One step per query.
        """
        config = cast(RetimeConfig, self.config)
        wp.launch(
            _fastest_feasible_step_kernel,
            dim=out.shape[0] * (32 if self.device.is_cuda else 1),
            inputs=[
                q,
                scores,
                self._velocity_limits,
                self.dt,
                config.speed_scale,
                config.minimum_dt,
                config.max_dt,
                *self.limits,
                config.virtual_hold_frames,
                num_candidates,
                32 if self.device.is_cuda else 1,
            ],
            outputs=[out],
            block_dim=32 if self.device.is_cuda else 256,
            device=self.device,
        )


__all__ = ["TimingVar", "TrajectoryRetimer", "RetimeConfig"]
