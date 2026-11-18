# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false
"""MPPI trajectory optimization.

MPPI: https://arxiv.org/abs/1509.01149.
"""

import dataclasses
import weakref
from dataclasses import field
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import warp as wp

from robokit.helpers.motion_plan.gradient_trajectory_optimizer import (
    MotionPlanContext,
    _active_dof_mask_kernel,
    _build_terms,
    _dof_links,
)
from robokit.helpers.motion_plan.motion_plan_score_task import MotionPlanScoreTask
from robokit.lie.se3 import se3_identity
from robokit.lie.se3_kernels import se3_exp_map_func, se3_inverse_func, se3_log_map_func, se3_multiply_func
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig, Stage
from robokit.opt.sampler import GaussianSampler, GaussianSamplerConfig
from robokit.opt.var_values import Var, VarKey, VarValues, VectorVar
from robokit.robo.robot import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.composite_task import CompositeTask
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.sparse.trajectory_limit_task import TrajectoryLimitTask
from robokit.terms.sparse.trajectory_position_task import TrajectoryPositionTask
from robokit.terms.sparse.trajectory_self_collision_task import TrajectorySelfCollisionTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.task import ResidualTask
from robokit.terms.trajectory_task import TrajectoryTask
from robokit.utils import warp_utils
from robokit.utils.warp_utils import resolve_device, wp_device_type, wp_vec6, wp_vec7


if TYPE_CHECKING:
    from robokit.geom import WarpScene
    from robokit.helpers.motion_plan.motion_planner import OnlineState


# --- kernels ---------------------------------------------------------------
@wp.kernel
def _set_link_position_targets_kernel(
    T_world_target: wp.array2d(dtype=wp_vec7),  # (candidate rows, target links): the first link's position
    out: wp.array3d(dtype=wp.float32),
):
    draw, frame = wp.tid()  # type: ignore
    target = T_world_target[draw // (out.shape[0] // T_world_target.shape[0]), 0]  # the draws share their row's
    out[draw, frame, 0] = target[0]
    out[draw, frame, 1] = target[1]
    out[draw, frame, 2] = target[2]


@wp.kernel
def _settle_std_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    target: wp.array2d(dtype=wp_vec7),  # (queries, target links): the first link settles
    ee_link_index: int,
    settle_radius: float,
    scale_in: wp.array1d(dtype=wp.float32),
    use_fk: int,
    min_std: float,
    init_std: float,
    std_scale: wp.array1d(dtype=wp.float32),
    std_reset: wp.array2d(dtype=wp.float32),
):
    """Reopen sampler width using `scale_in`, or end-effector distance to `target` with `use_fk`.

    Distance cases:
    - Outside `settle_radius`: use `init_std`.
    - Inside `settle_radius`: use `min_std`.
    """
    batch_idx, dim_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    scale = scale_in[batch_idx]
    if use_fk != 0:
        ee = T_world_link[batch_idx, ee_link_index]
        goal = target[batch_idx, 0]
        distance = wp.sqrt(
            (ee[0] - goal[0]) * (ee[0] - goal[0])
            + (ee[1] - goal[1]) * (ee[1] - goal[1])
            + (ee[2] - goal[2]) * (ee[2] - goal[2])
        )
        scale = wp.min(1.0, distance / settle_radius) if settle_radius > 0.0 else 1.0
    std_reset[batch_idx, dim_idx] = (min_std + (init_std - min_std) * scale) * std_scale[dim_idx % std_scale.shape[0]]


@wp.kernel
def _start_kernel(
    q: wp.array3d(dtype=wp.float32),
    current_q: wp.array2d(dtype=wp.float32),
    action_prev: wp.array2d(dtype=wp.float32),
    warm: int,
    dt: float,
    num_frames: int,
    num_dofs: int,
    action: wp.array2d(dtype=wp.float32),
    q0: wp.array2d(dtype=wp.float32),
    prev_q: wp.array2d(dtype=wp.float32),
    v0: wp.array2d(dtype=wp.float32),
):
    """Rollout start state, and the mean action shifted one frame (warm) or zeroed (cold)."""
    batch_idx, dof_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    frame_idx = dof_idx / num_dofs
    joint_idx = dof_idx % num_dofs
    if frame_idx == 0:  # start state: a warm start takes its velocity from the executed step
        if warm != 0:
            current = current_q[batch_idx, joint_idx]
            v0[batch_idx, joint_idx] = (current - prev_q[batch_idx, joint_idx]) / dt
        else:
            current = q[batch_idx, 0, joint_idx]
            v0[batch_idx, joint_idx] = 0.0
        q0[batch_idx, joint_idx] = current
        prev_q[batch_idx, joint_idx] = current
    if warm == 0:
        action[batch_idx, dof_idx] = 0.0
    elif frame_idx < num_frames - 1:
        action[batch_idx, dof_idx] = action_prev[batch_idx, (frame_idx + 1) * num_dofs + joint_idx]
    else:
        action[batch_idx, dof_idx] = action_prev[batch_idx, (num_frames - 1) * num_dofs + joint_idx]


@wp.kernel
def _start_base_kernel(
    current_T: wp.array1d(dtype=wp_vec7),
    warm: int,
    dt: float,
    T0: wp.array1d(dtype=wp_vec7),
    prev_T: wp.array1d(dtype=wp_vec7),
    twist0: wp.array1d(dtype=wp_vec6),
):
    """Floating-base rollout start pose, and the twist a warm start reads back from the executed step."""
    batch_idx = wp.tid()
    current = current_T[batch_idx]
    if warm != 0:
        delta = se3_multiply_func(se3_inverse_func(prev_T[batch_idx]), current)
        twist0[batch_idx] = se3_log_map_func(delta, wp.float32(1e-6)) / dt
    else:
        twist0[batch_idx] = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    T0[batch_idx] = current
    prev_T[batch_idx] = current


@wp.kernel
def _compute_base_rollout_kernel(
    T0: wp.array1d(dtype=wp_vec7),
    twist0: wp.array1d(dtype=wp_vec6),
    dt_schedule: wp.array1d(dtype=wp.float32),
    velocity_limit: float,
    num_frames: int,
    T_world_base: wp.array2d(dtype=wp_vec7),
):
    """Carry the measured base twist along the horizon, as `T <- T * exp(v dt)`; the base itself is never sampled."""
    batch_idx = wp.tid()
    instance_idx = batch_idx * T0.shape[0] / T_world_base.shape[0]  # as the joint rollout's start row
    current_T_world_base = T0[instance_idx]
    twist = twist0[instance_idx]
    for axis in range(6):
        twist[axis] = wp.clamp(twist[axis], -velocity_limit, velocity_limit)
    T_world_base[batch_idx, 0] = current_T_world_base
    for frame_idx in range(1, num_frames):
        current_T_world_base = se3_multiply_func(
            current_T_world_base, se3_exp_map_func(twist * dt_schedule[frame_idx], wp.float32(1e-6))
        )
        T_world_base[batch_idx, frame_idx] = current_T_world_base


@wp.kernel
def _set_shifted_pinned_trajectory_kernel(
    src: wp.array3d(dtype=wp.float32),
    current: wp.array2d(dtype=wp.float32),
    num_frames: int,
    dst: wp.array3d(dtype=wp.float32),
):
    batch_idx, frame_idx, joint_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    if frame_idx == 0:
        dst[batch_idx, 0, joint_idx] = current[batch_idx, joint_idx]
    elif frame_idx < num_frames - 1:
        dst[batch_idx, frame_idx, joint_idx] = src[batch_idx, frame_idx + 1, joint_idx]
    else:
        dst[batch_idx, frame_idx, joint_idx] = src[batch_idx, num_frames - 1, joint_idx]


@wp.kernel
def _compute_acceleration_rollout_kernel(
    q0: wp.array2d(dtype=wp.float32),
    v0: wp.array2d(dtype=wp.float32),
    action: wp.array2d(dtype=wp.float32),
    dt_schedule: wp.array1d(dtype=wp.float32),
    accel_limit: float,
    velocity_limit: wp.array1d(dtype=wp.float32),
    q_lower: wp.array1d(dtype=wp.float32),
    q_upper: wp.array1d(dtype=wp.float32),
    num_frames: int,
    num_dofs: int,
    q: wp.array3d(dtype=wp.float32),
):
    batch_idx, joint_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    # the start row: draws share their center's, and a selected winner takes its query's first candidate's
    instance_idx = batch_idx * q0.shape[0] / (q.shape[0])
    position = q0[instance_idx, joint_idx]
    velocity = v0[instance_idx, joint_idx]
    q[batch_idx, 0, joint_idx] = position
    max_velocity = velocity_limit[joint_idx]
    for frame_idx in range(1, num_frames):
        dt = dt_schedule[frame_idx]
        accel = wp.clamp(action[batch_idx, frame_idx * num_dofs + joint_idx], -accel_limit, accel_limit)
        velocity = wp.clamp(velocity + accel * dt, -max_velocity, max_velocity)
        position = wp.clamp(position + velocity * dt, q_lower[joint_idx], q_upper[joint_idx])
        q[batch_idx, frame_idx, joint_idx] = position


@wp.kernel
def _color_noise_kernel(
    noise: wp.array2d(dtype=wp.float32),
    num_draws: int,
    knot_count: int,
    knot_frames: wp.array1d(dtype=wp.int32),
    active_mask: wp.array1d(dtype=wp.float32),
    b0: float,
    b1: float,
    b2: float,
    num_frames: int,
    num_dofs: int,
):
    """Low-pass horizon noise after piecewise-linear knot interpolation for each center's first `knot_count` draws."""
    batch_idx, joint_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    num_knots = knot_frames.shape[0]
    use_knots = batch_idx % num_draws < knot_count and num_knots >= 2
    raw_first = wp.float32(0.0)  # the unfiltered draw at the knot before this frame
    for frame_idx in range(num_frames):
        col = frame_idx * num_dofs + joint_idx
        value = noise[batch_idx, col]
        if use_knots:
            interval = int(0)
            for knot_idx in range(num_knots - 1):
                if frame_idx > knot_frames[knot_idx]:
                    interval = knot_idx
            first = knot_frames[interval]
            second = knot_frames[interval + 1]
            if frame_idx == first or frame_idx == second:  # noqa: PLR1714
                raw_first = value
            else:
                alpha = wp.float32(frame_idx - first) / wp.float32(second - first)
                value = (1.0 - alpha) * raw_first + alpha * noise[batch_idx, second * num_dofs + joint_idx]
        if frame_idx >= 2:
            value = b0 * value + b1 * noise[batch_idx, col - num_dofs] + b2 * noise[batch_idx, col - 2 * num_dofs]
        noise[batch_idx, col] = value * active_mask[col]  # the filter must not leak noise into masked entries


# --- acceleration control space --------------------------------------------
class AccelerationTrajectory(VarValues):
    """Acceleration actions with lazy joint-trajectory rollout; the tangent contains actions only.

    Operations:
    - `integrate`: add to actions and invalidate the trajectory.
    - `robot`: roll out from the start state when terms next read it.
    - Copy into draws: copy actions without rolling out the center.
    """

    def __init__(self, action: VectorVar, robot: RobotState, rollout: "weakref.WeakMethod"):
        super().__init__({VarKey.ACTION: action, VarKey.ROBOT: robot})
        self._rollout = rollout  # weak: the optimizer owns this variable, so a strong method would form a cycle
        self._stale = False  # `robot` lags `action` until it is next read

    @property
    def tangent_dim(self) -> int:
        return self.get(VarKey.ACTION).tangent_dim

    def get(self, key: Enum) -> Var:
        if key is VarKey.ROBOT and self._stale:
            self._stale = False
            rollout = cast(Callable[[wp.array, RobotState], None], self._rollout())
            rollout(cast(VectorVar, super().get(VarKey.ACTION)).x, cast(RobotState, super().get(VarKey.ROBOT)))
        return super().get(key)

    def integrate(
        self,
        velocity: wp.array,
        out: Optional[VarValues] = None,
        tangent_mask: Optional[wp.array] = None,
        weight_decay: float = 0.0,
        out_velocity: Optional[wp.array] = None,
    ) -> "AccelerationTrajectory":
        out = self.clone() if out is None else cast(AccelerationTrajectory, out)
        self.get(VarKey.ACTION).integrate(
            velocity,
            out=out.get(VarKey.ACTION),
            tangent_mask=tangent_mask,
            weight_decay=weight_decay,
            out_velocity=out_velocity,
        )
        out._stale = True
        return out

    def gather(self, indices: wp.array, out: Optional[VarValues] = None) -> "AccelerationTrajectory":
        if out is None:
            action = cast(VectorVar, self.get(VarKey.ACTION)).gather(indices)
            out = AccelerationTrajectory(
                action, cast(RobotState, super().get(VarKey.ROBOT)).gather(indices), self._rollout
            )
            out._stale = self._stale
            return out
        # Only the actions move; `out` rolls its own trajectory out when read.
        out = cast(AccelerationTrajectory, out)
        cast(VectorVar, self.get(VarKey.ACTION)).gather(indices, out=cast(VectorVar, out.get(VarKey.ACTION)))
        out._stale = True
        return out

    def clone(self) -> "AccelerationTrajectory":
        action, robot = (
            cast(VectorVar, self.get(VarKey.ACTION)).clone(),
            cast(RobotState, super().get(VarKey.ROBOT)).clone(),
        )
        out = AccelerationTrajectory(action, robot, self._rollout)
        out._stale = self._stale
        return out

    def view(self, batch_size: int, offset: int = 0) -> "AccelerationTrajectory":
        action = cast(VectorVar, self.get(VarKey.ACTION)).view(batch_size, offset)
        out = AccelerationTrajectory(
            action, cast(RobotState, super().get(VarKey.ROBOT)).view(batch_size, offset), self._rollout
        )
        out._stale = self._stale
        return out

    def accept(self, accept_mask: wp.array, proposed: VarValues) -> "AccelerationTrajectory":
        super().accept(accept_mask, proposed)
        self._stale = True
        return self


def _default_mppitrajectoryoptimizer_terms() -> List[ResidualTask]:
    ctx = MotionPlanContext
    return [
        # first, so its term streams fork first: the swept cost runs longest and otherwise waits for SMs
        TrajectoryCollisionTask(
            scene=ctx.SCENE,
            scene_indices=ctx.SCENE_INDICES,
            weight=0.0,
            margin=0.01,
            name="collision",
        ),
        TrajectoryTask(
            FrameTask(
                frame_index=ctx.TARGET_LINKS, T_world_target=ctx.TARGET, position_weight=0.0, orientation_weight=0.0
            ),
            frame_indices=[-1],
            name="target",
        ),
        TrajectoryTask(ConfigurationTask(rest_q=ctx.START_Q, weight=100.0), frame_indices=[0], name="start"),
        TrajectorySmoothnessTask(weight=0.0, smooth_boundary=False, name="smoothness"),
        TrajectorySmoothnessTask(weight=0.0, order=3, dt=ctx.STEP, name="jerk_smoothness"),
        TrajectorySmoothnessTask(weight=0.0, order=2, dt=ctx.STEP, name="acceleration_smoothness"),
        TrajectoryLimitTask(
            order=1, upper=ctx.VELOCITY_LIMITS, weight=1.0, dt=ctx.STEP, stencil="forward", name="velocity_limit"
        ),
        TrajectoryTask(PositionLimit(weight=100.0), name="position_limit"),
        TrajectoryTask(ConfigurationTask(rest_q="midrange", weight=0.01), name="rest"),
        TrajectorySelfCollisionTask(weight=10.0, name="self_collision"),
        TrajectoryTask(ConfigurationTask(rest_q=ctx.GOAL_Q, weight=0.0), frame_indices=[-1], name="goal"),
        TrajectoryTask(ConfigurationTask(rest_q=ctx.GOAL_Q, weight=0.0), name="goal_rest"),
        TrajectoryPositionTask(
            frame_index=ctx.EE_LINK, target_positions=ctx.LINK_TARGETS, weight=0.0, name="link_position"
        ),
    ]


@dataclasses.dataclass(frozen=True)
class ColoredGaussianSamplerConfig(GaussianSamplerConfig):
    """Horizon-filtered noise on the acceleration actions, with an optional share of knot draws."""

    var_key: Enum = VarKey.ACTION
    """The sampled variable; MPPI sets it from its control space."""
    init_std: float = 0.3
    """Noise standard deviation at the start of a solve."""
    knot_ratio: float = 0.0
    """Share of each center's draws that interpolate between knots instead of filtering white noise."""
    num_knots: int = 5
    """Frames a knot draw interpolates between."""
    noise_filter: Tuple[float, float, float] = (0.3, 0.3, 0.4)
    """Second-order low-pass along the horizon."""

    def build(
        self, var: VarValues, num_candidates: int, active_dof_mask: Optional[wp.array] = None
    ) -> "ColoredGaussianSampler":
        frames = cast(RobotState, VarValues.get(var, VarKey.ROBOT)).q.shape[1]
        knot_count = round(self.knot_ratio * self.num_draws)
        if knot_count > 0 and self.num_knots < 2:
            raise ValueError("knot sampling needs at least two knots")
        options = dataclasses.asdict(self)
        for key in ("knot_ratio", "num_knots", "noise_filter"):
            options.pop(key)
        width, offset = var.get(self.var_key).tangent_dim, var.tangent_offset(self.var_key)
        return ColoredGaussianSampler(
            frames,
            width // frames,
            self.noise_filter,
            np.round(np.linspace(0, frames - 1, self.num_knots)).astype(np.int32),
            knot_count,
            **options,
            tangent_dim=width,
            instances=var.batch_size,
            active_dof_mask=None if active_dof_mask is None else active_dof_mask[offset : offset + width],
            device=var.device,
        )


class ColoredGaussianSampler(GaussianSampler):
    """Draw low-pass Gaussian acceleration noise; first `knot_count` draws interpolate `knot_frames`."""

    def __init__(
        self,
        num_frames: int,
        num_dofs: int,
        noise_filter: Tuple[float, float, float],
        knot_frames: np.ndarray,
        knot_count: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_frames, self.num_dofs, self.noise_filter = num_frames, num_dofs, noise_filter
        self.knot_frames = wp.array(knot_frames, dtype=wp.int32, device=self._device)
        self.knot_count = knot_count

    def fill(self, out_tangent: wp.array, iteration: int = 0) -> None:
        super().fill(out_tangent, iteration)
        b0, b1, b2 = self.noise_filter
        wp.launch(
            _color_noise_kernel,
            dim=(out_tangent.shape[0], self.num_dofs),
            inputs=[
                out_tangent,
                self.num_draws,
                self.knot_count,
                self.knot_frames,
                self.active_dof_mask,
                b0,
                b1,
                b2,
                self.num_frames,
                self.num_dofs,
            ],
            device=self._device,
        )


# --- configuration ---------------------------------------------------------
@dataclasses.dataclass
class MppiTrajectoryOptimizerConfig:
    """Configuration for position- or acceleration-controlled trajectory MPPI."""

    terms: List[ResidualTask] = field(default_factory=_default_mppitrajectoryoptimizer_terms)
    """Task templates the stage builds from its robot, inputs, horizon and timing."""
    score_terms: List[ResidualTask] = field(default_factory=lambda: [MotionPlanScoreTask(keep_candidate_zero=False)])
    """Rank the candidates when there are several; several tasks combine."""
    solver: PopulationSolverConfig = field(
        default_factory=lambda: PopulationSolverConfig(stages=[Stage(50)], samplers=[ColoredGaussianSamplerConfig()])
    )
    """One stage of sampling updates (`iters`) and trajectories per query (`num_candidates`), and one `ColoredGaussianSamplerConfig`: draws, spread and update rule. Position control draws white noise, as knots and the filter shape actions; the planner sets graph capture."""
    method: Literal["mppi"] = "mppi"
    """The stage's method, which the planner dispatches on."""
    control_space: Literal["position", "acceleration"] = "position"
    """Sample joint positions directly, or accelerations that roll out from the start state."""
    lock_frames: Tuple[int, ...] = ()
    """Frames held where they start, e.g. `(0,)` the start or `(0, -1)` both ends."""
    std_scale: Optional[Sequence[float]] = None
    """Per-joint noise scale, where 0 never samples a joint; None samples every joint at full scale."""
    q_limit_margin: float = 0.0
    """Margin (rad) an acceleration rollout may pass the joint limits by before it clamps."""
    dt_max: float = 0.0
    """Above the planning step: an acceleration rollout's step grows linearly to this over the horizon's second half."""
    settle_radius: float = 0.0
    """>0: an online update narrows the spread toward the sampler's `min_std` as the end effector comes within this distance (m) of its goal."""
    fresh_noise: bool = False
    """Each online update draws new noise instead of replaying one sequence."""

    def add(self, term: ResidualTask, *, name: Optional[str] = None) -> "MppiTrajectoryOptimizerConfig":
        """Append a task, optionally assigning its input role.

        >>> from robokit.terms.dense.configuration_task import ConfigurationTask
        >>> config = MppiTrajectoryOptimizerConfig(terms=[]).add(ConfigurationTask(weight=0.1), name="rest")
        >>> config.task("rest").weight
        0.1
        """
        if name is not None:
            term.name = name
        if term.name and any(t.name == term.name for t in self.terms):
            raise ValueError(f"Duplicate task name {term.name!r}.")
        self.terms.append(term)
        return self

    def task(self, name: str) -> ResidualTask:
        """Return a named task; raise KeyError when absent.

        >>> from robokit.terms.dense.configuration_task import ConfigurationTask
        >>> config = MppiTrajectoryOptimizerConfig(terms=[]).add(ConfigurationTask(weight=0.2), name="rest")
        >>> config.task("rest").weight
        0.2
        """
        for term in self.terms:
            if term.name == name:
                return term
        raise KeyError(f"No task named {name!r}.")


# --- optimizer -------------------------------------------------------------
class MppiTrajectoryOptimizer:
    """Trajectory MPPI with optional caller-owned state for receding-horizon updates.

    Lifecycle:
        1. `__init__` stores configuration and creates the active-DOF mask.
        2. `warmup(batch_size)` builds batch-shaped tasks, buffers, execution graphs, and the variable:
           a. position control samples the trajectory `RobotState` directly;
           b. acceleration control samples an `AccelerationTrajectory`, whose actions roll out to `robot` with `ColoredGaussianSampler` noise.
        3. `solve(...)` uses the same path for offline, cold-online, and warm-online optimization:
           a. `_set_inputs(...)` updates tasks and restores or initializes the optimization state.
           b. `_solver.solve(...)` runs the `PopulationSolver` schedule.
           c. `wp.copy(...)` saves the trajectory, previous configuration, and mean action to `online_state` when provided.
    """

    def __init__(
        self,
        config: MppiTrajectoryOptimizerConfig,
        robot: Robot,
        target_links: Union[str, int, Sequence[Union[str, int]]],
        num_frames: int,
        dt: float,
        *,
        enable_T_world_base: bool = False,
        scene: Optional["WarpScene"] = None,
        use_cuda_graph: bool = True,
        acceleration_limit: float = 15.0,
        sphere_mask: Optional[wp.array] = None,
        device: Optional[wp_device_type] = None,
    ):
        device = str(resolve_device(device))
        self.robot = robot
        links = [target_links] if isinstance(target_links, (str, int)) else list(target_links)
        self.target_link_indices = [robot.link_names.index(x) if isinstance(x, str) else int(x) for x in links]
        self.ee_link_index = self.target_link_indices[0]  # settling reads the first target link
        self._base_dofs = 6 if enable_T_world_base else 0
        if self._base_dofs and config.control_space != "acceleration":
            raise ValueError("A floating base needs acceleration control: it carries its measured twist.")
        self.num_frames = num_frames
        self._scene = scene
        self._sphere_mask = sphere_mask  # each frame's checked spheres, for the score
        self.device = device
        self.num_dofs = robot.num_actuated_joints
        if len(config.solver.samplers) != 1 or config.solver.optimizer is not None or len(config.solver.stages) != 1:
            raise ValueError("MPPI samples with one sampler, no optimizer and one stage.")
        self.num_candidates = config.solver.num_candidates[0]
        self._use_cuda_graph = use_cuda_graph
        self._acceleration_limit = acceleration_limit  # the plan's, which acceleration rollouts clamp to
        self._config = config
        self._dt = dt
        self.batch_size = 0
        self._initialized = False
        std_scale = np.ones(self.num_dofs) if config.std_scale is None else np.asarray(config.std_scale)
        if std_scale.shape != (self.num_dofs,):
            raise ValueError("std_scale needs one value per actuated joint.")
        self._std_scale_wp = wp.from_numpy(std_scale.astype(np.float32), dtype=wp.float32, device=device)
        self._active_dof_mask_wp = wp.empty(num_frames * self.num_dofs, dtype=wp.float32, device=device)
        self._sampled = (std_scale != 0).astype(np.float32)  # unsampled joints stay put
        self._dof_mask_wp = wp.array(self._sampled, dtype=wp.float32, device=device)
        config_locked = np.zeros(num_frames, np.int32)
        config_locked[[f % num_frames for f in config.lock_frames]] = 1
        self._config_locked_wp = wp.array(config_locked, dtype=wp.int32, device=device)
        self._dof_links = _dof_links(robot, self.target_link_indices, 0, device)
        self.locked_frames = wp.zeros(
            num_frames, dtype=wp.int32, device=device
        )  # each frame's held target links as bits, -1 all; may change on device
        self.set_lock_frames(())

    def set_lock_frames(self, lock_frames: Sequence[int]):
        """Hold these frames, besides `config.lock_frames`, where they start; never while capturing."""
        locked = np.zeros(self.num_frames, np.int32)
        locked[[f % self.num_frames for f in lock_frames]] = -1
        self.locked_frames.assign(locked)
        self._apply_active_dof_mask()

    def set_active_joint_mask(self, joint_mask: Sequence[float]):
        """Lock or unlock actuated joints without rebuilding the optimizer."""
        self._dof_mask_wp.assign(np.asarray(joint_mask, dtype=np.float32) * self._sampled)
        self._apply_active_dof_mask()

    def _apply_active_dof_mask(self):
        wp.launch(
            _active_dof_mask_kernel,
            dim=(self.num_frames, self.num_dofs),
            inputs=[self._dof_mask_wp, self.locked_frames, self._config_locked_wp, self._dof_links, None, 0],
            outputs=[self._active_dof_mask_wp],
            device=self.device,
        )

    def _set_inputs(
        self,
        start_q: wp.array,
        T_world_target: Optional[wp.array],
        target_q: Optional[wp.array],
        traj_q_init: Optional[wp.array],
        target_q_per_candidate: wp.array,
        scene_indices: wp.array,
        online_state: Optional["OnlineState"],
        explore_scale: Optional[Union[float, wp.array]],
        start_T_world_base: Optional[wp.array] = None,
    ) -> bool:
        """Set task inputs and return whether this solve starts from `traj_q_init`."""
        # the tasks read these buffers, so refilling them updates every task
        target, goal_q = self._goal_vec7_buf, self._target_q_per_candidate_buf
        sq = warp_utils.repeat(start_q, self.num_candidates, out=self._sq_buf)
        if self._base_dofs:
            if start_T_world_base is None:
                raise ValueError("A floating base needs start_T_world_base.")
            warp_utils.repeat(start_T_world_base, self.num_candidates, out=self._start_state.T_world_base)
        if target_q is not None:
            warp_utils.repeat(target_q, self.num_candidates, out=goal_q)
        else:
            wp.copy(goal_q, target_q_per_candidate)
        if T_world_target is not None:
            warp_utils.repeat(
                T_world_target.reshape((start_q.shape[0], target.shape[1])), self.num_candidates, out=target
            )
        else:  # a joint goal: its pose is the goal configuration's, at the start base
            self._goal_state.invalidate()
            self.robot.forward_kinematics(self._goal_state)
            warp_utils.gather(
                self._goal_state.T_world_link.reshape((-1,)), self._goal_link_rows, out=target.reshape((-1,))
            )
        warp_utils.repeat(scene_indices, self.num_candidates, out=self._scene_indices_buf)  # the tasks' buffer
        wp.launch(
            _set_link_position_targets_kernel,
            dim=(self._link_targets.shape[0], self.num_frames),
            inputs=[target],
            outputs=[self._link_targets],
            device=self.device,
        )

        acceleration = self._config.control_space == "acceleration"
        cold_start = traj_q_init is not None or online_state is None or online_state.prev_q_traj is None
        use_fk = 0
        if cold_start:
            if traj_q_init is None:
                warp_utils.repeat(
                    start_q,
                    self.num_frames,
                    out=self._state.q.reshape((start_q.shape[0] * self.num_frames, self.num_dofs)),
                )
            else:
                wp.copy(self._state.q, traj_q_init)
            self._var.invalidate()
            self._settle_scale.fill_(1.0)
            if acceleration:
                self._start(sq, warm=False)
        else:
            assert online_state is not None
            if self.num_candidates != 1:
                raise ValueError("Online MPPI requires num_candidates=1.")
            if acceleration:
                if online_state.prev_start_q is None or online_state.mean_action is None:
                    raise ValueError("Acceleration MPPI state requires prev_start_q and mean_action.")
                wp.copy(self.prev_q, online_state.prev_start_q)
                wp.copy(self._action.x, online_state.mean_action)
                self._start(sq, warm=True)
            else:
                assert 0 in self._config.lock_frames, "position update must lock frame zero"
                wp.copy(self._shift_scratch, online_state.prev_q_traj)
                wp.launch(
                    _set_shifted_pinned_trajectory_kernel,
                    dim=self._state.q.shape,
                    inputs=[self._shift_scratch, sq, self.num_frames],
                    outputs=[self._state.q],
                    device=self.device,
                )
                self._var.invalidate()
            if isinstance(explore_scale, wp.array):
                wp.copy(self._settle_scale, explore_scale)
            elif explore_scale is not None:
                self._settle_scale.fill_(explore_scale)
            elif T_world_target is None:
                self._settle_scale.fill_(1.0)
            else:  # settle by the end effector's distance to the goal; `sq` is the start state's q
                self._start_state.invalidate()
                self.robot.forward_kinematics(self._start_state)
                use_fk = 1
        sampler = cast(GaussianSampler, self._solver.samplers[0])
        wp.launch(
            _settle_std_kernel,
            dim=sampler.std_reset.shape,
            inputs=[
                self._start_state.T_world_link,
                target,
                self.ee_link_index,
                self._config.settle_radius,
                self._settle_scale,
                use_fk,
                sampler.min_std,
                sampler.init_std,
                self._std_scale_wp,
                sampler.std_reset,
            ],
            device=self.device,
        )
        return cold_start

    def _start(self, current_q: wp.array, warm: bool):
        """Set the rollout start state and shift the mean action one frame (or zero it on a cold start)."""
        wp.copy(self._action_scratch, self._action.x)
        wp.launch(
            _start_kernel,
            dim=self._action.x.shape,
            inputs=[
                self._state.q,
                current_q,
                self._action_scratch,
                int(warm),
                self._dt,
                self.num_frames,
                self.num_dofs,
            ],
            outputs=[self._action.x, self.q0, self.prev_q, self.v0],
            device=self.device,
        )
        if self._base_dofs:
            wp.launch(
                _start_base_kernel,
                dim=self.base_T0.shape[0],
                inputs=[self._start_state.T_world_base, int(warm), self._dt],
                outputs=[self.base_T0, self.base_prev_T, self.base_twist0],
                device=self.device,
            )

    def _rollout(self, action: wp.array, out_state: RobotState) -> None:
        """Integrate acceleration actions from the start state into joint trajectories."""
        wp.launch(
            _compute_acceleration_rollout_kernel,
            dim=(action.shape[0], self.num_dofs),
            inputs=[
                self.q0,
                self.v0,
                action,
                self.dt_schedule,
                self._acceleration_limit,
                self.velocity_limit,
                self.q_lower,
                self.q_upper,
                self.num_frames,
                self.num_dofs,
            ],
            outputs=[out_state.q],
            device=self.device,
        )
        if self._base_dofs:
            wp.launch(
                _compute_base_rollout_kernel,
                dim=action.shape[0],
                inputs=[
                    self.base_T0,
                    self.base_twist0,
                    self.dt_schedule,
                    1.0,  # clamp on the measured base twist the horizon carries
                    self.num_frames,
                ],
                outputs=[out_state.T_world_base],
                device=self.device,
            )
        out_state.invalidate()

    def warmup(self, batch_size: int):
        """Build the batch-shaped tasks, buffers, and MPPI solver."""
        if self._initialized:
            return
        config = self._config
        actual_batch = batch_size * self.num_candidates
        self.batch_size = batch_size
        num_dofs, device = self.num_dofs, self.device
        q = wp.zeros((actual_batch, self.num_frames, num_dofs), dtype=wp.float32, device=device)
        base = se3_identity(shape=(actual_batch, self.num_frames), device=device) if self._base_dofs else None
        self._state = self.robot.state(q=q, T_world_base=base)
        self._goal_vec7_buf = wp.zeros((actual_batch, len(self.target_link_indices)), dtype=wp_vec7, device=device)
        self._sq_buf = wp.zeros((actual_batch, num_dofs), dtype=wp.float32, device=device)
        self._target_q_per_candidate_buf = wp.zeros((actual_batch, num_dofs), dtype=wp.float32, device=device)
        self._scene_indices_buf = wp.zeros(actual_batch, dtype=wp.int32, device=device)
        # the first target link's goal position at every frame of every draw
        self._link_targets = wp.zeros(
            (actual_batch * config.solver.num_draws, self.num_frames, 3), dtype=wp.float32, device=device
        )
        ctx = MotionPlanContext
        context = {
            ctx.TARGET_LINKS: self.target_link_indices,
            ctx.TARGET: self._goal_vec7_buf,
            ctx.START_Q: self._sq_buf,
            ctx.GOAL_Q: self._target_q_per_candidate_buf,
            ctx.STEP: self._dt,
            ctx.VELOCITY_LIMITS: self.robot.spec.actuated_joint_velocity_limits.astype(np.float32),
            ctx.SCENE: self._scene,
            ctx.SCENE_INDICES: self._scene_indices_buf,
            ctx.EE_LINK: self.ee_link_index,
            ctx.LINK_TARGETS: self._link_targets,
            ctx.CANDIDATES: self.num_candidates,
            # weak: the solver holds the score, so a strong `self` would keep the scene alive until a GC pass
            ctx.CANDIDATE_COSTS: lambda self_ref=weakref.ref(self): self_ref()._solver.candidate_scores,
            ctx.DRAWS: config.solver.num_draws,
            ctx.SPHERE_MASK: self._sphere_mask,
        }
        var = VarValues({VarKey.ROBOT: self._state})
        self._tasks = _build_terms(config.terms, var, context)
        # every task that reports penetration makes a candidate infeasible through the score
        self._collision_tasks = [task for task in self._tasks if isinstance(task, TrajectoryCollisionTask)]
        context[ctx.COLLISION_TASKS] = self._collision_tasks
        if not self._tasks:
            raise ValueError("Trajectory optimizer requires at least one enabled task.")
        self._settle_scale = wp.ones(actual_batch, dtype=wp.float32, device=self.device)
        # FK of the start configurations, for settling; a floating base needs its own pose buffer
        base_start = se3_identity(shape=(actual_batch,), device=self.device) if self._base_dofs else None
        self._start_state = self.robot.state(q=self._sq_buf, T_world_base=base_start)
        # a joint goal's pose: forward kinematics of each candidate's goal configuration, which pose-goal tasks read
        self._goal_state = self.robot.state(q=self._target_q_per_candidate_buf, T_world_base=base_start)
        rows = np.arange(actual_batch)[:, None] * self.robot.spec.num_links + np.asarray(self.target_link_indices)
        self._goal_link_rows = wp.array(rows.ravel().astype(np.int32), dtype=wp.int32, device=device)
        score = None
        if self.num_candidates > 1 and config.score_terms:
            scores = _build_terms(config.score_terms, var, context)
            score = scores[0] if len(scores) == 1 else CompositeTask(scores)
        tangent_dim = self.num_frames * self.num_dofs
        acceleration = config.control_space == "acceleration"
        sampler = dataclasses.replace(
            cast(ColoredGaussianSamplerConfig, config.solver.samplers[0]),
            var_key=VarKey.ACTION if acceleration else VarKey.ROBOT,
        )
        if acceleration:
            dt_schedule = np.full(self.num_frames, self._dt, dtype=np.float32)
            if config.dt_max > self._dt:
                base_frames = self.num_frames // 2
                dt_schedule[base_frames:] = np.linspace(
                    self._dt, config.dt_max, self.num_frames - base_frames, dtype=np.float32
                )
            self.dt_schedule = wp.array(dt_schedule, dtype=wp.float32, device=self.device)
            self.q0 = wp.zeros((actual_batch, self.num_dofs), dtype=wp.float32, device=self.device)
            self.prev_q = wp.zeros_like(self.q0)
            self.v0 = wp.zeros_like(self.q0)
            spec = self.robot.spec
            velocity_limit = spec.actuated_joint_velocity_limits.astype(np.float32)
            self.velocity_limit = wp.array(velocity_limit, dtype=wp.float32, device=self.device)
            limits = spec.actuated_joint_limits.astype(np.float32)
            self.q_lower = wp.array(limits[:, 0] - config.q_limit_margin, dtype=wp.float32, device=self.device)
            self.q_upper = wp.array(limits[:, 1] + config.q_limit_margin, dtype=wp.float32, device=self.device)
            if self._base_dofs:
                self.base_T0 = se3_identity(shape=(actual_batch,), device=self.device)
                self.base_prev_T = se3_identity(shape=(actual_batch,), device=self.device)
                self.base_twist0 = wp.zeros(actual_batch, dtype=wp_vec6, device=self.device)
            self._action = VectorVar(wp.zeros((actual_batch, tangent_dim), dtype=wp.float32, device=self.device))
            self._action_scratch = wp.zeros_like(self._action.x)
            self._var: VarValues = AccelerationTrajectory(self._action, self._state, weakref.WeakMethod(self._rollout))
        else:
            self._shift_scratch = wp.zeros_like(self._state.q)
            self._var = VarValues({VarKey.ROBOT: self._state})
            options = dataclasses.asdict(sampler)  # positions draw white noise: knots and the filter shape actions
            for key in ("knot_ratio", "num_knots", "noise_filter"):
                options.pop(key)
            sampler = GaussianSamplerConfig(**options)
        solver_config = dataclasses.replace(  # plus one sample after the last score, so every score moves the mean
            config.solver, cuda_graph_mode="full" if self._use_cuda_graph else "none", samplers=[sampler]
        )
        self._solver = PopulationSolver(
            [self._tasks],
            solver_config,
            self.device,
            [score] if score is not None else None,
            active_dof_mask=self._active_dof_mask_wp,
        )
        self._solver.setup(self._var)
        self._initialized = True
        if self.num_candidates == 1:
            zero_q = wp.zeros((batch_size, self.num_dofs), dtype=wp.float32, device=self.device)
            zero_traj = wp.zeros_like(self._state.q)
            zero_scene = wp.zeros(batch_size, dtype=wp.int32, device=self.device)
            base = se3_identity(shape=(batch_size,), device=self.device) if self._base_dofs else None
            self.solve(zero_q, None, None, zero_traj, zero_q, zero_scene, start_T_world_base=base)
        wp.synchronize_device(self.device)

    def solve(
        self,
        start_q: wp.array,
        T_world_target: Optional[wp.array],
        target_q: Optional[wp.array],
        traj_q_init: Optional[wp.array],
        target_q_per_candidate: wp.array,
        scene_indices: wp.array,
        online_state: Optional["OnlineState"] = None,
        explore_scale: Optional[Union[float, wp.array]] = None,
        start_T_world_base: Optional[wp.array] = None,
    ) -> Tuple[RobotState, wp.array]:
        """Solve from an initial trajectory or update caller-owned MPPI state.

        Args:
            start_q: Batched current configurations.
            T_world_target: Optional batched pose goals.
            target_q: Optional batched joint-space goals.
            traj_q_init: Initial trajectories for a cold solve.
            target_q_per_candidate: Goal configurations for every candidate.
            scene_indices: Scene index for each query.
            online_state: Optional caller-owned state for online updates.
            explore_scale: Optional scalar or per-query exploration scale.

        Returns:
            The selected robot state and final ranking costs.

        """
        batch_size = start_q.shape[0]
        if not self._initialized:
            self.warmup(batch_size)
        elif batch_size != self.batch_size:
            raise ValueError(f"Batch size mismatch: expected {self.batch_size}, got {batch_size}")
        self._apply_active_dof_mask()  # the locked frames may have moved on the device
        cold_start = self._set_inputs(
            start_q,
            T_world_target,
            target_q,
            traj_q_init,
            target_q_per_candidate,
            scene_indices,
            online_state,
            explore_scale,
            start_T_world_base,
        )
        if self._config.fresh_noise and online_state is not None:  # a replayed sequence would repeat its misses
            online_state.epoch += 1
            cast(GaussianSampler, self._solver.samplers[0]).epoch.fill_(online_state.epoch)
        best_var, costs = self._solver.solve(self._var)
        robot_state = cast(RobotState, best_var.get(VarKey.ROBOT))
        if cold_start:
            self.robot.forward_kinematics(robot_state)
        if online_state is not None:
            if online_state.prev_q_traj is None:
                online_state.prev_q_traj = wp.empty_like(robot_state.q)
            wp.copy(online_state.prev_q_traj, robot_state.q)
            if self._config.control_space == "acceleration":
                if cold_start:
                    online_state.prev_start_q = wp.zeros_like(self.prev_q)
                    online_state.mean_action = wp.zeros_like(self._action.x)
                wp.copy(cast(wp.array, online_state.prev_start_q), self.prev_q)
                wp.copy(cast(wp.array, online_state.mean_action), self._action.x)
        return robot_state, costs
