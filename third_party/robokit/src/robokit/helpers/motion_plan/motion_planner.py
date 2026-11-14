# pyright: reportArgumentType=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportIndexIssue=false, reportOperatorIssue=false
import copy
import dataclasses
import itertools
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import warp as wp

from robokit.geom import WarpScene
from robokit.geom.covering_spheres import _compute_covering_spheres
from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig
from robokit.helpers.motion_plan.evaluator import MotionPlanEvaluator, MotionPlanMetrics
from robokit.helpers.motion_plan.gradient_trajectory_optimizer import (
    GradientTrajectoryOptimizer,
    GradientTrajectoryOptimizerConfig,
    build_default_gradient_trajectory_optimizer_config,
)
from robokit.helpers.motion_plan.motion_planner_kernels import (
    _append_q_history_kernel,
    _blend_kernel,
    _collision_mask_kernel,
    _expand_ik_goal_candidates_kernel,
    _frame_flags_kernel,
    _frame_need_kernel,
    _held_radii_kernel,
    _hold_kernel,
    _min_distance_kernel,
    _motion_time_kernel,
    _pick_ik_kernel,
    _pin_inactive_joints_kernel,
    _resample_base_kernel,
    _resample_kernel,
    _resplit_kernel,
    _score_condition_kernel,
    _seed_base_kernel,
    _seed_kernel,
    _set_best_retry_result_kernel,
    _set_detour_seed_kernel,
    _waypoint_rows_kernel,
)
from robokit.helpers.motion_plan.mppi_trajectory_optimizer import (
    MppiTrajectoryOptimizer,
    MppiTrajectoryOptimizerConfig,
)
from robokit.helpers.motion_plan.plan_goal import PlanGoal
from robokit.helpers.motion_plan.trajectory_retimer import (
    RetimeConfig,
    TimingVar,
    TrajectoryRetimer,
    _interpolate_trajectory_kernel,
)
from robokit.lie.se3 import se3_from_matrix, se3_to_matrix
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.opt.var_values import VarKey, VarValues, VectorVar
from robokit.robo.robot import Robot
from robokit.robo.robot_spec import JointType
from robokit.robo.robot_state import RobotState
from robokit.terms.composite_task import CompositeTask
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.relative_frame_task import RelativeFrameTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.sparse.trajectory_self_collision_task import TrajectorySelfCollisionTask
from robokit.terms.trajectory_task import TrajectoryTask
from robokit.utils import warp_utils
from robokit.utils.warp_utils import resolve_device, wp_device_type, wp_vec7
from robokit.xform.numpy.rotation_conversions import normalize_quaternion
from robokit.xform.numpy.transforms import pose7_to_tf_mat, tf_mat_to_pose7


if TYPE_CHECKING:
    import torch


# --- results ---------------------------------------------------------------
ArrayT = TypeVar("ArrayT")


@dataclass
class OnlineState:
    """Previous IK goal plus executed history or MPPI state retained between updates."""

    prev_q_traj: Optional[wp.array] = None
    prev_goal_q: Optional[wp.array] = None
    prev_start_q: Optional[wp.array] = None
    mean_action: Optional[wp.array] = None
    q_history: Optional[wp.array] = None
    history_count: int = 0
    epoch: int = 0  # online MPPI updates so far, when each draws fresh noise


@dataclass
class MotionPlanResult(Generic[ArrayT]):
    """Planner output with offline timing or online state.

    Offline: `q_traj` `(batch, frames, dofs)` with `frames = num_timesteps + 1`, `motion_time` `(batch,)` (s) and
    `waypoint_indices` each waypoint's frame in `q_traj` (one per waypoint, 0 first, the same for every row).
    """

    q_traj: ArrayT
    T_world_base_traj: Optional[ArrayT] = None
    motion_time: Optional[ArrayT] = None
    online_state: Optional[OnlineState] = None
    waypoint_indices: Optional[Tuple[int, ...]] = None
    segment_fixed_duration: Optional[Tuple[Optional[float], ...]] = (
        None  # seconds of fixed segments, None where retimed
    )
    segment_duration: Optional[ArrayT] = None  # each plan's segment seconds `(batch, segments)`, when optimized

    def interpolate(self, dt: float, speed_scale: float = 1.0) -> ArrayT:
        """Joint trajectories sampled every `dt` (s) or slightly less across `motion_time`.

        Each segment between waypoints spreads `ceil(its time / dt)` steps evenly over its frames, so every
        waypoint is a sample; a plan then holds its last frame, so a batch shares one length. A segment with a
        fixed `segment_fixed_duration` takes that time; the others run at the plan's retimed step or `segment_duration`.

        Args:
            dt: Control step (s).
            speed_scale: Fraction of the plan's speed; 0.5 plays its retimed segments over twice their time.

        Returns:
            Joint positions with shape `(batch, samples, dofs)`, NumPy or Torch as `q_traj`.

        Example:
            >>> import numpy as np
            >>> q_traj = np.array([[[0.0], [1.0], [2.0]]], np.float32)
            >>> MotionPlanResult(q_traj, motion_time=np.array([1.0], np.float32)).interpolate(0.25)[0, :, 0]
            array([0. , 0.5, 1. , 1.5, 2. ], dtype=float32)
        """
        q_traj = cast(Any, self.q_traj)
        is_numpy = isinstance(q_traj, np.ndarray)
        convert = (
            (lambda x: wp.from_numpy(np.ascontiguousarray(x, np.float32), dtype=wp.float32, device="cpu"))
            if is_numpy
            else (lambda x: wp.from_torch(x.float().contiguous(), dtype=wp.float32))
        )
        anchors = np.asarray(self.waypoint_indices or (0, q_traj.shape[1] - 1), np.int32)
        durations = np.array(
            [-1.0 if d is None else d for d in self.segment_fixed_duration or [None] * (len(anchors) - 1)]
        )
        fixed = durations > 0
        moving_frames = max(int((np.diff(anchors) * ~fixed).sum()), 1)
        # the retimed segments' time, spread over their frames
        frame_step = (cast(Any, self.motion_time) - float(durations[fixed].sum())) / speed_scale / moving_frames
        q, step = convert(q_traj), convert(frame_step)
        step_max = float(frame_step.max())  # the longest plan takes the most samples
        plans = np.tile(durations, (q.shape[0], 1))  # each plan's; optimized ones one plan at a time
        if self.segment_duration is not None:
            plans = np.where(fixed, durations, np.asarray(cast(Any, self.segment_duration).tolist()) / speed_scale)
        samples = int(np.ceil(np.where(plans > 0, plans / dt, np.diff(anchors) * (step_max / dt))).sum(1).max()) + 1
        out = wp.empty((q.shape[0], samples, q.shape[2]), dtype=wp.float32, device=q.device)
        for rows in [slice(None)] if self.segment_duration is None else [slice(b, b + 1) for b in range(q.shape[0])]:
            wp.launch(
                _interpolate_trajectory_kernel,
                dim=out[rows].shape,
                inputs=[
                    q[rows],
                    step[rows],
                    dt,
                    wp.array(anchors, dtype=wp.int32, device=q.device),
                    wp.array(plans[rows][0], dtype=wp.float32, device=q.device),
                ],
                outputs=[out[rows]],
                device=q.device,
            )
        return cast(ArrayT, out.numpy() if is_numpy else wp.to_torch(out))


# --- configuration ---------------------------------------------------------
def build_goal_score(goal_tolerance: Tuple[float, float] = (0.005, 0.05)) -> CompositeTask:
    """Goal IK's score, ranking goals within `goal_tolerance` (m, rad) of the target, the joint limits and collision-free first, then the nearest the rest configuration; the planner keeps it for several goals."""
    return CompositeTask(
        [
            FrameTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, residual_mode="distance"),
            PositionLimit(weight=1.0),
            SceneCollisionTask(
                scene=IKContext.SCENE,
                scene_indices=IKContext.SCENE_INDICES,
                weight=1.0,
                margin=0.0,
                penalty="plain",
                movable_only=True,
            ),
            ConfigurationTask(rest_q=IKContext.REST_Q, weight=1.0),
        ],
        # thresholds are inclusive: the strict tolerance's next float below
        thresholds=[np.nextafter(np.array(goal_tolerance, np.float32), np.float32(-np.inf)), 0.0, 0.0, None],
        num_candidates=IKContext.NUM_SOLUTIONS,
    )


def build_default_motion_plan_ik_config() -> IKSolverConfig:
    """Build the default goal-IK configuration for motion planning."""
    return (
        IKSolverConfig(
            solver=PopulationSolverConfig(
                stages=[
                    Stage(6, 64, LMOptimizerConfig(lm_lambda=10.0)),
                    Stage(10, 4, LMOptimizerConfig(lm_lambda=1.0)),
                    Stage(0, 1, LMOptimizerConfig(lm_lambda=1.0)),
                ],
                cuda_graph_mode="full",
            )
        )
        .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
        .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
        .add(PositionLimit(weight=50.0))
        .add(
            SceneCollisionTask(
                scene=IKContext.SCENE,
                scene_indices=IKContext.SCENE_INDICES,
                weight=20.0,
                margin=0.01,
            ),
            name="ik_collision",
        )
        .add_score(build_goal_score(), name="goal_score")
    )


@dataclass
class AttachedObject:
    """A box (half extents `size`) or z-axis cylinder (radius, half height) that `link` carries, posed in its frame by
    4x4 `T_link_object`: checked against the world and the robot, but `link`'s body, the bodies one joint from it (the
    wrist, the fingers) and `collision_ignored_links`, over `segments` (indices; None: all). `link` must be rigid with
    a target link.
    """

    link: str
    shape: Literal["box", "cylinder"]
    size: Tuple[float, ...]
    T_link_object: Optional[np.ndarray] = None
    collision_ignored_links: Tuple[str, ...] = ()
    segments: Optional[Tuple[int, ...]] = None


@dataclass
class FixedRelativePose:
    """`link_b`'s pose in `link_a`'s frame, held over `segments` (indices; None: all) by the gradient stages at its
    value at waypoint `at_waypoint`, else at each segment's first waypoint (e.g. two hands carrying one object).
    """

    link_a: str
    link_b: str
    at_waypoint: Optional[int] = None
    segments: Optional[Tuple[int, ...]] = None


@dataclass
class MotionPlannerConfig:
    """Configuration for offline or online motion planning."""

    # --- plan ---
    num_timesteps: int = 32
    """Steps per plan; a plan has one more frame."""
    dt: float = 0.25
    """Planning step (s)."""
    max_attached_spheres: int = 0
    """Collision spheres reserved on each target link for `attached_objects`, which every solve places there."""
    max_fixed_relative_poses: int = 0
    """Link pairs reserved for `fixed_relative_poses`, which every solve sets."""
    fixed_relative_pose_weight: Tuple[float, float] = (3000.0, 1000.0)
    """Position and rotation weights of `fixed_relative_poses`, per m and per rad of drift."""
    acceleration_limit: float = 15.0
    """Joint acceleration limit (rad/s²): the stages' bounds and ranking, MPPI rollouts, retimes and resplits."""
    jerk_limit: float = 500.0
    """Joint jerk limit (rad/s³): the stages' bounds and ranking, retimes and resplits."""
    duration_weight: float = 0.0
    """>0: the last stage also optimizes each free segment's duration, from the query's step, at this cost per second
    of plan time; its derivative bounds use the optimized durations, which the returned retime only lengthens."""

    # --- goal ---
    goal_ik: IKSolverConfig = field(default_factory=build_default_motion_plan_ik_config)
    """Goal IK, which turns pose goals into joint goals; fewer goals than trajectory candidates repeat across them."""

    # --- seeding ---
    init_sample_range: float = 0.2
    """Spread of the sampled seed candidates, as a fraction of each joint range; MPPI stages sample their own."""
    init_detour_samples: Tuple[int, int] = (16, 64)
    """Random-midpoint detours per detour candidate (first round, retries); the shortest collision-free one seeds it."""

    # --- stages ---
    stages: Tuple[Union[GradientTrajectoryOptimizerConfig, MppiTrajectoryOptimizerConfig], ...] = field(
        default_factory=lambda: (build_default_gradient_trajectory_optimizer_config(),)
    )
    """Optimizer stages, run in order on the candidate population every planning round."""
    stage_dt_scale: float = 0.0
    """>0: each later stage optimizes at this multiple of the previous stage's fastest feasible retimed step."""
    stage_retime: Optional[RetimeConfig] = None
    """Retime between stages (`stage_dt_scale`); its `minimum_dt` floors the later stages' steps."""
    stage_retries: int = 0
    """Reruns of a failed last stage whose predecessor passed, at a growing step; each query keeps its better plan."""
    stage_retry_iters: int = 0
    """Iterations of each stage retry."""
    stage_retry_dt_growth: float = 1.025
    """Factor each stage retry grows the step by."""

    # --- retry rounds ---
    max_retries: int = 1
    """Rounds retrying failed plans with new goals and seeds; needs a gradient first stage with several candidates."""

    # --- output ---
    retime: Optional[RetimeConfig] = None
    """Retime of the returned plan; None keeps the planning step."""
    resplit_waypoints: bool = True
    """Resplit a waypoint plan's frames by the time each part needs, so a short segment takes fewer: its straight lines' before the solve, and its paths' after the first stage, which the later gradient stages smooth; False keeps the even split."""
    cuda_graph: Literal["none", "stages", "solve"] = "stages"
    """CUDA graphs: none, each stage's and goal IK's solve, or the whole NumPy solve with its device-side retries; a
    capture makes a planner's first solve take seconds longer."""


class WaypointLayout(NamedTuple):
    """A waypoint plan's waypoints, where they sit and how its segments behave, as the solve reads them."""

    q: Optional[wp.array]  # each query's waypoint joints before the goal, `(batch, >= waypoints - 1, dofs)`
    frames: wp.array  # each waypoint's frame
    locked: wp.array  # each segment: 1 keeps it on the straight line between its waypoints
    known_base: wp.array  # each waypoint: 1 when its base pose is known; the base between them is interpolated
    ik_picks: list  # `_solve_waypoint_ik`'s picks, each with its answers
    check_collision: Optional[wp.array]  # each segment: 1 checks world collision
    lock_goal: int  # 1 holds the goal frame: a joint goal's, or a locked last segment's
    resplit: Optional[Tuple[wp.array, wp.array]]  # with a resplit: the fixed segments, and the new frames
    # each segment's checked spheres and relative poses' reference waypoints, and each waypoint's named links as bits
    segment_items: Optional[Tuple[wp.array, wp.array, wp.array]] = None  # None: start and goal only, never split


# --- planner ---------------------------------------------------------------
class MotionPlanner:
    """Batched motion planning with goal IK, trajectory optimization stages, and retry.

    Lifecycle:
        1. `__init__` copies configuration and constructs goal IK, the optimizer stages, and their retiming.
        2. `warmup(batch_size)` fixes the batch size and allocates all batch-shaped state.
        3. `solve_offline(...)`, which `solve_offline_numpy` and `solve_offline_torch` wrap, places the waypoints, builds candidates, solves and retries trajectories, then retimes them with the last stage's `retime` and computes `motion_time`.
        4. `solve_online(...)` runs a fresh gradient solve with executed-history smoothing, or updates the retained MPPI action distribution.

    Args:
        config: Planner configuration.
        robot: Robot model.
        target_links: One target link or a sequence, using names or indices.
        scene: Optional collision scene.
        device: Warp device.
    """

    # --- setup ---
    def __init__(
        self,
        config: MotionPlannerConfig,
        robot: Robot,
        target_links: Union[str, int, Sequence[Union[str, int]]],
        *,
        scene: Optional[WarpScene] = None,
        device: Optional[wp_device_type] = None,
    ):
        # copies the planner mutates; completed preset terms hold device arrays, so they are shared, not deep-copied
        self.config = copy.copy(config)
        self.config.stages = tuple(dataclasses.replace(c, terms=list(c.terms)) for c in config.stages)
        self.device = resolve_device(device)
        links = [target_links] if isinstance(target_links, (str, int)) else list(target_links)
        self.target_link_indices = [robot.link_names.index(x) if isinstance(x, str) else int(x) for x in links]
        # the robot with `max_attached_spheres` spheres on a link welded to each target link, after its own links and
        # spheres; off (radius -1e9) but where a solve places an attached object
        slots, spec, targets = config.max_attached_spheres, robot.spec, self.target_link_indices
        self._num_robot_links, self._num_robot_spheres = spec.num_links, len(spec.collision_sphere_radii)
        if slots:
            count = len(targets)
            spec = dataclasses.replace(
                spec,
                link_names=[*robot.link_names, *(f"{robot.link_names[t]} attached" for t in targets)],
                link_parent_joint_indices=np.r_[
                    spec.link_parent_joint_indices, spec.link_parent_joint_indices[targets]
                ],
                link_masses=np.r_[spec.link_masses, np.zeros(count)],
                link_local_com_positions=np.r_[spec.link_local_com_positions, np.zeros((count, 3))],
                local_collision_sphere_centers=np.r_[spec.local_collision_sphere_centers, np.zeros((count * slots, 3))],
                collision_sphere_radii=np.r_[spec.collision_sphere_radii, np.full(count * slots, -1.0e9)],
                collision_spheres_link_indices=np.r_[
                    spec.collision_spheres_link_indices, np.repeat(spec.num_links + np.arange(count), slots)
                ].astype(np.int32),
            )
            robot = Robot(spec)
        self.robot = robot
        # the gradient stages hold `max_fixed_relative_poses` link pairs' relative poses at each frame's reference
        poses = config.max_fixed_relative_poses
        self._reference_frames = wp.full((config.num_timesteps + 1, max(1, poses)), -1, wp.int32, self.device)
        self._relative: Optional[RelativeFrameTask] = None
        if poses:
            weights = config.fixed_relative_pose_weight
            relative = RelativeFrameTask(robot, [(0, 0)] * poses, self._reference_frames, *weights)
            self._relative, term = relative, TrajectoryTask(relative, config.num_timesteps + 1, name="relative")
            term.init_buffers(self.device)
            self.config.stages = tuple(
                c if c.method == "mppi" else dataclasses.replace(c, terms=[*c.terms, term]) for c in self.config.stages
            )
        # each frame's checked spheres: all but attached objects outside their segments; the score's path checks too
        num_spheres = max(1, len(robot.spec.collision_sphere_radii))
        self._sphere_active = wp.ones((config.num_timesteps + 1, num_spheres), wp.int32, self.device)
        # a floating base plans its pose along with the joints, from goal IK that places it
        self._floating = self.config.goal_ik.enable_T_world_base

        self._num_frames = self.config.num_timesteps + 1
        self._solve_graph: Optional[tuple] = None  # the whole solve's CUDA graph key, inputs and graph
        self._batch_size: Optional[int] = None
        # geometries must not change after construction; update them in place with `geom.update(...)`
        self._scene = scene if scene is not None else WarpScene(1, self.device)
        self._active_joint_mask_np = np.ones(robot.num_actuated_joints, dtype=np.float32)
        self._collision_cols = np.arange(robot.num_actuated_joints, dtype=np.int32)  # the collision Jacobians' joints
        self._active_joint_mask_wp = wp.ones(robot.num_actuated_joints, dtype=wp.float32, device=self.device)
        self._joint_limits_wp = wp.from_numpy(
            robot.spec.actuated_joint_limits.astype(np.float32), dtype=wp.float32, device=self.device
        )
        self._velocity_limits_wp = wp.from_numpy(
            robot.spec.actuated_joint_velocity_limits.astype(np.float32), dtype=wp.float32, device=self.device
        )
        # each stage's retime: `stage_retime` between stages, `retime` for the returned plan
        last = len(self.config.stages) - 1
        stage_retimes = [self.config.stage_retime] * last + [self.config.retime]
        limits = (self.config.acceleration_limit, self.config.jerk_limit)
        self._retimers = [
            TrajectoryRetimer(robot, self._num_frames, self.config.dt, *limits, r, device) for r in stage_retimes
        ]
        if self.config.stage_dt_scale > 0.0 and last > 0 and self.config.stage_retime is None:
            raise ValueError("stage_dt_scale needs a retime between stages (stage_retime).")
        if self.config.stage_retries > 0 and (
            self.config.stage_dt_scale <= 0.0
            or self.config.stage_retry_iters < 1
            or any(c.method == "mppi" for c in self.config.stages[1:])
        ):
            raise ValueError("stage_retries needs stage_dt_scale > 0, stage_retry_iters >= 1 and gradient stages.")
        # each candidate's segment log steps, which the last stage optimizes: a segment spans at least two steps
        self._segments = self.config.num_timesteps // 2 if self.config.duration_weight > 0.0 else 0
        if self._segments and any(c.method == "mppi" for c in self.config.stages):
            raise ValueError("duration_weight needs gradient stages.")

        def build(
            config: Union[GradientTrajectoryOptimizerConfig, MppiTrajectoryOptimizerConfig], index: int
        ) -> Union[GradientTrajectoryOptimizer, MppiTrajectoryOptimizer]:
            args = (config, robot, self.target_link_indices, self._num_frames, self.config.dt)
            timing = dict(segments=self._segments, duration_weight=self.config.duration_weight) if index == last else {}
            options = dict(
                enable_T_world_base=self._floating,
                scene=self._scene,
                use_cuda_graph=self.config.cuda_graph != "none",
                acceleration_limit=self.config.acceleration_limit,
                sphere_mask=self._sphere_active if slots else None,
                device=self.device,
            )
            if config.method == "mppi":
                return MppiTrajectoryOptimizer(*args, **options)
            return GradientTrajectoryOptimizer(
                *args, **options, jerk_limit=self.config.jerk_limit, retime=stage_retimes[index], **timing
            )

        self._optimizers = [build(config, index) for index, config in enumerate(self.config.stages)]
        self._retry_optimizers = {  # the last stage's retry: no later stage reads the population it rewrites
            last: build(
                dataclasses.replace(
                    config,
                    solver=dataclasses.replace(  # the last solver stage runs what the earlier ones leave of the retry
                        config.solver,
                        stages=[
                            *config.solver.stages[:-1],
                            dataclasses.replace(
                                config.solver.stages[-1],
                                iters=self.config.stage_retry_iters - sum(s.iters for s in config.solver.stages[:-1]),
                            ),
                        ],
                    ),
                ),
                last,
            )
            for config in self.config.stages[last:]
            if self.config.stage_retries > 0 and last > 0
        }
        self._trajectory_optimizer = self._optimizers[0]

        # Retry ranks a gradient stage's candidates by their scores, so it needs several pose-goal candidates.
        cfg = self.config.stages[0]
        num_trajectory_candidates = self._trajectory_optimizer.num_candidates
        self._retry_enabled = self.config.max_retries > 0 and num_trajectory_candidates > 1 and cfg.method != "mppi"

        # Goal IK gives at most one goal per trajectory candidate, repeated across them; retries reserve more groups.
        goal_ik = self.config.goal_ik
        self._num_ik_goal_candidates = min(goal_ik.solver.num_candidates[-1], num_trajectory_candidates)
        if num_trajectory_candidates % self._num_ik_goal_candidates != 0:
            raise ValueError("Trajectory candidate count must be divisible by IK goal candidate count.")
        self._num_ik_goal_candidates_total = self._num_ik_goal_candidates * (
            1 + self.config.max_retries if self._retry_enabled else 1
        )
        total = self._num_ik_goal_candidates_total
        candidates = [max(c, total) for c in goal_ik.solver.num_candidates[:-1]] + [total]  # a beam never grows
        # several goals rank by the config's goal score: feasible first, then the nearest the start
        self._goal_score = goal_ik.task("goal_score") if total > 1 else None
        ik_config = dataclasses.replace(
            goal_ik,
            score_terms=[self._goal_score] if total > 1 else [],
            solver=dataclasses.replace(
                goal_ik.solver,
                stages=[dataclasses.replace(s, num_candidates=c) for s, c in zip(goal_ik.solver.stages, candidates)],
                rank_final=total > 1,
            ),
        )
        # the goal IK, public for IK alone: the planner's scene, and with several answers per query, best first
        self.ik = IKSolver(
            ik_config, robot=robot, target_links=self.target_link_indices, device=self.device, scene=self._scene
        )
        # the robot's sphere radii and a copy: each IK solve of a plan sets the attached objects its waypoint does not
        # hold not to collide (radius -1e9) while it runs
        radii = robot.spec.get_tensors(str(self.device)).collision_sphere_radii
        self._radii = (radii, wp.clone(radii)) if slots else None
        self._ik_goal_q: Optional[wp.array] = None
        self._ik_goal_state: Optional[RobotState] = None
        self._evaluator: Optional[MotionPlanEvaluator] = None

    def warmup(self, batch_size: int):
        """Fix the planner batch size and allocate its batch-shaped buffers.

        Args:
            batch_size: Number of queries solved together.

        """
        if self._batch_size is not None:
            if batch_size != self._batch_size:
                raise ValueError(f"Batch size mismatch: expected {self._batch_size}, got {batch_size}")
            return

        num_dofs = self.robot.num_actuated_joints
        n_total = batch_size * self._trajectory_optimizer.num_candidates
        self._default_scene_indices = wp.zeros(batch_size, dtype=wp.int32, device=self.device)
        self._target_q_per_candidate_buf = wp.empty((n_total, num_dofs), dtype=wp.float32, device=self.device)
        self._target_base_per_candidate_buf = wp.empty(n_total, dtype=wp_vec7, device=self.device)
        self._no_valid_count = wp.zeros(batch_size, dtype=wp.int32, device=self.device)
        self._ik_best_indices = wp.array(
            np.arange(batch_size, dtype=np.int32) * self.ik.num_solutions, dtype=wp.int32, device=self.device
        )

        def trajectories(rows: int) -> RobotState:  # joints, and base poses when the base floats
            base = np.tile(np.float32([0, 0, 0, 1, 0, 0, 0]), (rows, self._num_frames, 1))
            return self.robot.state(
                q=wp.zeros((rows, self._num_frames, num_dofs), dtype=wp.float32, device=self.device),
                T_world_base=wp.array(base, dtype=wp_vec7, device=self.device) if self._floating else None,
            )

        self._traj_init = trajectories(n_total)
        # each guess's waypoints by waypoint count, and a start-goal layout
        self._waypoint_bufs = {
            2: (
                wp.empty((n_total, 2, num_dofs), dtype=wp.float32, device=self.device),
                wp.empty((n_total, 2), dtype=wp_vec7, device=self.device),
            )
        }
        self._start_goal_layout = WaypointLayout(
            q=None,  # each solve's starts
            frames=wp.array([0, self._num_frames - 1], dtype=wp.int32, device=self.device),
            locked=wp.zeros(1, dtype=wp.int32, device=self.device),
            known_base=wp.ones(2, dtype=wp.int32, device=self.device),
            ik_picks=[],
            check_collision=None,
            lock_goal=1,
            resplit=None,
        )
        self._final_state = trajectories(batch_size)
        self._final_timing = TimingVar.full(batch_size, self.config.dt, self.device)
        self._motion_time = wp.empty(batch_size, dtype=wp.float32, device=self.device)
        if batch_size > 1 and self.ik.config.solver.cuda_graph_mode == "full":
            self.ik.config.solver.cuda_graph_mode = "iter"
        self.ik.warmup(batch_size)
        frames = self._num_frames
        if self.config.stages[0].method == "mppi":
            cast(MppiTrajectoryOptimizer, self._trajectory_optimizer).warmup(batch_size)
            optimizers = [self._trajectory_optimizer]
        else:
            self._timing = TimingVar.full(batch_size, self.config.dt, self.device)
            population = {VarKey.ROBOT: trajectories(n_total), VarKey.TIMING: self._timing}
            self._population = self._last_population = VarValues(population)
            if self._segments:  # the last stage's: each candidate's segment log steps
                steps = wp.zeros((n_total, self._segments), dtype=wp.float32, device=self.device)
                steps = VectorVar(steps, unit=3.0, max_step=0.05)  # moves 3x a joint's unit, at most 5% of a step
                self._last_population = VarValues({**population, VarKey.SEGMENT_STEP: steps})
                self._final_log_step = wp.zeros((batch_size, self._segments), dtype=wp.float32, device=self.device)
            optimizers = [*self._optimizers, *self._retry_optimizers.values()]
            for optimizer in optimizers:
                optimizer.warmup(self._last_population if optimizer.segments else self._population)
            self._fastest_step = wp.empty(batch_size, dtype=wp.float32, device=self.device)
            self._stage_condition = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._retry_condition_stage = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._resplit_bufs = (
            wp.empty((n_total, frames, num_dofs), dtype=wp.float32, device=self.device),  # paths held, then resampled
            *(wp.empty(frames, dtype=wp.float32, device=self.device) for _ in range(3)),  # need, elapsed, at
            wp.empty((n_total, frames), dtype=wp_vec7, device=self.device),  # resampled base poses
        )
        # each optimizer's collision masks, which a waypoint split rewrites per frame
        self._checked_frames = wp.empty(frames, dtype=wp.int32, device=self.device)
        self._frame_masks = {
            optimizer: [
                (task, wp.array(task._frame_sphere_mask_np, dtype=wp.bool, device=self.device))
                for task in optimizer._collision_tasks
                if isinstance(task, TrajectoryCollisionTask)
            ]
            for optimizer in optimizers
        }
        for task, _ in (m for masks in self._frame_masks.values() for m in masks):
            if task.frame_sphere_mask is None:  # built on first use, after a split first writes it
                task.init_buffers(self.device)
        spec = self.robot.spec
        self._world_collision = bool(self._scene.geoms) and spec.has_collision_spheres
        num_spheres = len(spec.collision_sphere_radii)
        # the self-collision checks of the stages and their scores, which a solve tells the attached objects' radii and
        # ignored pairs; the stages' also which spheres each frame checks
        stage_tasks = [t for o in optimizers for t in o._tasks if isinstance(t, TrajectorySelfCollisionTask)]
        scores = [s for o in optimizers for s in o._solver.score_terms if s is not None]
        scores = [getattr(t, "self_collision_task", None) for s in scores for t in getattr(s, "tasks", [s])]
        tasks = list(dict.fromkeys([*stage_tasks, *(t for t in scores if t is not None)]))
        self._self_collision_tasks = tasks if self.config.max_attached_spheres else []
        self._pair_active = wp.ones(max([1, *(t.num_pairs for t in tasks)]), wp.int32, self.device)
        for task in self._self_collision_tasks:
            task.pair_mask = self._pair_active
            task.sphere_mask = self._sphere_active if task in stage_tasks else task.sphere_mask
        self._eff_radii = wp.from_numpy(
            np.where(spec.collision_spheres_link_indices > 0, spec.collision_sphere_radii, -1.0e9).astype(np.float32),
            dtype=wp.float32,
            device=self.device,
        )

        def distance_bufs(num_configs: int) -> Tuple[RobotState, wp.array, wp.array]:  # state, sphere dists, scenes
            return (
                self.robot.state(q=wp.zeros((num_configs, num_dofs), dtype=wp.float32, device=self.device)),
                wp.empty(num_configs * num_spheres, dtype=wp.float32, device=self.device),
                wp.empty(num_configs * num_spheres, dtype=wp.int32, device=self.device),
            )

        self._seed_bufs = None
        self._start_bufs = None
        self._start_distance = wp.full(batch_size, 1.0e9, dtype=wp.float32, device=self.device)
        if self._retry_enabled:
            self._retry_condition = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._retry_best_score = wp.empty(batch_size, dtype=wp.float32, device=self.device)
            if self._world_collision:
                self._seed_bufs = {
                    per: (
                        wp.empty((n_total * per, self._num_frames, num_dofs), dtype=wp.float32, device=self.device),
                        *distance_bufs(n_total * per),
                        wp.empty(n_total * per, dtype=wp.float32, device=self.device),
                    )
                    for per in self.config.init_detour_samples
                }
                for _, state, dists, scene_pp, distance in self._seed_bufs.values():  # allocates: never in a branch
                    self._obstacle_distance(state, self._default_scene_indices, dists, scene_pp, distance)
                self._start_bufs = distance_bufs(batch_size)
        self._batch_size = batch_size

    def set_active_joint_mask(self, joint_mask: Sequence[float]):
        """Set which actuated joints IK and trajectory optimization may change; before the first solve it also limits the trajectory collision Jacobians to the active joints, which later masks must stay within.

        Args:
            joint_mask: One value per actuated joint; zero pins the joint.
        """
        joint_mask_np = np.asarray(joint_mask, dtype=np.float32)
        cols = np.flatnonzero(joint_mask_np).astype(np.int32)
        if self._batch_size is None:  # the stages' templates are still unbuilt: a copy of each with these columns
            self._collision_cols = cols
            for stage in self.config.stages:
                for index, term in enumerate(stage.terms):
                    if isinstance(term, TrajectoryCollisionTask) and term.robot is None:
                        stage.terms[index] = copy.copy(term)
                        stage.terms[index]._active_cols_np = cols
        elif not np.isin(cols, self._collision_cols).all():
            raise ValueError("The first solve built the collision tasks for fewer joints; set the widest mask before.")
        self._solve_graph = None
        self._active_joint_mask_np = joint_mask_np.copy()
        self._active_joint_mask_wp.assign(joint_mask_np)
        self.ik.set_active_joint_mask(joint_mask_np.tolist())
        if self.config.stages[0].method == "mppi":
            cast(MppiTrajectoryOptimizer, self._trajectory_optimizer).set_active_joint_mask(joint_mask_np)

    # --- internals ---
    def _solve_waypoint_ik(
        self,
        waypoints: Sequence[wp.array],
        seed_q: Optional[wp.array],
        start_T_world_base: Optional[wp.array],
        scene_indices: wp.array,
        named: Sequence[int],
        spheres: Optional[np.ndarray] = None,
    ) -> Tuple[wp.array, list]:
        """Each waypoint's joints: a pose's best goal IK, started and resting at its neighbors' blend, else `seed_q`;
        the target links a pose leaves out (`named`: its links as bits, -1 all) keep their poses at that blend.

        Returns:
            The `(batch, waypoints, dofs)` joints, and each later pose's index, next joint waypoint, IK answers and valid count, from which every guess picks its own.
        """
        batch = waypoints[0].shape[0]
        joints: List[Optional[wp.array]] = [None if w.dtype == wp_vec7 else w for w in waypoints]
        picks = []
        for index, waypoint in enumerate(waypoints):
            if joints[index] is not None:
                continue
            # the nearest joints before and after, blended by their places in the list
            before = next((j for j in range(index - 1, -1, -1) if joints[j] is not None), None)
            after = next((j for j in range(index + 1, len(waypoints)) if joints[j] is not None), len(waypoints))
            reference = seed_q if before is None else joints[before]
            if after < len(waypoints) and (before is not None or seed_q is None):
                reference = joints[after]
                if before is not None:
                    reference = wp.empty_like(joints[before])
                    alpha = (index - before) / (after - before)
                    wp.launch(
                        _blend_kernel,
                        dim=reference.shape,
                        inputs=[joints[before], joints[after], alpha],
                        outputs=[reference],
                        device=self.device,
                    )
            rest = (
                None if reference is None else self.robot.state(q=wp.clone(reference), T_world_base=start_T_world_base)
            )
            target = waypoint.reshape((batch, len(self.target_link_indices)))
            if named[index] != -1 and rest is None:  # nothing to keep the links it leaves out at
                raise ValueError("A start pose naming some target links needs seed_q or a later joint waypoint.")
            if named[index] != -1:
                kept = (named[index] >> np.arange(len(self.target_link_indices)) & 1)[None, :, None] == 0
                T_world_link = self.robot.forward_kinematics(rest).T_world_link.numpy()[:, self.target_link_indices]
                target = wp.array(np.where(kept, T_world_link, target.numpy()), dtype=wp_vec7, device=self.device)
            if self._radii is not None and spheres is not None:  # the spheres both segments at the waypoint check
                held = spheres[index - 1] * spheres[min(index, len(spheres) - 1)]
                self._radii[0].assign(np.where(held != 0, self.robot.spec.collision_sphere_radii, -1.0e9))
            state = self.ik.solve(
                T_world_target=target,
                init_state=rest,
                rest_state=rest,
                scene_indices=scene_indices,
            )
            if index:  # every answer: each guess picks its own, blending with the pose goal past the list if no joints
                valid = (
                    cast(CompositeTask, self.ik._built[self._goal_score][0]).valid_count
                    if self._num_ik_goal_candidates_total > 1
                    else self._no_valid_count
                )
                answers = (wp.clone(state.q), wp.clone(state.T_world_base), wp.clone(valid))
                picks.append((index, after, *answers))
            state = state if self.ik.num_solutions == 1 else state.gather(self._ik_best_indices)
            joints[index] = wp.clone(state.q)  # the solver reuses its buffers
        anchors = wp.empty((batch, len(joints), self.robot.num_actuated_joints), dtype=wp.float32, device=self.device)
        for index, q in enumerate(joints):
            wp.copy(anchors[:, index], q)
        if self._radii is not None:
            wp.copy(*self._radii)
        return anchors, picks

    def _build_target_q_per_candidate(
        self,
        T_world_target: Optional[wp.array],
        target_q: Optional[wp.array],
        scene_indices: wp.array,
        start_state: RobotState,
        seed_q: Optional[wp.array] = None,
        rest_q: Optional[wp.array] = None,
    ) -> wp.array:
        """Build joint or pose IK goal candidates, storing floating-base goals in `_target_base_per_candidate_buf`."""
        num_candidates = self._trajectory_optimizer.num_candidates
        num_dofs = self.robot.num_actuated_joints
        batch_size = start_state.batch_size
        n_total = batch_size * num_candidates
        self._ik_goal_state = None
        if target_q is not None or T_world_target is None:  # repeat joint-space goals, or hold the start
            warp_utils.repeat(
                target_q if target_q is not None else start_state.q,
                num_candidates,
                out=self._target_q_per_candidate_buf,
            )
            if self._floating:
                warp_utils.repeat(start_state.T_world_base, num_candidates, out=self._target_base_per_candidate_buf)
        else:  # solve pose-goal IK: sampled everywhere, or around the seed, else the start for a floating base or pins
            base = start_state.T_world_base if self._floating else None
            init_state = None
            if seed_q is not None or self._floating or np.any(self._active_joint_mask_np == 0.0):
                init_q = wp.clone(start_state.q if seed_q is None else seed_q)
                init_state = self.robot.state(q=init_q, T_world_base=base)
                wp.launch(  # pinned joints stay where they start
                    _pin_inactive_joints_kernel,
                    dim=init_q.shape,
                    inputs=[start_state.q, self._active_joint_mask_wp, 1, init_q],
                    device=self.device,
                )
            follow = seed_q is not None and rest_q is seed_q  # online: rest at, and stay near, the last goal
            rest_state = init_state if rest_q is seed_q else None
            if rest_state is None and rest_q is not None:
                rest_state = self.robot.state(q=rest_q, T_world_base=base)
            target = cast(wp.array, T_world_target).reshape((batch_size, len(self.target_link_indices)))
            result = self.ik.solve(
                T_world_target=target,
                prev_state=init_state if follow else None,
                init_state=init_state,
                rest_state=None if self._floating else rest_state,  # a floating-base preset holds its own posture
                scene_indices=scene_indices,
            )
            self._ik_goal_state = result
            self._ik_goal_q = result.q  # ranked: feasible goals first, nearest the start configuration first
            self._ik_valid_count = (  # only valid IK goals seed trajectories while there are any
                cast(CompositeTask, self.ik._built[self._goal_score][0]).valid_count
                if self._num_ik_goal_candidates_total > 1
                else self._no_valid_count
            )
            wp.launch(
                _expand_ik_goal_candidates_kernel,
                dim=(n_total, num_dofs),
                inputs=[
                    self._ik_goal_q,
                    0,
                    self._num_ik_goal_candidates,
                    self._num_ik_goal_candidates_total,
                    num_candidates,
                    0,
                    self._ik_valid_count,
                    result.T_world_base,
                    self._target_q_per_candidate_buf,
                    self._target_base_per_candidate_buf,
                ],
                device=self.device,
            )

        return self._target_q_per_candidate_buf

    def _build_initial_trajectories(
        self,
        start_state: RobotState,
        target_q_per_candidate: wp.array,
        waypoint_layout: Optional[WaypointLayout] = None,
        round_index: int = 0,
    ) -> RobotState:
        """Each guess's waypoints, its own goal last, joined by straight lines; retry rounds redraw the detours.

        Args:
            start_state: Starts, and base poses for a floating base.
            target_q_per_candidate: Each guess's goal.
            waypoint_layout: The waypoints and how their segments behave; None: the starts to the goals.
            round_index: The retry round, which seeds the detours.
        """
        start_q = start_state.q
        if waypoint_layout is None:
            waypoint_layout = self._start_goal_layout._replace(
                q=start_q.reshape((start_q.shape[0], 1, start_q.shape[1]))
            )
        frames, locked = waypoint_layout.frames, waypoint_layout.locked
        num_candidates = self._trajectory_optimizer.num_candidates
        rows_q, rows_base = self._waypoint_bufs[frames.shape[0]]
        wp.launch(
            _pin_inactive_joints_kernel,
            dim=target_q_per_candidate.shape,
            inputs=[start_q, self._active_joint_mask_wp, num_candidates, target_q_per_candidate],
            device=self.device,
        )
        wp.launch(
            _waypoint_rows_kernel,
            dim=rows_q.shape,
            inputs=[
                waypoint_layout.q,
                target_q_per_candidate,
                start_state.T_world_base,
                self._target_base_per_candidate_buf,
                num_candidates,
            ],
            outputs=[rows_q, rows_base],
            device=self.device,
        )
        for index, after, pool_q, pool_base, valid in waypoint_layout.ik_picks:  # in order: each follows the last
            wp.launch(
                _pick_ik_kernel,
                dim=rows_q.shape[0],
                inputs=[pool_q, pool_base, valid, index, after],
                outputs=[rows_q, rows_base],
                device=self.device,
            )
        if round_index == 0 and waypoint_layout.resplit is not None:  # resplit by the time each line needs
            seed_inputs = [rows_q, frames, locked, 0, self._active_joint_mask_wp, self._joint_limits_wp, 0.0]
            wp.launch(
                _seed_kernel,
                dim=self._traj_init.q.shape,
                inputs=[*seed_inputs, num_candidates, num_candidates, 0],
                outputs=[self._traj_init.q],
                device=self.device,
            )
            fixed, new_frames = waypoint_layout.resplit
            self._resplit_frames(self._traj_init.q, frames, fixed, new_frames)
            wp.copy(frames, new_frames)
        wp.launch(
            _seed_kernel,
            dim=self._traj_init.q.shape,
            inputs=[
                rows_q,
                frames,
                locked,
                round_index,
                self._active_joint_mask_wp,
                self._joint_limits_wp,
                self.config.init_sample_range,
                num_candidates,
                1 if round_index == 0 else num_candidates // 2,
                int(round_index > 0),
            ],
            outputs=[self._traj_init.q],
            device=self.device,
        )
        if self._floating:
            wp.launch(
                _seed_base_kernel,
                dim=self._traj_init.T_world_base.shape,
                inputs=[rows_base, frames, waypoint_layout.known_base],
                outputs=[self._traj_init.T_world_base],
                device=self.device,
            )
        self._traj_init.invalidate()
        return self._traj_init

    def _obstacle_distance(
        self,
        state: RobotState,
        scene_indices: wp.array,
        dists: wp.array,
        scene_pp: wp.array,
        distance: wp.array,
        reset: bool = True,
    ):
        """Least obstacle distance per group, negative in collision; optionally accumulate into the existing minima."""
        state.invalidate()
        self.robot.forward_kinematics(state)
        if reset:
            distance.fill_(1.0e9)
        if not self._world_collision:
            return
        if not state.is_collision_spheres_computed:
            self.robot.transform_collision_spheres(state)
        warp_utils.repeat(scene_indices, scene_pp.shape[0] // scene_indices.shape[0], out=scene_pp)
        self._scene.query_sdf(
            state.collision_sphere_centers_world.reshape(dists.shape),
            scene_indices=scene_pp,
            distance_only=True,
            out_signed_dists=dists,
        )
        wp.launch(
            _min_distance_kernel,
            dim=dists.shape[0],
            inputs=[dists, self._eff_radii, dists.shape[0] // distance.shape[0], distance],
            device=self.device,
        )

    # --- optimization stages ---
    def _optimize(
        self,
        goal: PlanGoal,
        traj_init: RobotState,
        dt: float,
        waypoint_layout: Optional[WaypointLayout] = None,
        round_index: int = 0,
    ) -> Tuple[RobotState, wp.array]:
        """Run the optimizer stages in order on the candidate population, giving each query's best plan and score; a waypoint layout holds its waypoints and checks collision on its first split; with a resplit, the stages after the first plan on frames spread by the time each part of the first stage's paths needs."""
        if waypoint_layout is not None:  # every round starts on the first split
            self._apply_split(self._frame_masks, waypoint_layout.frames, waypoint_layout)
        if self.config.stages[0].method == "mppi":
            return cast(MppiTrajectoryOptimizer, self._trajectory_optimizer).solve(
                start_q=goal.start_q,
                T_world_target=goal.T_world_target,
                target_q=goal.target_q,
                traj_q_init=traj_init.q,
                target_q_per_candidate=goal.target_q_per_candidate,
                scene_indices=goal.scene_indices,
                start_T_world_base=goal.start_T_world_base,
            )
        var = self._population
        population = cast(RobotState, var.get(VarKey.ROBOT))
        wp.copy(population.q, traj_init.q)
        if self._floating:
            wp.copy(population.T_world_base, traj_init.T_world_base)
        population.invalidate()
        self._timing.fill(dt)  # every round starts at its step
        state, scores = self._optimizers[0].solve(self._last_population if self._optimizers[0].segments else var, goal)
        for index in range(1, len(self._optimizers)):
            previous, stage = self._optimizers[index - 1], self._optimizers[index]
            # resplit by the time the first stage's best paths need, and resample every candidate onto the new frames; retry rounds reuse the first round's split, so every plan shares one
            resplit = waypoint_layout.resplit if waypoint_layout is not None else None
            if index == 1 and resplit is not None and self.config.stages[1].method != "mppi":
                (fixed, new_frames), (resampled_q, _, _, at, resampled_base) = resplit, self._resplit_bufs
                if round_index == 0:
                    self._resplit_frames(previous.best[0].q, waypoint_layout.frames, fixed, new_frames)
                population = cast(RobotState, var.get(VarKey.ROBOT))
                wp.launch(
                    _resample_kernel,
                    dim=population.q.shape,
                    inputs=[population.q, at],
                    outputs=[resampled_q],
                    device=self.device,
                )
                wp.copy(population.q, resampled_q)
                if self._floating:
                    wp.launch(
                        _resample_base_kernel,
                        dim=resampled_base.shape,
                        inputs=[population.T_world_base, at],
                        outputs=[resampled_base],
                        device=self.device,
                    )
                    wp.copy(population.T_world_base, resampled_base)
                population.invalidate()
                optimizers = [*self._optimizers[1:], *self._retry_optimizers.values()]
                self._apply_split(optimizers, new_frames, waypoint_layout)
            if self.config.stage_dt_scale > 0.0:  # the previous stage's retime of its fastest feasible candidate
                self._retimers[index - 1].fastest_feasible(
                    cast(RobotState, var.get(VarKey.ROBOT)).q,
                    previous.candidate_scores,
                    stage.num_candidates,
                    self._fastest_step,
                )
                minimum_dt = self.config.stage_retime.minimum_dt
                self._timing.scale(self._fastest_step, self.config.stage_dt_scale, minimum_dt)
            previous_scores = previous.best[1]
            state, scores = stage.solve(self._last_population if stage.segments else var, goal)
            if index in self._retry_optimizers:  # retries only follow a stage that planned some query
                self._retry_optimizers[index].prepare(self._last_population, goal)  # a branch cannot set a solver up
                self._stage_condition.zero_()
                wp.launch(
                    _score_condition_kernel,
                    dim=previous_scores.shape[0],
                    inputs=[previous_scores, 1],
                    outputs=[self._stage_condition],
                    device=self.device,
                )
                wp.capture_if(self._stage_condition, self._retry_stage, index=index, goal=goal)
        return state, scores

    def _resplit_frames(self, path: wp.array, frames: wp.array, fixed: wp.array, new_frames: wp.array):
        """Waypoint frames spread by the time each part of `path` needs under the limits, and each new frame's place."""
        held, need, elapsed, at, _ = self._resplit_bufs
        wp.launch(_hold_kernel, dim=path.shape, inputs=[path, frames, fixed], outputs=[held], device=self.device)
        need.zero_()
        wp.launch(
            _frame_need_kernel,
            dim=path.shape,
            inputs=[held, self._velocity_limits_wp, self.config.acceleration_limit, self.config.jerk_limit],
            outputs=[need],
            device=self.device,
        )
        wp.launch(
            _resplit_kernel, dim=1, inputs=[need, frames, fixed, elapsed], outputs=[new_frames, at], device=self.device
        )

    def _apply_split(self, optimizers: Iterable, frames: wp.array, waypoint_layout: WaypointLayout):
        """Hold a split's waypoints on `frames`, and check collision where its segments ask, in each optimizer."""
        locked, check, lock_goal = waypoint_layout.locked, waypoint_layout.check_collision, waypoint_layout.lock_goal
        for optimizer in optimizers:  # attached objects and relative poses follow their segments too
            wp.launch(
                _frame_flags_kernel,
                dim=self._num_frames,
                inputs=[frames, locked, check, lock_goal, *cast(tuple, waypoint_layout.segment_items)],
                outputs=[optimizer.locked_frames, self._checked_frames, self._sphere_active, self._reference_frames]
                + [getattr(optimizer, "frame_segment", None)],  # a gradient stage's
                device=self.device,
            )
            for task, base in self._frame_masks[optimizer]:
                wp.launch(
                    _collision_mask_kernel,
                    dim=base.shape,
                    inputs=[base, self._checked_frames, self._sphere_active, task._sphere_indices_wp],
                    outputs=[task.frame_sphere_mask],
                    device=self.device,
                )

    def _retry_stage(self, index: int, goal: PlanGoal, attempt: int = 0):
        """Retry failed stage `index` with larger steps; nested branches skip remaining retries on success."""
        stage, retry = self._optimizers[index], self._retry_optimizers[index]
        if attempt > 0:
            scale = self.config.stage_dt_scale * self.config.stage_retry_dt_growth**attempt
            self._timing.scale(self._fastest_step, scale, self.config.stage_retime.minimum_dt)
            retry.solve(self._last_population, goal)
            (state, scores), (retry_state, retry_scores) = stage.best, retry.best
            self._retry_condition_stage.zero_()
            wp.launch(  # each query keeps its better plan; the condition flags any still infeasible
                _set_best_retry_result_kernel,
                dim=scores.shape[0],
                inputs=[
                    retry_scores,
                    retry_state.q,
                    retry_state.T_world_base,
                    self._start_distance,
                    scores,
                    self._retry_condition_stage,
                    retry.best_steps if self._segments else None,
                ],
                outputs=[state.q, state.T_world_base, stage.best_steps if self._segments else None],
                device=self.device,
            )
        else:
            self._retry_condition_stage.zero_()
            wp.launch(
                _score_condition_kernel,
                dim=stage.best[1].shape[0],
                inputs=[stage.best[1], 0],
                outputs=[self._retry_condition_stage],
                device=self.device,
            )
        if attempt < self.config.stage_retries:
            wp.capture_if(self._retry_condition_stage, self._retry_stage, index=index, goal=goal, attempt=attempt + 1)

    # --- offline planning ---
    def _solve_offline_attempt(
        self,
        round_index: int,
        start_state: RobotState,
        T_world_target: Optional[wp.array],
        target_q: Optional[wp.array],
        scene_indices: wp.array,
        target_q_per_candidate: wp.array,
        waypoint_layout: WaypointLayout,
        retries: int,
    ):
        """One optimization round: retries reseed pose goals from the IK funnel and redraw every guess's detours."""
        start_q = start_state.q
        batch_size = start_q.shape[0]
        num_candidates = self._trajectory_optimizer.num_candidates
        num_dofs = self.robot.num_actuated_joints
        if round_index > 0 and T_world_target is not None:  # reseed: this round's slice of the refined IK pool
            wp.launch(
                _expand_ik_goal_candidates_kernel,
                dim=(batch_size * num_candidates, num_dofs),
                inputs=[
                    self._ik_goal_q,
                    round_index * self._num_ik_goal_candidates,
                    self._num_ik_goal_candidates,
                    self._num_ik_goal_candidates_total,
                    num_candidates,
                    1,
                    self._ik_valid_count,
                    self._ik_goal_state.T_world_base,
                    target_q_per_candidate,
                    self._target_base_per_candidate_buf,
                ],
                device=self.device,
            )
        self._build_initial_trajectories(start_state, target_q_per_candidate, waypoint_layout, round_index)
        if self._seed_bufs is not None and retries:  # detour candidates: many paths of growing detour through the range
            per = self.config.init_detour_samples[0 if round_index == 0 else 1]
            seeds_q, seeds_state, sphere_dists, scene_pp, seed_distance = self._seed_bufs[per]
            wp.launch(
                _seed_kernel,
                dim=seeds_q.shape,
                inputs=[
                    self._waypoint_bufs[waypoint_layout.frames.shape[0]][0],
                    waypoint_layout.frames,
                    waypoint_layout.locked,
                    round_index,
                    self._active_joint_mask_wp,
                    self._joint_limits_wp,
                    1.0,
                    num_candidates,
                    0,
                    1,
                ],
                outputs=[seeds_q],
                device=self.device,
            )
            seed_distance.fill_(1.0e9)
            for frame in range(self._num_frames):  # a guess's detours share its base path
                wp.copy(seeds_state.q, seeds_q[:, frame, :])
                if self._floating:
                    warp_utils.repeat(self._traj_init.T_world_base[:, frame], per, out=seeds_state.T_world_base)
                self._obstacle_distance(seeds_state, scene_indices, sphere_dists, scene_pp, seed_distance, reset=False)
            lanes = 32 if self.device.is_cuda else 1
            wp.launch(
                _set_detour_seed_kernel,
                dim=batch_size * num_candidates * lanes,
                inputs=[
                    seeds_q,
                    seed_distance,
                    per,
                    num_candidates,
                    num_candidates // 2,
                    target_q_per_candidate,
                    lanes,
                    self._traj_init.q,
                ],
                device=self.device,
            )
        goal = PlanGoal(
            start_q,
            T_world_target,
            target_q if round_index == 0 else None,
            target_q_per_candidate,
            scene_indices,
            active_joint_mask=self._active_joint_mask_np,
            start_T_world_base=start_state.T_world_base if self._floating else None,
        )
        state, scores = self._optimize(goal, self._traj_init, self.config.dt, waypoint_layout, round_index)
        if retries == 0:
            if self._segments:
                wp.copy(self._final_log_step, self._optimizers[-1].best_steps)
            wp.copy(self._final_state.q, state.q)
            if self._floating:
                wp.copy(self._final_state.T_world_base, state.T_world_base)
            return
        self._retry_condition.fill_(0)
        wp.launch(
            _set_best_retry_result_kernel,
            dim=batch_size,
            inputs=[
                scores,
                state.q,
                state.T_world_base,
                self._start_distance,
                self._retry_best_score,
                self._retry_condition,
                self._optimizers[-1].best_steps if self._segments else None,
                self._final_state.q,
                self._final_state.T_world_base,
                self._final_log_step if self._segments else None,
            ],
            device=self.device,
        )
        if round_index < retries:  # the next round nests in this one's branch: a plan found skips all the rest at once
            wp.capture_if(
                self._retry_condition,
                self._solve_offline_attempt,
                round_index=round_index + 1,
                start_state=start_state,
                T_world_target=T_world_target,
                target_q=target_q,
                scene_indices=scene_indices,
                target_q_per_candidate=target_q_per_candidate,
                waypoint_layout=waypoint_layout,
                retries=retries,
            )

    def solve_offline(
        self,
        waypoints: Sequence[Union[wp.array, Dict[str, wp.array]]],
        *,
        start_T_world_base: Optional[wp.array] = None,
        seed_q: Optional[wp.array] = None,
        scene_indices: Optional[wp.array] = None,
        segment_check_collision: Optional[Sequence[bool]] = None,
        segment_fixed_duration: Optional[Sequence[Optional[float]]] = None,
        segment_locked: Optional[Sequence[bool]] = None,
        attached_objects: Optional[Sequence[AttachedObject]] = None,
        fixed_relative_poses: Optional[Sequence[FixedRelativePose]] = None,
    ) -> MotionPlanResult[wp.array]:
        """Plan batched Warp trajectories through waypoints, start first and goal last; the NumPy and Torch APIs wrap it.

        Lifecycle:
            1. Parse segments: each segment's world collision flag, fixed duration and lock.
            2. Solve waypoint IK: each pose but a final goal by goal IK (scene collision, no self collision),
               resting at its neighbors' blend.
            3. Create split: each segment's frames by its duration, else evenly, then the locked frames and collision.
            4. Solve: `_solve_offline_attempt` with retries, one CUDA graph under `cuda_graph="solve"` but for durations; it builds each guess's goal by goal IK, its later poses' answers nearest its neighbors' blend, and straight lines through its waypoints, on frames resplit by the time each line needs; after the first stage, several segments resplit their frames again.
            5. Retime: every plan under the last stage's retime, its fixed segments held still, and its motion time.

        Args:
            waypoints: Ordered list of at least two: joint arrays `(batch, dofs)` in `robot.spec.actuated_joint_names`
                order, `wp_vec7` arrays `(batch,)` or `(batch, target links)` posing the target links, or dicts of
                target link names to `wp_vec7` `(batch,)` posing some of them (other keys are ignored). A link
                a dict leaves out is free at its frame, so each link reaches its own waypoints in turn: the first
                guess keeps it at its pose of the waypoint before (blended toward any later joint waypoint; a start
                pose's at `seed_q` or the next joint waypoint, one needed), and the stages move it from there only as
                their costs pull. Every link passes each waypoint at the same instant, and the goal names all. An
                inner waypoint holds the links it names at its joints or IK answer, contact included.
            start_T_world_base: A floating base's start poses, `wp_vec7` `(batch,)`; such plans take a start and a goal.
            seed_q: Optional IK seed: the goal IK's initialization, or the rest of a pose waypoint before any joint one.
            scene_indices: Optional scene index per batch row; defaults to zero.
            segment_check_collision: One flag per segment (waypoint `i` to `i + 1`); False drops world collision there
                (self collision stays), and from the checks of anchors only such segments touch, e.g. a contact.
                Defaults to all True. Every `segment_*` and `waypoint_*` list applies to all batch rows alike.
            segment_fixed_duration: Seconds per segment, or None to let the planner time it. A duration gives the
                segment `round(duration / dt)` frames (at least two) in the solve, keeps them through the resplit
                and is its time in `interpolate`; None segments share the rest of `num_timesteps`, which the
                durations must fit.
            segment_locked: One flag per segment; True keeps its frames on the straight joint-space line between its
                waypoints (the tool's path between them is not straight), whatever it passes through, and holds the
                links both its waypoints name.
            attached_objects: Objects the robot carries, on the spheres `max_attached_spheres` reserves on the target
                link each one's `link` is rigid with: checked by world and self collision over their segments (an
                inner waypoint: those both its segments carry), in the stages, their scores and waypoint and goal IK.
            fixed_relative_poses: Link pairs whose relative pose the gradient stages hold over their segments, at most
                `max_fixed_relative_poses`.

        Returns:
            `q_traj` `(batch, frames, dofs)`, `motion_time` `(batch,)` and a floating base's `T_world_base_traj`,
            borrowed buffers the next solve overwrites; `motion_time` counts fixed segments at their duration.
            `waypoint_indices` locates every fixed anchor, `segment_fixed_duration` repeats the fixed durations, and
            `segment_duration` holds each plan's segment times when optimized.
            The first solve (or `warmup`) fixes the batch, goal IK's included: fewer queries are padded with the
            last and only their rows return; more raise `ValueError("Batch size mismatch")`.
        """
        if len(waypoints) < 2:
            raise ValueError("solve_offline needs a start and a goal; use `planner.ik` for IK alone.")
        num_segments, waypoints = len(waypoints) - 1, list(waypoints)
        robot, device, dofs = self.robot, self.device, self.robot.num_actuated_joints
        names = [robot.link_names[i] for i in self.target_link_indices]
        named = []  # each waypoint's target links as bits, -1 all; a dict's poses become one per target link
        for index, w in enumerate(waypoints):
            keys = [k for k in w if k in names] if isinstance(w, dict) else names
            if not keys:
                raise ValueError(f"A waypoint dict must name target links among {names}.")
            named.append(-1 if len(keys) == len(names) else sum(1 << names.index(k) for k in keys))
            if isinstance(w, dict) and len(names) > 1:  # a link it leaves out takes a filler pose
                waypoints[index] = wp.empty((w[keys[0]].shape[0], len(names)), dtype=wp_vec7, device=device)
                for k, name in enumerate(names):
                    wp.copy(waypoints[index][:, k], w.get(name, w[keys[0]]))
            elif isinstance(w, dict):
                waypoints[index] = w[names[0]]
        if named[-1] != -1:
            raise ValueError("The goal must name every target link.")
        queries = waypoints[0].shape[0]
        batch = max(queries, self._batch_size or 0)
        if batch > queries:
            rows = wp.array(np.minimum(np.arange(batch), queries - 1), dtype=wp.int32, device=device)
            waypoints = [warp_utils.gather(w, rows) for w in waypoints]
            start_T_world_base, seed_q, scene_indices = (
                None if x is None else warp_utils.gather(x, rows) for x in (start_T_world_base, seed_q, scene_indices)
            )
        scene_indices = scene_indices if scene_indices is not None else wp.zeros(batch, dtype=wp.int32, device=device)
        self.warmup(batch)

        # --- parse segments ---
        check_collision = [True] * num_segments if segment_check_collision is None else list(segment_check_collision)
        durations = [None] * num_segments if segment_fixed_duration is None else list(segment_fixed_duration)
        locked = [False] * num_segments if segment_locked is None else list(segment_locked)
        fixed = np.array([d is not None for d in durations], dtype=bool)
        goal_pose = waypoints[-1].dtype == wp_vec7  # its IK runs per guess, in the solve

        # --- place attached objects and relative poses ---
        # each segment's checked spheres: the robot's, and the reserved ones that an object it carries takes, on the
        # target link its `link` is rigid with, in that link's frame
        objects, poses, spec = list(attached_objects or ()), list(fixed_relative_poses or ()), robot.spec
        slots, first, tools = self.config.max_attached_spheres, self._num_robot_spheres, self.target_link_indices
        if objects and not slots or len(poses) > self.config.max_fixed_relative_poses:
            raise ValueError(
                "attached_objects need max_attached_spheres; fixed_relative_poses max_fixed_relative_poses."
            )

        def segments_of(x: Union[AttachedObject, FixedRelativePose]) -> List[int]:
            return list(range(num_segments)) if x.segments is None else [k % num_segments for k in x.segments]

        def body_of(link: int) -> int:  # the joint that moves it
            joint = int(spec.link_parent_joint_indices[link])
            while joint >= 0 and spec.joint_types[joint] == JointType.FIXED:
                joint = int(spec.parent_joint_indices[joint])
            return joint

        spheres = np.ones((num_segments, len(spec.collision_sphere_radii)), np.int32)
        if slots:
            spheres[:, first:] = 0
            spec.local_collision_sphere_centers[first:], spec.collision_sphere_radii[first:] = 0.0, -1.0e9  # off
            ignored = np.zeros((spheres.shape[1], spec.num_links), bool)  # each sphere's ignored links
            T_world_link = pose7_to_tf_mat(
                robot.forward_kinematics_numpy(np.zeros((1, dofs), np.float32), device=str(device))[0]
            )
            used = [0] * len(tools)
            for obj in objects:
                holder = robot.link_names.index(obj.link)
                k = next((k for k, t in enumerate(tools) if body_of(t) == body_of(holder)), None)
                if k is None:
                    raise ValueError(f"An attached object's link {obj.link} must be rigid with a target link.")
                T_link_object = np.eye(4) if obj.T_link_object is None else np.asarray(obj.T_link_object)
                T_target_object = np.linalg.inv(T_world_link[tools[k]]) @ T_world_link[holder] @ T_link_object
                cover = _compute_covering_spheres(obj.shape, obj.size, T_target_object)
                rows = first + k * slots + used[k] + np.arange(len(cover))
                used[k] += len(cover)
                if used[k] > slots:
                    raise ValueError(f"{names[k]} carries {used[k]} spheres, over max_attached_spheres={slots}.")
                spec.local_collision_sphere_centers[rows], spec.collision_sphere_radii[rows] = cover[:, :3], cover[:, 3]
                spheres[np.ix_(segments_of(obj), rows)] = 1
                skip = [robot.link_names.index(x) for x in obj.collision_ignored_links]
                skip += [self._num_robot_links + tools.index(x) for x in skip if x in tools]  # and its spheres
                ignored[np.ix_(rows, skip)] = True
            # every copy of the spheres: the device's, goal IK's, the seeds' and the self-collision checks', and each
            # reserved link's bounding sphere
            tensors, radii = spec.get_tensors(str(device)), spec.collision_sphere_radii.astype(np.float32)
            tensors.local_collision_sphere_centers.assign(spec.local_collision_sphere_centers.astype(np.float32))
            for array in (tensors.collision_sphere_radii, cast(tuple, self._radii)[1]):  # and goal IK's copy
                array.assign(radii)
            self._eff_radii[first:].assign(radii[first:])
            centers, r = (
                spec.local_collision_sphere_centers[first:].reshape(-1, slots, 3),
                radii[first:].reshape(-1, slots),
            )
            on = r > -1.0e8
            mean = (centers * on[..., None]).sum(1) / np.maximum(on.sum(1), 1)[:, None]
            bound = np.where(on, np.linalg.norm(centers - mean[:, None], axis=-1) + r, -1.0e9).max(1)
            tensors.local_link_bounding_sphere_centers[self._num_robot_links :].assign(mean.astype(np.float32))
            tensors.link_bounding_sphere_radii[self._num_robot_links :].assign(bound.astype(np.float32))
            for task in self._self_collision_tasks:
                task._radii_np[first:] = radii[first:]
                if getattr(task, "_radii_wp", None) is not None:
                    task._radii_wp.assign(task._radii_np)
            link = spec.collision_spheres_link_indices
            for task in self._self_collision_tasks[:1]:  # all check the same pairs
                i, j = task._active_pairs_np.T
                self._pair_active[: len(i)].assign((~(ignored[i, link[j]] | ignored[j, link[i]])).astype(np.int32))
            self._evaluator = None  # built again with these spheres
        if self._relative is not None:
            indices = [(robot.link_names.index(p.link_a), robot.link_names.index(p.link_b)) for p in poses]
            indices += [(0, 0)] * (self._reference_frames.shape[1] - len(poses))
            self._relative.link_pairs.assign(np.array(indices, np.int32))
        references = [  # each segment's relative poses' reference waypoint + 1, 0 off
            [
                (k if p.at_waypoint is None else p.at_waypoint % len(waypoints)) + 1 if k in segments_of(p) else 0
                for p in poses
            ]
            + [0] * (self._reference_frames.shape[1] - len(poses))
            for k in range(num_segments)
        ]
        segment_items = (spheres, references, named)

        # --- solve waypoint IK ---
        # a pose goal is left to the solver's goal IK, one answer per guess
        waypoint_q, picks = self._solve_waypoint_ik(
            waypoints[: num_segments + 1 - goal_pose],
            seed_q,
            start_T_world_base,
            scene_indices,
            named,
            spheres,
        )

        # --- create split ---
        # split frames to segments by duration, else evenly; the solve resplits them by the time each needs
        steps = self.config.num_timesteps
        duration_frames = [None if d is None else max(2, round(d / self.config.dt)) for d in durations]
        spare_frames = steps - sum(c or 2 for c in duration_frames)
        if spare_frames < 0 or (spare_frames and all(fixed)):
            raise ValueError("segment_fixed_duration must fit num_timesteps frames, and fill them if all are fixed.")
        free = list(itertools.accumulate(int(c is None) for c in duration_frames))  # free segments so far
        min_frames = itertools.accumulate(c or 2 for c in duration_frames)  # frames so far before the spare frames
        frames = (0, *(b + round(spare_frames * k / (free[-1] or 1)) for b, k in zip(min_frames, free)))

        # --- solve ---
        if num_segments + 1 not in self._waypoint_bufs:  # each guess's waypoints
            rows = batch * self._trajectory_optimizer.num_candidates
            self._waypoint_bufs[num_segments + 1] = (
                wp.empty((rows, num_segments + 1, dofs), dtype=wp.float32, device=device),
                wp.empty((rows, num_segments + 1), dtype=wp_vec7, device=device),
            )
        known = [1, *(w.dtype == wp_vec7 for w in waypoints[1:-1]), 1]  # waypoints with a base pose
        # frames resplit by the time each part needs: the straight lines', then the first stage's paths'
        resplit = num_segments > 1 and self.config.resplit_waypoints and not fixed.all()
        free = np.r_[~fixed, np.zeros(self._num_frames - num_segments, bool)].astype(np.int32)  # the last stage's steps
        for optimizer in [self._optimizers[-1], *self._retry_optimizers.values()] if self._segments else ():
            optimizer.segment_free.assign(free)

        def solve(start_q, base, T_world_target, target_q, seed_q, scene_indices, waypoint_q, rest_q, *layout_inputs):
            start_state = robot.state(q=start_q, T_world_base=base)
            new_frames, frames_wp, locked_wp, known_wp, fixed_wp, check_wp, *items_wp = layout_inputs[:9]
            if self._radii is not None:  # goal IK checks the spheres the last segment checks
                held_radii = (self._radii[1], items_wp[0], num_segments - 1)
                wp.launch(_held_radii_kernel, self._radii[0].shape, held_radii, [self._radii[0]], device=self.device)
            targets = self._build_target_q_per_candidate(
                T_world_target, target_q, scene_indices, start_state, seed_q=seed_q, rest_q=rest_q
            )
            if self._radii is not None:
                wp.copy(*self._radii)
            answers = layout_inputs[9:]
            waypoint_layout = WaypointLayout(
                q=waypoint_q,
                frames=frames_wp,
                locked=locked_wp,
                known_base=known_wp,
                ik_picks=[(*p[:2], *answers[3 * k : 3 * k + 3]) for k, p in enumerate(picks)],
                check_collision=check_wp,
                lock_goal=int(not goal_pose or locked[-1]),
                resplit=(fixed_wp, new_frames) if resplit else None,
                segment_items=tuple(items_wp),
            )
            # retries draw new detours, and pose goals new goal IK candidates, over the full batch
            retries = self.config.max_retries if self._retry_enabled else 0
            if retries:
                self._retry_best_score.fill_(1.0e30)
                if self._start_bufs is not None:
                    distance_state, dists, scene_pp = self._start_bufs
                    wp.copy(distance_state.q, start_q)
                    if self._floating:
                        wp.copy(distance_state.T_world_base, base)
                    self._obstacle_distance(distance_state, scene_indices, dists, scene_pp, self._start_distance)
            self._solve_offline_attempt(
                0, start_state, T_world_target, target_q, scene_indices, targets, waypoint_layout, retries
            )
            self._final_state.invalidate()

        inputs = (
            wp.clone(waypoint_q[:, 0]),
            start_T_world_base,
            waypoints[-1] if goal_pose else None,
            None if goal_pose else wp.clone(waypoint_q[:, num_segments]),
            seed_q if goal_pose else None,
            scene_indices,
            waypoint_q,
            wp.clone(waypoint_q[:, num_segments - 1]),  # the goal IK's rest
            # the final frames (input 8), then the first split, its locks, base poses, durations, collision checks,
            # checked spheres, relative pairs and named links
            *(
                wp.array(x, dtype=wp.int32, device=device)
                for x in (frames, frames, locked, known, fixed, check_collision, *segment_items)
            ),
            *(answer for p in picks for answer in p[2:]),
        )
        graph = self.config.cuda_graph == "solve" and device.is_cuda and not fixed.any()
        if graph:
            key = (
                self._scene.graph_revision,
                frames,
                tuple(locked),
                tuple(p[:2] for p in picks),
                tuple(None if x is None else x.shape for x in inputs),
            )
            if self._solve_graph is None or self._solve_graph[0] != key:
                # the buffers the graph reads
                inputs = tuple(wp.clone(x) if x is not None else None for x in inputs)
                solve(*inputs)  # initialize optimization before graph capture
                with wp.ScopedCapture(device=device) as capture:
                    solve(*inputs)
                self._solve_graph = (key, inputs, capture.graph)
            else:
                for source, destination in zip(inputs, self._solve_graph[1]):
                    if source is not None:
                        wp.copy(destination, source)
            inputs = self._solve_graph[1]  # the buffers the graph writes
            wp.capture_launch(self._solve_graph[2])
        else:
            solve(*inputs)

        # --- retime ---
        # each plan's step under the last stage's retime, and its motion time, with the fixed segments held still
        new_frames, q = inputs[8], self._final_state.q
        if resplit:
            frames = tuple(int(f) for f in new_frames.numpy())
        if fixed.any():
            q = wp.empty_like(q)
            wp.launch(
                _hold_kernel,
                dim=q.shape,
                inputs=[self._final_state.q, new_frames, inputs[12]],
                outputs=[q],
                device=device,
            )
        self._retimers[-1].retime(q, self._final_timing)
        moving, offset = steps - sum(c or 0 for c in duration_frames), sum(d or 0.0 for d in durations)
        wp.launch(
            _motion_time_kernel,
            dim=batch,
            inputs=[self._final_timing.dt, moving, offset],
            outputs=[self._motion_time],
            device=device,
        )
        segment_duration = None
        if self._segments:  # each free segment's optimized time, stretched to the retime of it alone, the rest held
            free_segments, held_q = np.flatnonzero(~fixed), wp.empty_like(q)
            times = np.exp(self._final_log_step.numpy()[:, :num_segments]) * np.diff(frames)
            for segment in free_segments if self._retimers[-1].config is not None else ():  # one: the retime above
                if len(free_segments) > 1:
                    held = wp.array(np.arange(num_segments) != segment, dtype=wp.int32, device=device)
                    wp.launch(_hold_kernel, q.shape, [self._final_state.q, new_frames, held], [held_q], device=device)
                    self._retimers[-1].retime(held_q, self._final_timing)
                needed = self._final_timing.dt.numpy() * (frames[segment + 1] - frames[segment])
                times[:, segment] = np.maximum(times[:, segment], needed)
            times = np.where(fixed, [d or 0.0 for d in durations], times).astype(np.float32)
            segment_duration = wp.array(times, dtype=wp.float32, device=device)
            wp.copy(self._motion_time, wp.array(times.sum(1), dtype=wp.float32, device=device))

        return MotionPlanResult(  # pyright: ignore[reportReturnType]  # slices are arrays, not indexed arrays
            q_traj=self._final_state.q[:queries],
            T_world_base_traj=self._final_state.T_world_base[:queries] if self._floating else None,
            motion_time=self._motion_time[:queries],
            waypoint_indices=frames,
            segment_fixed_duration=tuple(durations) if fixed.any() else None,
            segment_duration=None if segment_duration is None else segment_duration[:queries],
        )

    def solve_offline_numpy(
        self,
        waypoints: Sequence[Union[np.ndarray, Dict[str, np.ndarray]]],
        *,
        start_T_world_base: Optional[np.ndarray] = None,
        seed_q: Optional[np.ndarray] = None,
        scene_indices: Optional[np.ndarray] = None,
        segment_check_collision: Optional[Sequence[bool]] = None,
        segment_fixed_duration: Optional[Sequence[Optional[float]]] = None,
        segment_locked: Optional[Sequence[bool]] = None,
        attached_objects: Optional[Sequence[AttachedObject]] = None,
        fixed_relative_poses: Optional[Sequence[FixedRelativePose]] = None,
    ) -> MotionPlanResult[np.ndarray]:
        """NumPy wrapper for `solve_offline`: a pose waypoint is a dict of target links to 4x4 or
        `[x, y, z, qw, qx, qy, qz]` poses, unbatched values broadcast across queries, and the result holds owned
        copies.
        """
        device = self.device

        def joints(q: np.ndarray) -> np.ndarray:
            return np.atleast_2d(np.asarray(q, dtype=np.float32))

        def pose7(T: np.ndarray) -> np.ndarray:  # a 4x4 matrix or a 7-vector, batched or not
            T = np.asarray(T, np.float32)
            T = np.atleast_2d(tf_mat_to_pose7(T) if T.shape[-2:] == (4, 4) else T)
            return np.concatenate([T[:, :3], normalize_quaternion(T[:, 3:])], axis=-1)

        values = [{k: pose7(v) for k, v in w.items()} if isinstance(w, dict) else joints(w) for w in waypoints]
        seed_q = joints(seed_q) if seed_q is not None else None
        base = None if start_T_world_base is None else np.atleast_2d(np.asarray(start_T_world_base, np.float32))
        scene_indices = np.atleast_1d(np.zeros(1, np.int32) if scene_indices is None else np.asarray(scene_indices))
        arrays = [a for w in values for a in (w.values() if isinstance(w, dict) else [w])]
        arrays += [a for a in (seed_q, base, scene_indices) if a is not None]
        batch = max(len(a) for a in arrays)

        def array(value: np.ndarray, dtype=wp.float32) -> wp.array:  # an unbatched value broadcasts across queries
            return wp.from_numpy(np.broadcast_to(value, (batch, *value.shape[1:])).copy(), dtype=dtype, device=device)

        result = self.solve_offline(
            [{k: array(v, wp_vec7) for k, v in w.items()} if isinstance(w, dict) else array(w) for w in values],
            start_T_world_base=None if base is None else array(base, wp_vec7),
            seed_q=None if seed_q is None else array(seed_q),
            scene_indices=array(scene_indices.astype(np.int32), wp.int32),
            segment_check_collision=segment_check_collision,
            segment_fixed_duration=segment_fixed_duration,
            segment_locked=segment_locked,
            attached_objects=attached_objects,
            fixed_relative_poses=fixed_relative_poses,
        )
        return MotionPlanResult(
            q_traj=cast(wp.array, result.q_traj).numpy().copy(),
            T_world_base_traj=None if result.T_world_base_traj is None else result.T_world_base_traj.numpy().copy(),
            motion_time=cast(wp.array, result.motion_time).numpy().copy(),
            waypoint_indices=result.waypoint_indices,
            segment_fixed_duration=result.segment_fixed_duration,
            segment_duration=None if result.segment_duration is None else result.segment_duration.numpy().copy(),
        )

    def solve_offline_torch(
        self,
        waypoints: Sequence[Union["torch.Tensor", Dict[str, "torch.Tensor"]]],
        *,
        start_T_world_base: Optional["torch.Tensor"] = None,
        seed_q: Optional["torch.Tensor"] = None,
        scene_indices: Optional["torch.Tensor"] = None,
        segment_check_collision: Optional[Sequence[bool]] = None,
        segment_fixed_duration: Optional[Sequence[Optional[float]]] = None,
        segment_locked: Optional[Sequence[bool]] = None,
        attached_objects: Optional[Sequence[AttachedObject]] = None,
        fixed_relative_poses: Optional[Sequence[FixedRelativePose]] = None,
    ) -> MotionPlanResult["torch.Tensor"]:
        """Torch wrapper for `solve_offline`, as `solve_offline_numpy` with 4x4 pose matrices; plans without waypoint
        IK stay on device.
        """
        batched = [(w, 2) for w in (*waypoints, seed_q) if w is not None and not isinstance(w, dict)]
        batched += [(T, 3) for w in waypoints if isinstance(w, dict) for T in w.values()]
        batched += [(x, ndim) for x, ndim in ((start_T_world_base, 3), (scene_indices, 1)) if x is not None]
        batch = max((x.shape[0] for x, ndim in batched if x.ndim == ndim), default=1)

        def array(value: "torch.Tensor", dtype, ndim: int) -> wp.array:
            value = value[None] if value.ndim == ndim - 1 else value
            value = value.expand(batch, *value.shape[1:])  # an unbatched value broadcasts across queries
            if dtype is wp.mat44:
                return se3_from_matrix(wp.from_torch(value.contiguous(), dtype=wp.mat44))
            return wp.from_torch(value.contiguous(), dtype=dtype)

        result = self.solve_offline(
            [
                {k: array(T, wp.mat44, 3) for k, T in w.items()} if isinstance(w, dict) else array(w, wp.float32, 2)
                for w in waypoints
            ],
            start_T_world_base=None if start_T_world_base is None else array(start_T_world_base, wp.mat44, 3),
            seed_q=None if seed_q is None else array(seed_q, wp.float32, 2),
            scene_indices=None if scene_indices is None else array(scene_indices.int(), wp.int32, 1),
            segment_check_collision=segment_check_collision,
            segment_fixed_duration=segment_fixed_duration,
            segment_locked=segment_locked,
            attached_objects=attached_objects,
            fixed_relative_poses=fixed_relative_poses,
        )
        return MotionPlanResult(
            q_traj=wp.to_torch(cast(wp.array, result.q_traj)).clone(),
            T_world_base_traj=None
            if result.T_world_base_traj is None
            else wp.to_torch(se3_to_matrix(cast(wp.array, result.T_world_base_traj))).clone(),
            motion_time=wp.to_torch(cast(wp.array, result.motion_time)).clone(),
            waypoint_indices=result.waypoint_indices,
            segment_fixed_duration=result.segment_fixed_duration,
            segment_duration=None if result.segment_duration is None else wp.to_torch(result.segment_duration).clone(),
        )

    # --- online planning ---
    def _solve_online_mppi(
        self,
        start_state: RobotState,
        T_world_target: Optional[wp.array],
        target_q: Optional[wp.array],
        scene_indices: wp.array,
        online_state: Optional[OnlineState],
    ) -> Tuple[RobotState, OnlineState]:
        """Solve one MPPI update with its retained trajectory or action distribution."""
        online_state = online_state or OnlineState()
        start_q = start_state.q
        target_q_per_candidate = self._build_target_q_per_candidate(
            T_world_target, target_q, scene_indices, start_state, rest_q=start_q
        )
        wp.launch(
            _pin_inactive_joints_kernel,
            dim=target_q_per_candidate.shape,
            inputs=[
                start_q,
                self._active_joint_mask_wp,
                self._trajectory_optimizer.num_candidates,
                target_q_per_candidate,
            ],
            device=self.device,
        )
        state, _ = cast(MppiTrajectoryOptimizer, self._trajectory_optimizer).solve(
            start_q=start_q,
            T_world_target=T_world_target,
            target_q=target_q,
            traj_q_init=None,
            target_q_per_candidate=target_q_per_candidate,
            scene_indices=scene_indices,
            online_state=online_state,
            start_T_world_base=start_state.T_world_base if self._floating else None,
        )
        return state, online_state

    def _solve_online_gradient(
        self,
        start_state: RobotState,
        T_world_target: Optional[wp.array],
        target_q: Optional[wp.array],
        scene_indices: wp.array,
        online_state: Optional[OnlineState],
    ) -> Tuple[RobotState, OnlineState]:
        """Solve one fresh gradient trajectory.

        Lifecycle:
            1. Build joint goals, stabilizing pose-goal IK with the previous joint goal.
            2. Sample fresh trajectories from the current configurations to those goals.
            3. Append the current configurations to the executed history.
            4. Optimize the trajectories and save the joint goal as the next IK initialization.
        """
        start_q = start_state.q
        # --- build goals and candidates ---
        seed_q = (
            online_state.prev_goal_q
            if target_q is None and T_world_target is not None and online_state is not None
            else None
        )
        rest_q = seed_q if seed_q is not None and self.ik.num_solutions == 1 else start_q
        target_q_per_candidate = self._build_target_q_per_candidate(
            T_world_target,
            target_q,
            scene_indices,
            start_state,
            seed_q=seed_q,
            rest_q=rest_q,
        )
        traj_init = self._build_initial_trajectories(start_state, target_q_per_candidate)

        # --- update online state ---
        if online_state is None:
            online_state = OnlineState()
        if online_state.prev_goal_q is None:
            online_state.prev_goal_q = wp.empty(
                (start_q.shape[0], self.robot.num_actuated_joints), dtype=wp.float32, device=self.device
            )
        wp.copy(
            online_state.prev_goal_q,
            target_q_per_candidate.reshape(
                (start_q.shape[0], self._trajectory_optimizer.num_candidates, self.robot.num_actuated_joints)
            )[:, 0],
        )
        if online_state.q_history is None:
            online_state.q_history = wp.zeros(
                (start_q.shape[0], 3, self.robot.num_actuated_joints), dtype=wp.float32, device=self.device
            )
            online_state.history_count = 0
        wp.launch(
            _append_q_history_kernel,
            dim=start_q.shape,
            inputs=[online_state.q_history, start_q],
            device=self.device,
        )
        online_state.history_count = min(online_state.history_count + 1, 3)

        # --- solve trajectory ---
        goal = PlanGoal(
            start_q,
            T_world_target,
            target_q,
            target_q_per_candidate,
            scene_indices,
            online_state.q_history,
            online_state.history_count,
            self._active_joint_mask_np,
            start_state.T_world_base if self._floating else None,
        )
        state, _ = self._optimize(goal, traj_init, self.config.dt)
        return state, online_state

    def solve_online(
        self,
        start_state: RobotState,
        *,
        T_world_target: Optional[wp.array] = None,
        target_q: Optional[wp.array] = None,
        online_state: Optional[OnlineState] = None,
        scene_indices: Optional[wp.array] = None,
    ) -> Tuple[RobotState, OnlineState]:
        """Solve one online planning update.

        Args:
            start_state: Current configurations with `q` shaped `(batch, dofs)`, and base poses for a floating base.
            T_world_target: Optional `wp_vec7` pose goals, `(batch,)` or `(batch, target links)`.
            target_q: Optional solved IK goals or joint-space goals with shape `(batch, dofs)`.
            online_state: Optional state returned by the previous update.
            scene_indices: Optional scene indices with shape `(batch,)`.

        Returns:
            The optimized trajectories, a borrowed buffer, and the caller-owned online state.
        """
        self.warmup(start_state.batch_size)
        scene_indices = scene_indices if scene_indices is not None else self._default_scene_indices
        solve = self._solve_online_mppi if self.config.stages[0].method == "mppi" else self._solve_online_gradient
        state, online_state = solve(start_state, T_world_target, target_q, scene_indices, online_state)
        wp.copy(self._final_state.q, state.q)
        if self._floating:
            wp.copy(self._final_state.T_world_base, state.T_world_base)
        self._final_state.invalidate()
        return self._final_state, online_state

    def solve_online_numpy(
        self,
        start_q: np.ndarray,
        *,
        start_T_world_base: Optional[np.ndarray] = None,
        T_world_target: Optional[np.ndarray] = None,
        target_q: Optional[np.ndarray] = None,
        online_state: Optional[OnlineState] = None,
        scene_indices: Optional[np.ndarray] = None,
    ) -> MotionPlanResult[np.ndarray]:
        """NumPy wrapper for `solve_online`."""

        def array(value: Optional[np.ndarray], dtype, ndim: int) -> Optional[wp.array]:
            if value is None:
                return None
            value = np.asarray(value, dtype=np.float32 if dtype is not wp.int32 else np.int32)
            return wp.from_numpy(value[None] if value.ndim == ndim - 1 else value, dtype=dtype, device=self.device)

        state, online_state = self.solve_online(
            self.robot.state(q=array(start_q, wp.float32, 2), T_world_base=array(start_T_world_base, wp_vec7, 2)),
            T_world_target=array(T_world_target, wp_vec7, 2 if len(self.target_link_indices) == 1 else 3),
            target_q=array(target_q, wp.float32, 2),
            online_state=online_state,
            scene_indices=array(scene_indices, wp.int32, 1),
        )
        return MotionPlanResult(
            q_traj=state.q.numpy().copy(),
            T_world_base_traj=state.T_world_base.numpy().copy() if self._floating else None,
            online_state=online_state,
        )

    def compute_metrics(
        self,
        state: RobotState,
        *,
        T_world_target: Optional[wp.array] = None,
        target_q: Optional[wp.array] = None,
        scene_indices: Optional[wp.array] = None,
        position_threshold_m: float = 0.005,
        orientation_threshold_rad: float = 0.05,
        q_threshold: float = 0.05,
        penetration_threshold_mm: float = 5.0,
        out: Optional[MotionPlanMetrics[wp.array]] = None,
    ) -> MotionPlanMetrics[wp.array]:
        """Endpoint accuracy of `state` against exactly one pose or joint goal, independent of term weights.

        Args:
            state: Trajectories with shape `(batch, frames, dofs)`; any planner's output works.
            T_world_target: Optional batched pose goals.
            target_q: Optional batched joint-space goals.
            position_threshold_m: Success threshold on translation error.
            orientation_threshold_rad: Success threshold on rotation error.
            q_threshold: Success threshold on joint error, for a joint goal.
            penetration_threshold_mm: Success threshold on scene penetration.
            out: Buffers to write in place, from an earlier call; allocated when omitted.

        Returns:
            Endpoint errors and success per batch element.
        """
        if self._evaluator is None:
            self._evaluator = MotionPlanEvaluator(
                self.robot,
                self.target_link_indices,
                self._num_frames,
                device=self.device,
                scene=self._scene,
            )
        return self._evaluator.compute_metrics(
            state,
            T_world_target=T_world_target,
            target_q=target_q,
            position_threshold_m=position_threshold_m,
            orientation_threshold_rad=orientation_threshold_rad,
            q_threshold=q_threshold,
            penetration_threshold_mm=penetration_threshold_mm,
            out=out,
            scene_indices=scene_indices,
        )

    def compute_metrics_numpy(
        self,
        q_traj: np.ndarray,
        *,
        T_world_base_traj: Optional[np.ndarray] = None,
        T_world_target: Optional[np.ndarray] = None,
        target_q: Optional[np.ndarray] = None,
        scene_indices: Optional[np.ndarray] = None,
        position_threshold_m: float = 0.005,
        orientation_threshold_rad: float = 0.05,
        q_threshold: float = 0.05,
        penetration_threshold_mm: float = 5.0,
    ) -> MotionPlanMetrics[np.ndarray]:
        """`compute_metrics` for numpy inputs, returning numpy arrays."""

        def array(value: Optional[np.ndarray], dtype) -> Optional[wp.array]:
            return None if value is None else wp.array(value, dtype=dtype, device=self.device)

        targets = None if T_world_target is None else np.reshape(T_world_target, (len(q_traj), -1, 7))
        metrics = self.compute_metrics(
            self.robot.state(q=array(q_traj, wp.float32), T_world_base=array(T_world_base_traj, wp_vec7)),
            T_world_target=array(targets, wp_vec7),
            target_q=array(None if target_q is None else np.atleast_2d(target_q), wp.float32),
            scene_indices=array(scene_indices, wp.int32),
            position_threshold_m=position_threshold_m,
            orientation_threshold_rad=orientation_threshold_rad,
            q_threshold=q_threshold,
            penetration_threshold_mm=penetration_threshold_mm,
        )
        return MotionPlanMetrics(**{k: None if v is None else v.numpy() for k, v in vars(metrics).items()})
