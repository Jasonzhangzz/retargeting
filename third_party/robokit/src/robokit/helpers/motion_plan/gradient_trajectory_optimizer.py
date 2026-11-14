# pyright: reportArgumentType=false, reportIndexIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportOperatorIssue=false
import copy
import dataclasses
import weakref
from dataclasses import field
from typing import TYPE_CHECKING, Any, List, Literal, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np
import warp as wp

from robokit.helpers.motion_plan.motion_plan_score_task import MotionPlanScoreTask
from robokit.helpers.motion_plan.plan_goal import MotionPlanContext, PlanGoal
from robokit.helpers.motion_plan.trajectory_retimer import RetimeConfig, TimingVar
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig, Stage
from robokit.opt.var_values import VarKey, VarValues, VectorVar
from robokit.robo.robot import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.composite_task import CompositeTask
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.sparse.trajectory_limit_task import TrajectoryLimitTask
from robokit.terms.sparse.trajectory_self_collision_task import TrajectorySelfCollisionTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.task import Context, ResidualTask
from robokit.terms.trajectory_task import TrajectoryTask
from robokit.utils import warp_utils
from robokit.utils.warp_utils import resolve_device, wp_device_type, wp_vec7


if TYPE_CHECKING:
    from robokit.geom import WarpScene


@wp.kernel
def _goal_distance_kernel(
    start_q: wp.array2d(dtype=wp.float32),
    goal_q: wp.array2d(dtype=wp.float32),
    goal_distance: wp.array1d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    distance_sq = float(0.0)
    for joint_idx in range(start_q.shape[1]):
        delta = goal_q[batch_idx, joint_idx] - start_q[batch_idx, joint_idx]
        distance_sq += delta * delta
    goal_distance[batch_idx] = wp.sqrt(distance_sq)


@wp.kernel
def _active_dof_mask_kernel(
    dof_mask: wp.array1d(dtype=wp.float32),
    locked: wp.array1d(dtype=wp.int32),
    config_locked: wp.array1d(dtype=wp.int32),
    dof_links: wp.array1d(dtype=wp.int32),  # each dof's target links as bits, which `locked` holds by bit
    segment_free: wp.array1d(dtype=wp.int32),  # each segment: 1 optimizes its step, whose column follows the frames'
    segments: int,
    out: wp.array1d(dtype=wp.float32),
):
    frame, dof = wp.tid()  # pyright: ignore[reportAssignmentType]
    held = (locked[frame] & dof_links[dof]) != 0 or config_locked[frame] != 0
    out[frame * dof_mask.shape[0] + dof] = wp.where(held, 0.0, dof_mask[dof])
    if dof == 0 and frame < segments:
        out[locked.shape[0] * dof_mask.shape[0] + frame] = float(segment_free[frame])


@wp.kernel
def _log_step_kernel(dt: wp.array1d(dtype=wp.float32), log_step: wp.array2d(dtype=wp.float32)):
    row, segment = wp.tid()  # pyright: ignore[reportAssignmentType]
    log_step[row, segment] = wp.log(dt[row // (log_step.shape[0] // dt.shape[0])])


def _dof_links(robot: Robot, target_link_indices: Sequence[int], base_dofs: int, device: str) -> wp.array:
    """Each tangent dof's target links as bits: those its joint moves; all for the base and joints moving none."""
    spec = robot.spec
    moves = spec.link_ancestor_joints_mask[target_link_indices] @ (spec.joints_to_actuated_mapping != 0)  # bool
    bits = (moves.astype(np.int64) << np.arange(len(target_link_indices))[:, None]).sum(0)
    links = np.r_[np.full(base_dofs, -1), np.where(bits > 0, bits, -1)].astype(np.int32)
    return wp.array(links, dtype=wp.int32, device=device)


# --- task construction -----------------------------------------------------
def _build_terms(
    templates: Sequence[ResidualTask], var: VarValues, context: Mapping[Context, Any]
) -> List[ResidualTask]:
    """Resolve task templates against the population `var` and a stage's context.

    - Templates with all-zero weights are skipped; a task the caller built keeps its identity and live targets.
    - Position limits and central-difference derivative bounds at the per-query step fuse into one launch.
    """
    tasks, kept = [], []
    for template in templates:
        if template.robot is not None:
            tasks.append(template)
            kept.append(template)
            continue
        if template.active:
            task = copy.copy(template)
            task.resolve(var, context)
            if task.active:
                tasks.append(task)
    return TrajectoryLimitTask.fuse(tasks, context.get(MotionPlanContext.QUERY_STEP), kept)


# --- configuration ---------------------------------------------------------
def _default_gradienttrajectoryoptimizer_terms() -> List[ResidualTask]:
    ctx = MotionPlanContext
    derivative_limits = (ctx.VELOCITY_LIMITS, ctx.ACCELERATION_LIMIT, ctx.JERK_LIMIT)
    return cast(
        List[ResidualTask],
        [
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
                frame_indices=ctx.GOAL_FRAMES,
                name="target",
            ),
            TrajectoryTask(ConfigurationTask(rest_q=ctx.START_Q, weight=100.0), frame_indices=[0], name="start"),
            TrajectorySmoothnessTask(weight=0.0, smooth_boundary=False, name="smoothness"),
            TrajectorySmoothnessTask(
                weight=0.0,
                order=3,
                dt=ctx.STEP,
                history=ctx.HISTORY,
                name="jerk_smoothness",
            ),
            TrajectorySmoothnessTask(
                weight=0.0,
                order=2,
                dt=ctx.STEP,
                history=ctx.HISTORY,
                name="acceleration_smoothness",
            ),
            TrajectoryLimitTask(
                order=1, upper=ctx.VELOCITY_LIMITS, weight=1.0, dt=ctx.STEP, stencil="forward", name="velocity_limit"
            ),
            TrajectoryTask(PositionLimit(weight=100.0), name="position_limit"),
            TrajectoryLimitTask(
                order=(1, 2, 3),
                upper=derivative_limits,
                weight=(0.0, 0.0, 0.0),
                dt=ctx.QUERY_STEP,
                name="derivative_bounds",
            ),
            TrajectoryLimitTask(
                order=(2, 3),
                lower=0.0,
                upper=0.0,
                weight=(0.0, 0.0),
                dt=ctx.QUERY_STEP,
                dt_power=(1.0, 1.5),
                name="derivative_smoothness",
            ),
            TrajectoryLimitTask(
                order=(1, 2, 3),
                upper=derivative_limits,
                weight=0.0,
                dt=ctx.QUERY_STEP,
                normalize=True,
                name="normalized_bounds",
            ),
            TrajectoryTask(ConfigurationTask(rest_q="midrange", weight=0.01), name="rest"),
            TrajectorySelfCollisionTask(weight=10.0, name="self_collision"),
            TrajectoryTask(ConfigurationTask(rest_q=ctx.GOAL_Q, weight=0.0), frame_indices=[-1], name="goal"),
        ],
    )


@dataclasses.dataclass
class GradientTrajectoryOptimizerConfig:
    """Configuration for one gradient trajectory optimization stage."""

    terms: List[ResidualTask] = field(default_factory=_default_gradienttrajectoryoptimizer_terms)
    """Task templates the stage builds from its robot, inputs, horizon and timing."""
    score_terms: List[ResidualTask] = field(default_factory=lambda: [MotionPlanScoreTask()])
    """Rank the candidates when there are several; several tasks combine."""
    solver: PopulationSolverConfig = field(
        default_factory=lambda: PopulationSolverConfig(
            stages=[
                Stage(50, optimizer=LMOptimizerConfig(lm_lambda=0.5, lambda_min=1e-6, lambda_max=1e6, rho_min=1e-4))
            ]
        )
    )
    """The stages: iterations, candidate widths and update rules; the planner sets graph capture."""
    method: Literal["gradient"] = "gradient"
    """The stage's method, which the planner dispatches on."""

    clamp_to_limits: bool = False
    """Clamp every line-search trial to the joint position limits."""
    lock_frames: Tuple[int, ...] = ()
    """Frames held where they start, e.g. `(0,)` the start or `(0, -1)` both ends; a locked last frame pursues no pose goal. Smoothness still spans them."""

    def add(self, term: ResidualTask, *, name: Optional[str] = None) -> "GradientTrajectoryOptimizerConfig":
        """Append a task, optionally assigning its input role.

        >>> from robokit.terms.dense.configuration_task import ConfigurationTask
        >>> config = GradientTrajectoryOptimizerConfig(terms=[]).add(ConfigurationTask(weight=0.1), name="rest")
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
        >>> config = GradientTrajectoryOptimizerConfig(terms=[]).add(ConfigurationTask(weight=0.2), name="rest")
        >>> config.task("rest").weight
        0.2
        """
        for term in self.terms:
            if term.name == name:
                return term
        raise KeyError(f"No task named {name!r}.")


def build_default_gradient_trajectory_optimizer_config() -> GradientTrajectoryOptimizerConfig:
    """Build the default gradient trajectory optimizer configuration."""
    config = GradientTrajectoryOptimizerConfig()
    config.task("target").dense_task.position_weight = 400.0
    config.task("target").dense_task.orientation_weight = 200.0
    config.task("smoothness").dense_task.weight = 5.0
    config.task("collision").dense_task.weight = 10.0
    return config


# --- optimizer -------------------------------------------------------------
class GradientTrajectoryOptimizer:
    """One gradient optimization stage over a population of candidate trajectories.

    Lifecycle:
        1. `__init__` stores configuration and creates the active-DOF mask.
        2. `warmup(var)` ties the stage to the population `var` (its `robot` trajectories and their `timing`) and builds the tasks and the solver once; the first `solve` calls it when needed.
        3. `solve(var, goal)` optimizes the population in place and returns each query's best trajectory.
    """

    def __init__(
        self,
        config: GradientTrajectoryOptimizerConfig,
        robot: Robot,
        target_links: Union[str, int, Sequence[Union[str, int]]],
        num_frames: int,
        dt: float,
        *,
        enable_T_world_base: bool = False,
        scene: Optional["WarpScene"] = None,
        use_cuda_graph: bool = True,
        acceleration_limit: float = 15.0,
        jerk_limit: float = 500.0,
        sphere_mask: Optional[wp.array] = None,
        retime: Optional[RetimeConfig] = None,
        segments: int = 0,
        duration_weight: float = 0.0,
        device: Optional[wp_device_type] = None,
    ):
        device = str(resolve_device(device))
        self.robot = robot
        links = [target_links] if isinstance(target_links, (str, int)) else list(target_links)
        self.target_link_indices = [robot.link_names.index(x) if isinstance(x, str) else int(x) for x in links]
        self._base_dofs = 6 if enable_T_world_base else 0
        self.num_frames = num_frames
        self._scene = scene
        self._sphere_mask = sphere_mask  # each frame's checked spheres, for the score
        self.device = device
        self.num_dofs = robot.num_actuated_joints
        self.num_candidates = config.solver.num_candidates[0]
        self._use_cuda_graph = use_cuda_graph
        self._limits = (acceleration_limit, jerk_limit)  # the plan's, for the derivative bounds and the score
        self._retime = retime  # the planner's retime of this stage, where the score samples the plan
        self.config = config
        self._dt = dt
        self.batch_size = 0
        self._initialized = False
        width = self.num_dofs + self._base_dofs  # tangent per frame: the base twist, then the joints
        for frame in config.lock_frames:
            if not isinstance(frame, int) or not -num_frames <= frame < num_frames:
                raise ValueError(f"lock_frames index {frame!r} outside [-{num_frames}, {num_frames}).")
        locked = sorted({frame % num_frames for frame in config.lock_frames})
        # a locked end frame never moves: a pose cost there is a constant with a masked gradient
        self._goal_frames = [] if num_frames - 1 in locked else [-1]
        self.segments, self._duration_weight = segments, duration_weight  # VarKey.SEGMENT_STEP columns; >0 optimizes
        self.frame_segment = wp.zeros(num_frames, dtype=wp.int32, device=device)  # each frame interval's segment
        self.segment_free = wp.zeros(num_frames, dtype=wp.int32, device=device)  # each segment: 1 optimizes it
        self._active_dof_mask_wp = wp.empty(num_frames * width + segments, dtype=wp.float32, device=device)
        self._joint_mask = np.ones(self.num_dofs, dtype=np.float32)  # the goal's, last applied to the mask above
        self._dof_mask_wp = wp.ones(width, dtype=wp.float32, device=device)  # the base, then the joints
        config_locked = np.zeros(num_frames, np.int32)
        config_locked[locked] = 1
        self._config_locked_wp = wp.array(config_locked, dtype=wp.int32, device=device)
        self._dof_links = _dof_links(robot, self.target_link_indices, self._base_dofs, device)
        self.locked_frames = wp.zeros(
            num_frames, dtype=wp.int32, device=device
        )  # each frame's held target links as bits, -1 all; may change on device
        self.set_lock_frames(())
        self._solver: Optional[PopulationSolver] = None
        self._var: Optional[VarValues] = None

    def set_lock_frames(self, lock_frames: Sequence[int]):
        """Hold these frames, besides `config.lock_frames`, where they start; never while capturing."""
        locked = np.zeros(self.num_frames, np.int32)
        locked[[f % self.num_frames for f in lock_frames]] = -1
        self.locked_frames.assign(locked)
        self._apply_active_dof_mask()

    def _apply_active_dof_mask(self):
        wp.launch(
            _active_dof_mask_kernel,
            dim=(self.num_frames, self._dof_mask_wp.shape[0]),
            inputs=[self._dof_mask_wp, self.locked_frames, self._config_locked_wp, self._dof_links]
            + [self.segment_free, self.segments],
            outputs=[self._active_dof_mask_wp],
            device=self.device,
        )

    def make_var(self, q: wp.array, T_world_base: Optional[wp.array] = None) -> VarValues:
        """A population for this stage alone: trajectories `q` (and floating-base poses) at the stage's step."""
        return VarValues(
            {
                VarKey.ROBOT: self.robot.state(q=q, T_world_base=T_world_base),
                VarKey.TIMING: TimingVar.full(q.shape[0] // self.num_candidates, self._dt, q.device),
            }
        )

    @property
    def candidate_scores(self) -> wp.array:
        """Every candidate's score after the last solve; 1e6 and above is infeasible."""
        return cast(wp.array, cast(PopulationSolver, self._solver).candidate_scores)

    @property
    def best(self) -> Tuple[RobotState, wp.array]:
        """Each query's best trajectory and score after the last solve, in buffers the next solve reuses."""
        solver = cast(PopulationSolver, self._solver)
        return cast(RobotState, solver._best_var.get(VarKey.ROBOT)), solver.best_costs

    @property
    def best_steps(self) -> wp.array:
        """Each query's best segment log steps after the last solve."""
        return cast(VectorVar, cast(PopulationSolver, self._solver)._best_var.get(VarKey.SEGMENT_STEP)).x

    def _build_problem(self, var: VarValues):
        """Build batch-shaped tasks on the population `var`."""
        config = self.config
        self._var = var
        self._state = cast(RobotState, var.get(VarKey.ROBOT))
        timing = cast(TimingVar, var.get(VarKey.TIMING))
        actual_batch = var.batch_size
        self.batch_size = actual_batch // self.num_candidates
        num_dofs, device = self.num_dofs, self.device
        scene_indices = self._scene_indices_buf = wp.zeros(actual_batch, dtype=wp.int32, device=device)
        self._goal_vec7_buf = wp.zeros((actual_batch, len(self.target_link_indices)), dtype=wp_vec7, device=device)
        self._goal_distance = wp.zeros(actual_batch, dtype=wp.float32, device=device)
        self._sq_buf = wp.zeros((actual_batch, num_dofs), dtype=wp.float32, device=device)
        self._target_q_per_candidate_buf = wp.zeros((actual_batch, num_dofs), dtype=wp.float32, device=device)
        # a joint goal's pose: forward kinematics of each candidate's goal configuration, which pose-goal tasks read
        base = wp.empty(actual_batch, dtype=wp_vec7, device=device) if self._base_dofs else None
        self._goal_state = self.robot.state(q=self._target_q_per_candidate_buf, T_world_base=base)
        self.robot.forward_kinematics(self._goal_state)  # allocates now: a first joint goal may come in a retry branch
        rows = np.arange(actual_batch)[:, None] * self.robot.spec.num_links + np.asarray(self.target_link_indices)
        self._goal_link_rows = wp.array(rows.ravel().astype(np.int32), dtype=wp.int32, device=device)
        self._history = (
            wp.zeros((actual_batch, 3, num_dofs), dtype=wp.float32, device=device),
            wp.zeros(actual_batch, dtype=wp.float32, device=device),
        )
        ctx = MotionPlanContext
        self._context = context = {
            ctx.TARGET_LINKS: self.target_link_indices,
            ctx.TARGET: self._goal_vec7_buf,
            ctx.START_Q: self._sq_buf,
            ctx.GOAL_Q: self._target_q_per_candidate_buf,
            ctx.STEP: self._dt,
            ctx.QUERY_STEP: timing.dt,
            ctx.HISTORY: self._history,
            ctx.VELOCITY_LIMITS: self.robot.spec.actuated_joint_velocity_limits.astype(np.float32),
            ctx.ACCELERATION_LIMIT: self._limits[0],
            ctx.JERK_LIMIT: self._limits[1],
            ctx.SCENE: self._scene,
            ctx.SCENE_INDICES: scene_indices,
            ctx.GOAL_FRAMES: self._goal_frames,
            ctx.GOAL_DISTANCE: self._goal_distance,
            ctx.CANDIDATES: self.num_candidates,
            # weak: the solver holds the score, so a strong `self` would keep the scene alive until a GC pass
            ctx.CANDIDATE_COSTS: lambda self_ref=weakref.ref(self): (
                cast(PopulationSolver, self_ref()._solver).candidate_scores
            ),
            ctx.RETIME: self._retime,
            ctx.SPHERE_MASK: self._sphere_mask,
        }
        self._reads_goal_distance = ctx.GOAL_DISTANCE in set().union(*(t.needs() for t in config.terms if t.active))
        self._tasks = _build_terms(config.terms, var, context)
        timed = [t for t in self._tasks if isinstance(t, TrajectoryLimitTask) and t.dt is timing.dt]
        for index, task in enumerate(timed if self._duration_weight > 0.0 else ()):  # bounds at the optimized steps
            task.timing = (self.frame_segment, self.segments, self._duration_weight if index == 0 else 0.0)
        # every task that reports penetration makes a candidate infeasible through the score
        self._collision_tasks = [task for task in self._tasks if isinstance(task, TrajectoryCollisionTask)]
        context[ctx.COLLISION_TASKS] = self._collision_tasks
        if not self._tasks:
            raise ValueError("Trajectory optimizer requires at least one enabled task.")
        self._state.clamp_q_on_integrate = config.clamp_to_limits  # project every line-search trial

    def _set_inputs(self, goal: PlanGoal):
        """Copy the goal into the tasks' inputs."""
        joint_mask = goal.active_joint_mask if goal.active_joint_mask is not None else np.ones_like(self._joint_mask)
        if not np.array_equal(joint_mask, self._joint_mask):  # a changed mask is copied once, never while capturing
            self._joint_mask = np.asarray(joint_mask, dtype=np.float32).copy()
            self._dof_mask_wp.assign(np.r_[np.ones(self._base_dofs, dtype=np.float32), self._joint_mask])
        self._apply_active_dof_mask()  # the locked frames may have moved on the device
        # the tasks read these buffers, so refilling them updates every task
        target, goal_q = self._goal_vec7_buf, self._target_q_per_candidate_buf
        sq = warp_utils.repeat(goal.start_q, self.num_candidates, out=self._sq_buf)
        if goal.target_q is not None:
            warp_utils.repeat(goal.target_q, self.num_candidates, out=goal_q)
        else:
            wp.copy(goal_q, goal.target_q_per_candidate)
        if goal.T_world_target is not None:
            warp_utils.repeat(
                goal.T_world_target.reshape((goal.start_q.shape[0], target.shape[1])), self.num_candidates, out=target
            )
        else:  # a joint goal: its pose is the goal configuration's, at the start base
            if self._base_dofs:
                warp_utils.repeat(goal.start_T_world_base, self.num_candidates, out=self._goal_state.T_world_base)
            self._goal_state.invalidate()
            self.robot.forward_kinematics(self._goal_state)
            warp_utils.gather(
                self._goal_state.T_world_link.reshape((-1,)), self._goal_link_rows, out=target.reshape((-1,))
            )
        q_history, history_valid = self._history
        if goal.q_history is not None:
            warp_utils.repeat(goal.q_history, self.num_candidates, out=q_history)
        history_valid.fill_(1.0 if goal.history_count >= 3 else 0.0)
        warp_utils.repeat(goal.scene_indices, self.num_candidates, out=self._scene_indices_buf)  # the tasks' buffer
        if self._reads_goal_distance:
            wp.launch(
                _goal_distance_kernel,
                dim=goal_q.shape[0],
                inputs=[sq, goal_q],
                outputs=[self._goal_distance],
                device=self.device,
            )
        cast(VarValues, self._var).invalidate()

    def warmup(self, var: VarValues):
        """Build the batch-shaped tasks, buffers, and solver for the population `var`."""
        if self._initialized:
            if var is not self._var:
                raise ValueError("an optimizer stays tied to the population it was warmed up on")
            return
        self._build_problem(var)
        score = None
        if self.num_candidates > 1 and self.config.score_terms:
            scores = _build_terms(self.config.score_terms, var, self._context)
            score = scores[0] if len(scores) == 1 else CompositeTask(scores)
        self._solver = PopulationSolver(
            terms=[self._tasks],
            config=dataclasses.replace(self.config.solver, cuda_graph_mode="full" if self._use_cuda_graph else "none"),
            device=self.device,
            score_terms=[score] if score is not None else None,
            active_dof_mask=self._active_dof_mask_wp,
        )
        self._initialized = True

    def prepare(self, var: VarValues, goal: PlanGoal):
        """Set the solver up now, outside any graph capture; the population is left as it is."""
        self.warmup(var)
        solver = cast(PopulationSolver, self._solver)
        if not solver._initialized:
            self._set_inputs(goal)
            solver.setup(var)

    def solve(self, var: VarValues, goal: PlanGoal) -> Tuple[RobotState, wp.array]:
        """Optimize the population `var` in place towards `goal`.

        Args:
            var: Candidate trajectories (`robot`) and their step (`timing`); always the one warmed up on.
            goal: Starts, goals and scenes of the queries.

        Returns:
            Each query's best trajectory and its score.
        """
        self.warmup(var)
        self._set_inputs(goal)
        if self._duration_weight > 0.0:  # every segment starts at its query's step
            dt, steps = cast(TimingVar, var.get(VarKey.TIMING)).dt, cast(VectorVar, var.get(VarKey.SEGMENT_STEP)).x
            wp.launch(_log_step_kernel, steps.shape, [dt], [steps], device=self.device)
        cast(PopulationSolver, self._solver).solve(var)
        return self.best
