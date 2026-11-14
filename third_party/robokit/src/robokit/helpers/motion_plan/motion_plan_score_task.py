# pyright: reportArgumentType=false, reportIndexIssue=false, reportGeneralTypeIssues=false, reportCallIssue=false
from typing import Any, Callable, Mapping, Optional, cast

import numpy as np
import warp as wp

from robokit.helpers.motion_plan.plan_goal import MotionPlanContext
from robokit.helpers.motion_plan.trajectory_retimer import (
    TimingVar,
    TrajectoryRetimer,
    _compute_retimed_kinematics,
    _interpolate_trajectory_kernel,
)
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo.robot_state import RobotState
from robokit.terms.dense.frame_task import _goal_error_func
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.sparse.trajectory_self_collision_task import TrajectorySelfCollisionTask
from robokit.terms.task import Context, ResidualTask
from robokit.utils.warp_utils import warp_sum, wp_vec7


# --- kernel ----------------------------------------------------------------
@wp.kernel
def _path_sphere_mask_kernel(
    frame_mask: wp.array2d(dtype=wp.int32),  # each plan frame's checked spheres
    steps: int,  # path samples per frame interval
    scene_spheres: wp.array1d(dtype=wp.int32),
    path_mask: wp.array2d(dtype=wp.int32),
    scene_mask: wp.array3d(dtype=wp.bool),
):
    """Each path sample's checked spheres: those both plan frames about it check (the scene's: but at the start)."""
    p, k = wp.tid()  # pyright: ignore[reportAssignmentType]
    a = p // steps
    b = wp.min(a + wp.where(p % steps == 0, 0, 1), frame_mask.shape[0] - 1)
    path_mask[p, k] = frame_mask[a, k] * frame_mask[b, k]
    if k < scene_mask.shape[2]:
        sphere = scene_spheres[k]
        scene_mask[0, p, k] = p > 0 and frame_mask[a, sphere] * frame_mask[b, sphere] != 0


@wp.kernel
def _accumulate_row_max_kernel(values: wp.array2d(dtype=wp.float32), out: wp.array1d(dtype=wp.float32)):
    row, col = wp.tid()  # pyright: ignore[reportAssignmentType]
    if values[row, col] > 0.0:
        wp.atomic_max(out, row, values[row, col])


@wp.kernel
def _accumulate_limit_violation_kernel(
    q: wp.array3d(dtype=wp.float32),
    joint_limits: wp.array2d(dtype=wp.float32),
    tolerance: float,
    violation: wp.array1d(dtype=wp.float32),
):
    batch, frame, dof = wp.tid()  # pyright: ignore[reportAssignmentType]
    value = q[batch, frame, dof]
    wp.atomic_max(violation, batch, wp.max(joint_limits[dof, 0] - value, value - joint_limits[dof, 1]) - tolerance)  # pyright: ignore[reportOperatorIssue]


@wp.func
def _pose_error_func(
    T_world_link: wp.array3d(dtype=wp_vec7),
    T_world_target: wp.array2d(dtype=wp_vec7),
    target_link_indices: wp.array1d(dtype=wp.int32),
    row: int,
    last_frame: int,
    draws_per_goal: int,
) -> wp.vec2:
    """Max terminal position/rotation error over target links; all draws around a candidate share its goal."""
    error = wp.vec2(0.0, 0.0)
    for i in range(target_link_indices.shape[0]):
        e = _goal_error_func(
            T_world_link[row, last_frame, target_link_indices[i]], T_world_target[row / draws_per_goal, i]
        )
        error = wp.vec2(wp.max(error[0], e[0]), wp.max(error[1], e[1]))
    return error


@wp.kernel
def _motion_plan_score_residual_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),
    T_world_target: wp.array2d(dtype=wp_vec7),
    cost: wp.array1d(dtype=wp.float32),
    penetration: wp.array1d(dtype=wp.float32),
    violation: wp.array1d(dtype=wp.float32),
    target_link_indices: wp.array1d(dtype=wp.int32),
    last_frame: int,
    row_offset: int,
    num_candidates: int,
    keep_candidate_zero: int,
    draws_per_goal: int,
    q: wp.array3d(dtype=wp.float32),
    velocity_limits: wp.array1d(dtype=wp.float32),
    dt: float,
    acceleration_limit: float,
    jerk_limit: float,
    hold_frames: int,
    use_retimed: int,
    rotation_tolerance: float,
    lanes: int,  # threads per candidate: a CUDA warp's 32, or 1 on CPU
    residual: wp.array2d(dtype=wp.float32),
):
    tid = wp.tid()
    batch, lane = tid // lanes, tid % lanes  # pyright: ignore[reportOperatorIssue]
    error = _pose_error_func(T_world_link, T_world_target, target_link_indices, batch, last_frame, draws_per_goal)
    score = 1000.0 * error[0] + 500.0 * error[1] + 0.01 * cost[batch] + 1.0e5 * penetration[batch]  # pyright: ignore[reportOperatorIssue]
    if use_retimed != 0:
        # goal error, then the joint path length, the retimed step and the mean frame cost
        frames = q.shape[1]
        opt_dt = _compute_retimed_kinematics(
            q,
            batch,
            velocity_limits,
            frames,
            q.shape[2],
            dt,
            1.0,
            wp.min(0.02, dt),
            dt,
            acceleration_limit,
            jerk_limit,
            1,
            hold_frames,
            lane,
            lanes,
        )[0]
        path = float(0.0)
        for pair in range(lane, (frames - 1) * q.shape[2], lanes):
            frame, dof = 1 + pair // q.shape[2], pair % q.shape[2]
            path += wp.abs(q[batch, frame, dof] - q[batch, frame - 1, dof])  # pyright: ignore[reportOperatorIssue]
        path = warp_sum(path)
        score = error[0] + error[1] + 40.0 * path + 0.1 * opt_dt + 1.0e-4 * cost[batch] / float(frames)  # pyright: ignore[reportOperatorIssue]
    if error[0] >= 0.005 or error[1] > rotation_tolerance:  # goal unreached  # pyright: ignore[reportOperatorIssue]
        score += 1.0e6
    if penetration[batch] > 0.0 or violation[batch] > 0.0:  # colliding or beyond the joint-limit tolerance
        score += 1.0e6
    candidate_zero = batch - batch % num_candidates
    if keep_candidate_zero != 0 and batch != candidate_zero:  # a feasible candidate zero wins
        zero = _pose_error_func(
            T_world_link, T_world_target, target_link_indices, candidate_zero, last_frame, draws_per_goal
        )
        if (
            zero[0] < 0.005  # pyright: ignore[reportOperatorIssue]
            and zero[1] < 0.05  # pyright: ignore[reportOperatorIssue]
            and penetration[candidate_zero] <= 0.0
            and violation[candidate_zero] <= 0.0
        ):
            score += 1.0e6
    if lane == 0:
        residual[batch, row_offset] = wp.sqrt(2.0 * score)


# --- task ------------------------------------------------------------------
class MotionPlanScoreTask(ResidualTask):
    """Rank a stage's candidates for the population solver: feasible plans first, then the lowest cost."""

    def __init__(
        self,
        rotation_tolerance: float = 0.05,  # a plan whose goal rotation error exceeds it counts as unreached
        joint_limit_tolerance: Optional[
            float
        ] = None,  # set: a joint this far beyond its limits makes a plan infeasible
        interpolation_dt: float = 0.0,  # >0: also check the joint path sampled at this step between frames
        # a self-collision, with spheres inflated by the robot asset's link buffers, makes a plan infeasible
        self_collision: bool = False,
        rank_retimed_path: bool = False,  # rank by the retimed path length, holding the stage retime's still frames
        keep_candidate_zero: bool = True,  # a feasible candidate zero, the initial guess, wins its query
    ):
        self.rotation_tolerance = rotation_tolerance
        self.joint_limit_tolerance = joint_limit_tolerance
        self.interpolation_dt = interpolation_dt
        self.self_collision = self_collision
        self.rank_retimed_path = rank_retimed_path
        self.keep_candidate_zero = keep_candidate_zero

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Build from the population's robot and horizon, the stage's inputs, and the score-only path checks."""
        super().resolve(var_values, context)
        ctx = MotionPlanContext
        state = cast(RobotState, var_values.get(VarKey.ROBOT))
        robot, num_frames, device, step = state.robot, state.q.shape[1], state.q.device, context[ctx.STEP]
        self.robot = robot
        self.target_link_indices = list(context[ctx.TARGET_LINKS])
        self._target_link_indices = wp.array(self.target_link_indices, dtype=wp.int32, device=device)
        self.last_frame = num_frames - 1
        self.T_world_target = context[ctx.TARGET].reshape((-1, len(self.target_link_indices)))  # (goals, links)
        self.cost_input: Callable[[], wp.array] = context[ctx.CANDIDATE_COSTS]
        self.collision_tasks = tuple(context[ctx.COLLISION_TASKS])
        self.num_candidates = context[ctx.CANDIDATES]
        self.draws_per_goal = context.get(ctx.DRAWS, 1)  # the ranked population may hold this many draws per goal
        batch_size = self.T_world_target.shape[0] * self.draws_per_goal
        self._penetrations = wp.zeros(batch_size, dtype=wp.float32, device=device)
        self._violations = wp.zeros(batch_size, dtype=wp.float32, device=device)
        # joint-space samples between frames, scored like frames
        steps = round(step / self.interpolation_dt) if self.interpolation_dt > 0 else 0
        path_frames = (num_frames - 1) * steps + 1 if self.collision_tasks and steps > 1 else num_frames
        movable = np.flatnonzero(robot.spec.collision_spheres_link_indices > 0)  # base spheres never move
        after_start = np.ones((path_frames, len(movable)), dtype=bool)
        after_start[0] = False  # as the frame check: the given start is not the plan's to fix
        # the stages' per-frame checked spheres, if any, and each path sample's: `_path_sphere_mask_kernel`'s inputs
        frame_mask, self._sphere_masks = context.get(ctx.SPHERE_MASK), None
        if frame_mask is not None:
            path_mask = wp.ones((path_frames, len(robot.spec.collision_sphere_radii)), wp.int32, device)
            after_start = wp.array(after_start[None], wp.bool, device=device)
            steps = (path_frames - 1) // (num_frames - 1)
            spheres = wp.array(movable, wp.int32, device=device)
            self._sphere_masks = (frame_mask, steps, spheres, path_mask, after_start)
        self.interpolated_collision_tasks = tuple(
            [
                TrajectoryCollisionTask(
                    robot=robot,
                    scene=context[ctx.SCENE],
                    scene_indices=context[ctx.SCENE_INDICES],
                    num_frames=path_frames,
                    weight=1.0,
                    margin=1.0e-4,  # headroom for float differences from the judge's check at the same samples
                    sphere_indices=movable,
                    frame_sphere_mask=after_start,
                )
            ]
            if path_frames != num_frames
            else []
        )
        self.self_collision_task = (  # its penetration makes a trajectory infeasible, on the sampled path
            TrajectorySelfCollisionTask(
                robot=robot,
                num_frames=path_frames,
                weight=1.0,
                margin=0.0,
            )
            if self.self_collision
            else None
        )
        if self.self_collision_task is not None and self._sphere_masks is not None:
            self.self_collision_task.sphere_mask = self._sphere_masks[3]
        retime = context.get(ctx.RETIME)
        # (velocity limits, step, acceleration and jerk limits, hold frames): rank by the retimed plan
        self.retimed_timing = (
            (
                wp.from_numpy(context[ctx.VELOCITY_LIMITS], dtype=wp.float32, device=device),
                step,
                context[ctx.ACCELERATION_LIMIT],
                context[ctx.JERK_LIMIT],
                retime.virtual_hold_frames if retime is not None else 0,
            )
            if self.rank_retimed_path
            else None
        )
        # with a retime, the path is sampled at the step the stage's retime gives
        self.retimer = (
            TrajectoryRetimer(
                robot, num_frames, step, context[ctx.ACCELERATION_LIMIT], context[ctx.JERK_LIMIT], retime, str(device)
            )
            if retime is not None
            else None
        )
        self._timing = TimingVar.full(batch_size, 0.0, device)
        self._anchors = wp.array([0, num_frames - 1], dtype=wp.int32, device=device)  # one segment
        self._durations = wp.array([0.0], dtype=wp.float32, device=device)  # timed by its step
        self._interpolated_var: Optional[VarValues] = None
        self._interpolated_residuals = []
        if self.interpolated_collision_tasks:
            q = wp.zeros((batch_size, path_frames, robot.num_actuated_joints), dtype=wp.float32, device=device)
            self._interpolated_var = VarValues({VarKey.ROBOT: robot.state(q=q)})
            self._interpolated_residuals = [  # margin 0, plain penalty: each residual is a sphere's penetration
                wp.empty((batch_size, task.residual_dim), dtype=wp.float32, device=device)
                for task in self.interpolated_collision_tasks
            ]

    @property
    def residual_dim(self) -> int:
        return 1

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        *args: object,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        state: RobotState = var_values.get(VarKey.ROBOT)  # type: ignore[assignment]
        if not state.is_fk_computed:
            self.robot.forward_kinematics(state)
        if out_residual is None:
            out_residual = wp.empty((state.q.shape[0], 1), dtype=wp.float32, device=state.q.device)
            row_offset = 0
        self._penetrations.zero_()
        if self._sphere_masks is not None:
            dim = self._sphere_masks[3].shape
            wp.launch(_path_sphere_mask_kernel, dim=dim, inputs=self._sphere_masks, device=state.q.device)
        for task in self.collision_tasks:
            task._query_collision_distances(var_values)
            task.accumulate_max_penetration(self._penetrations)
        if self._interpolated_var is not None:
            interpolated: RobotState = self._interpolated_var.get(VarKey.ROBOT)  # type: ignore[assignment]
            if self.retimer is not None:
                self.retimer.retime(state.q, self._timing)
            wp.launch(
                _interpolate_trajectory_kernel,
                dim=interpolated.q.shape,
                inputs=[
                    state.q,
                    self._timing.dt,
                    self.interpolation_dt if self.retimer is not None else 0.0,
                    self._anchors,
                    self._durations,
                ],
                outputs=[interpolated.q],
                device=state.q.device,
            )
            interpolated.invalidate()
            for task, residual in zip(self.interpolated_collision_tasks, self._interpolated_residuals):
                task.compute_weighted_residual(self._interpolated_var, out_residual=residual)
                wp.launch(
                    _accumulate_row_max_kernel,
                    dim=residual.shape,
                    inputs=[residual],
                    outputs=[self._penetrations],
                    device=state.q.device,
                )
        self._violations.zero_()
        if self.joint_limit_tolerance is not None:  # infeasible: a joint beyond its bounds by more than the tolerance
            wp.launch(
                _accumulate_limit_violation_kernel,
                dim=state.q.shape,
                inputs=[state.q, state.spec_tensors.actuated_joint_limits, self.joint_limit_tolerance],
                outputs=[self._violations],
                device=state.q.device,
            )
        if self.self_collision_task is not None:  # infeasible: a self-collision, on the sampled path
            path = self._interpolated_var if self._interpolated_var is not None else var_values
            self.self_collision_task._accumulate_max_residual(path, self._violations)
        wp.launch(
            _motion_plan_score_residual_kernel,
            dim=state.q.shape[0] * (32 if state.q.device.is_cuda else 1),
            inputs=[
                state.T_world_link,
                self.T_world_target,
                self.cost_input(),
                self._penetrations,
                self._violations,
                self._target_link_indices,
                self.last_frame,
                row_offset,
                self.num_candidates,
                int(self.keep_candidate_zero),
                self.draws_per_goal,
                state.q,
                *(self.retimed_timing or (self._violations, 0.0, 0.0, 0.0, 0)),
                int(self.retimed_timing is not None),
                self.rotation_tolerance,
                32 if state.q.device.is_cuda else 1,
            ],
            outputs=[out_residual],
            block_dim=32 if state.q.device.is_cuda else 256,
            device=state.q.device,
        )
        return out_residual
