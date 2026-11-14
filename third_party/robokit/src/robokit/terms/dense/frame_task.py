# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false, reportGeneralTypeIssues=false
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np
import warp as wp

from robokit.lie.se3_kernels import (
    se3_adjoint_func,
    se3_compose_func,
    se3_inverse_func,
    se3_jlog_func,
    se3_log_map_func,
    se3_multiply_func,
)
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import Context, ResidualTask
from robokit.utils.warp_utils import pose_array, stack, wp_vec6, wp_vec7


@wp.func
def _goal_error_func(T_world_ee: wp_vec7, goal: wp_vec7) -> wp.vec2:
    error = se3_compose_func(se3_inverse_func(T_world_ee), goal)
    position = wp.sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2])  # pyright: ignore[reportOperatorIssue]
    angle = 2.0 * wp.asin(wp.min(wp.sqrt(error[4] * error[4] + error[5] * error[5] + error[6] * error[6]), 1.0))  # pyright: ignore[reportOperatorIssue]
    return wp.vec2(position, angle)


@wp.kernel
def _frame_distance_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    targets: wp.array2d(dtype=wp_vec7),
    frames: wp.array1d(dtype=wp.int32),
    weights: wp.array1d(dtype=wp.float32),
    row_offset: int,
    out: wp.array2d(dtype=wp.float32),
):
    i, frame = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    error = _goal_error_func(
        T_world_link[i, frames[frame]], targets[i // (T_world_link.shape[0] // targets.shape[0]), frame]
    )
    out[i, row_offset + 2 * frame] = weights[6 * frame] * error[0]
    out[i, row_offset + 2 * frame + 1] = weights[6 * frame + 3] * error[1]


# --- task -------------------------------------------------------------------
class FrameTask(RobotTask, ResidualTask):
    """Match world-frame link targets; "distance" scores translation/angle thresholds, while "log" uses log residuals.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.opt.var_values import VarKey, VarValues
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("ur10_description"))
        >>> target_link_index = robot.spec.link_names.index("ee_link")
        >>> state = robot.state(q=robot.spec.zero_q)
        >>> target_link_pose = robot.forward_kinematics(state).get_T_world_link(target_link_index)
        >>> frame_task = FrameTask(robot, target_link_index, target_link_pose, position_weight=1.0, orientation_weight=1.0)
        >>> random_q = wp.from_numpy(np.array([[-0.5, 0.3, -0.2, 0.4, -0.1, 0.25]], dtype=np.float32), dtype=wp.float32, device="cpu")
        >>> state = robot.state(q=random_q)
        >>> robot.forward_kinematics(state)
        >>> robot.compute_motion_subspace(state)
        >>> weighted_residual = frame_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))
        >>> weighted_jacobian = frame_task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state}))
        >>> assert np.allclose(weighted_residual.numpy()[0, :3], np.array([0.5839, -0.0795, -0.1784], dtype=np.float32), atol=1e-3)
        >>> assert np.allclose(weighted_jacobian.numpy()[0, :, 0], np.array([-1.1628, 0.1626, -0.167, -0.0765, 0.3698, 0.9521], dtype=np.float32), atol=1e-3)
    """

    def __init__(
        self,
        robot: Optional[Robot] = None,
        frame_index: Optional[Union[int, Sequence[int], Context]] = None,
        T_world_target: Optional[Union[wp.array, np.ndarray, Sequence[wp.array], Context]] = None,
        position_weight: Union[float, Sequence[float], np.ndarray, Mapping[str, float]] = 1.0,
        orientation_weight: Union[float, Sequence[float], np.ndarray, Mapping[str, float]] = 1.0,
        residual_mode: Literal["log", "distance"] = "log",
        trajectory_frames: int = 1,
        name: str = "",
    ):
        """`trajectory_frames` > 1: evaluate only the last pose of each trajectory, leaving other rows unchanged."""
        self.name = name
        self.trajectory_frames = trajectory_frames
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.residual_mode: Literal["log", "distance"] = residual_mode
        if residual_mode not in {"log", "distance"}:
            raise ValueError(f"Unsupported residual_mode: {residual_mode}")
        self.robot: Optional[Robot] = robot
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.frame_index_arg, self.T_world_target_arg = frame_index, T_world_target
        if (
            frame_index is None
            or T_world_target is None
            or any(isinstance(a, Context) for a in (frame_index, T_world_target))
        ):
            return  # config-time stub; `resolve` or `set_frames` completes it
        if isinstance(T_world_target, np.ndarray):
            T_world_target = pose_array(T_world_target)

        if isinstance(frame_index, int):
            frame_indices_list = [frame_index]
        else:
            frame_indices_list = list(frame_index)

        if isinstance(T_world_target, wp.array) and T_world_target.ndim > 1:
            if len(frame_indices_list) != T_world_target.shape[1]:
                raise ValueError(
                    f"Number of frame indices ({len(frame_indices_list)}) must match "
                    f"number of targets ({T_world_target.shape[1]})"
                )
            self.device = T_world_target.device
            self.T_world_target = wp.clone(T_world_target)
        else:
            # a bare [batch] array is one frame's target; anything else is a sequence of them
            targets_list = [T_world_target] if isinstance(T_world_target, wp.array) else list(T_world_target)
            if len(frame_indices_list) != len(targets_list):
                raise ValueError(
                    f"Number of frame indices ({len(frame_indices_list)}) must match "
                    f"number of targets ({len(targets_list)})"
                )
            self.device = targets_list[0].device
            self.T_world_target = stack(targets_list, axis=1)

        self.num_frames = len(frame_indices_list)
        self._frame_indices_np = np.array(frame_indices_list, dtype=np.int32)

        self.frame_indices = wp.from_numpy(self._frame_indices_np, dtype=wp.int32, device=self.device)
        if self.num_frames == 1:
            self.frame_index = frame_indices_list[0]
        else:
            self.frame_index = -1

        pos_w = (
            np.full(self.num_frames, position_weight, dtype=np.float32)
            if isinstance(position_weight, (int, float))
            else np.asarray(position_weight, dtype=np.float32)
        )
        ori_w = (
            np.full(self.num_frames, orientation_weight, dtype=np.float32)
            if isinstance(orientation_weight, (int, float))
            else np.asarray(orientation_weight, dtype=np.float32)
        )

        if pos_w.shape[0] != self.num_frames:
            raise ValueError(
                f"Number of position weights ({pos_w.shape[0]}) must match number of frames ({self.num_frames})"
            )
        if ori_w.shape[0] != self.num_frames:
            raise ValueError(
                f"Number of orientation weights ({ori_w.shape[0]}) must match number of frames ({self.num_frames})"
            )

        residual_weight = np.empty(self.num_frames * 6, dtype=np.float32)
        residual_weight.reshape(-1, 6)[:, :3] = np.repeat(pos_w, 3).reshape(-1, 3)
        residual_weight.reshape(-1, 6)[:, 3:] = np.repeat(ori_w, 3).reshape(-1, 3)
        self._residual_weight_np = residual_weight
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=self.device)

    def set_target(self, target: Union[wp.array, np.ndarray, Sequence[wp.array]]):
        """Update target poses in-place. Batch size must match the original."""
        if isinstance(target, np.ndarray):
            target = pose_array(target, device=self.device)
        if isinstance(target, wp.array) and target.ndim > 1:
            wp.copy(self.T_world_target, target)
            return
        targets = [target] if isinstance(target, wp.array) else list(target)
        stack(targets, axis=1, out=self.T_world_target)

    @property
    def active(self) -> bool:
        weights = (self.position_weight, self.orientation_weight)
        return any(np.any(list(w.values()) if isinstance(w, Mapping) else w) for w in weights)

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Complete a stub with the helper's links and target buffer (not copied); named weights pick those links."""
        super().resolve(var_values, context)
        if self.frame_index_arg is None or self.T_world_target_arg is None:
            raise ValueError(f"{type(self).__name__} needs frame_index and T_world_target, e.g. a helper's Context.")
        robot = cast(Robot, self.robot)
        index = cast(Union[int, Sequence[int]], self.frame_index_arg)
        links = [index] if isinstance(index, int) else list(index)
        if isinstance(self.position_weight, Mapping) or isinstance(self.orientation_weight, Mapping):
            self.position_weight, self.orientation_weight = self.weights_for([robot.link_names[i] for i in links])
        target = cast(wp.array, self.T_world_target_arg)
        self.set_frames(robot, links, target)
        self.T_world_target = target

    def weights_for(self, frame_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve scalar, ordered, or named template weights for the selected frames.

        >>> task = FrameTask(position_weight={"tool": 2.0}, orientation_weight=3.0)
        >>> task.weights_for(["tool"])[0].tolist()
        [2.0]
        """
        weights = []
        for value in (self.position_weight, self.orientation_weight):
            array = [value.get(name, 0.0) for name in frame_names] if isinstance(value, Mapping) else value
            weights.append(np.broadcast_to(np.asarray(array, dtype=np.float32), (len(frame_names),)).copy())
        return weights[0], weights[1]

    def set_frames(
        self,
        robot: Robot,
        frame_index: Union[int, Sequence[int]],
        T_world_target: Union[wp.array, np.ndarray, Sequence[wp.array]],
    ):
        """Give a config-time stub its robot, link indices, and targets, keeping the configured weights."""
        FrameTask.__init__(
            self,
            robot,
            frame_index,
            T_world_target,
            self.position_weight,
            self.orientation_weight,
            residual_mode=self.residual_mode,
            trajectory_frames=self.trajectory_frames,
            name=self.name,
        )

    @property
    def residual_dim(self) -> int:
        return (2 if self.residual_mode == "distance" else 6) * self.num_frames

    def compute_weighted_residual(
        self, var_values: VarValues, out_residual: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        var = var_values.get(self.var_key)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        kernel_device = out_residual.device if out_residual is not None else self.device
        batch_size = var.batch_size
        if out_residual is None:
            out_residual = wp.empty((batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if self.residual_mode == "distance":
            wp.launch(
                _frame_distance_kernel,
                dim=(batch_size, self.num_frames),
                inputs=[var.T_world_link, self.T_world_target, self.frame_indices, self.residual_weight, row_offset],
                outputs=[out_residual],
                device=kernel_device,
            )
        elif self.num_frames == 1:
            wp.launch(
                kernel=compute_frame_task_weighted_residual_kernel,
                dim=[batch_size // self.trajectory_frames],
                inputs=[
                    var.T_world_link,
                    self.frame_index,
                    self.T_world_target,
                    self.residual_weight,
                    row_offset,
                    self.trajectory_frames,
                ],
                outputs=[out_residual],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=compute_multi_frame_task_weighted_residual_kernel,
                dim=[batch_size],
                inputs=[
                    var.T_world_link,
                    self.frame_indices,
                    self.T_world_target,
                    self.num_frames,
                    self.residual_weight,
                    row_offset,
                ],
                outputs=[out_residual],
                device=kernel_device,
            )
        return out_residual

    def fill_terminal(
        self,
        var_values: VarValues,
        out_residual: wp.array,
        row_offset: int,
        out_jacobian: Optional[wp.array] = None,
        nnz_offset: int = 0,
    ):
        """Write terminal pose residuals and optional sparse Jacobian values in one launch.

        Layouts:
        - `var_values`: flattened poses, `trajectory_frames` per trajectory.
        - Jacobian: TrajectoryTask's single-frame, row-major values, one per actuated joint.
        """
        var = var_values.get(self.var_key)
        if var.has_floating_base or self.num_frames != 1:
            raise ValueError("fill_terminal takes one frame of a fixed-base robot")
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        if out_jacobian is not None and not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)
        spec_tensor = var.spec_tensors
        num_actuated = spec_tensor.joints_to_actuated_mapping.shape[1]
        wp.launch(
            kernel=_terminal_frame_kernel,
            dim=(var.batch_size // self.trajectory_frames, num_actuated),
            inputs=[
                var.S_world if out_jacobian is not None else None,
                var.T_world_link,
                self.T_world_target,
                self.frame_index,
                spec_tensor.link_ancestor_joints_mask,
                spec_tensor.joints_to_actuated_mapping,
                self.residual_weight,
                self.trajectory_frames,
                int(out_jacobian is not None),
                num_actuated,
                row_offset,
                nnz_offset,
            ],
            outputs=[out_residual, out_jacobian],
            device=out_residual.device,
        )

    def compute_weighted_jacobian_analytic(
        self,
        var_values: VarValues,
        out_jacobian: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        if self.residual_mode == "distance":
            raise ValueError("Frame distance residuals are for scoring, not analytic Jacobians")
        var = var_values.get(self.var_key)
        col_offset = var_values.tangent_offset(self.var_key)
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)
        spec_tensor = var.spec_tensors
        kernel_device = out_jacobian.device if out_jacobian is not None else self.device
        batch_size = var.batch_size
        if out_jacobian is None:
            total_dofs = var.tangent_dim
            out_jacobian = wp.zeros((batch_size, self.residual_dim, total_dofs), dtype=wp.float32, device=kernel_device)
            row_offset = 0
        num_actuated = spec_tensor.joints_to_actuated_mapping.shape[1]

        if self.num_frames == 1:
            wp.launch(
                kernel=compute_frame_task_weighted_jacobian_kernel,
                dim=(batch_size // self.trajectory_frames, num_actuated),
                inputs=[
                    var.S_world,
                    var.T_world_link,
                    var.T_world_base,
                    self.T_world_target,
                    self.frame_index,
                    spec_tensor.link_ancestor_joints_mask,
                    spec_tensor.joints_to_actuated_mapping,
                    var.has_floating_base,
                    self.residual_weight,
                    row_offset,
                    col_offset,
                    self.trajectory_frames,
                ],
                outputs=[out_jacobian],
                device=kernel_device,
            )
        else:
            wp.launch(
                kernel=compute_multi_frame_task_weighted_jacobian_kernel,
                dim=(batch_size, num_actuated),
                inputs=[
                    var.S_world,
                    var.T_world_link,
                    var.T_world_base,
                    self.T_world_target,
                    self.frame_indices,
                    self.num_frames,
                    spec_tensor.link_ancestor_joints_mask,
                    spec_tensor.joints_to_actuated_mapping,
                    var.has_floating_base,
                    self.residual_weight,
                    row_offset,
                    col_offset,
                ],
                outputs=[out_jacobian],
                device=kernel_device,
            )
        return out_jacobian


# --- kernels ----------------------------------------------------------------
@wp.kernel
def compute_frame_task_weighted_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),  # [batch, 6, num_joints]
    T_world_link: wp.array2d(dtype=wp_vec7),  # [batch, num_links]
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_target: wp.array2d(dtype=wp_vec7),  # [batch_size, num_frames]
    frame_index: int,
    ancestor_mask: wp.array2d(dtype=wp.bool),  # [num_links, num_joints]
    joints_to_actuated: wp.array2d(dtype=wp.float32),  # [num_joints, num_actuated]
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),  # (6,)
    row_offset: int,
    col_offset: int,
    stride: int,  # poses per trajectory: evaluate each one's last
    out_jacobian: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    instance, actuated_idx = wp.tid()  # pyright: ignore
    instance = instance * stride + stride - 1

    num_joints = S_world.shape[2]

    T_world_frame = T_world_link[instance, frame_index]

    n_repeat = wp.max(1, T_world_link.shape[0] // T_world_target.shape[0])
    target_batch_idx = instance // n_repeat
    T_target_world = se3_inverse_func(T_world_target[target_batch_idx, 0])
    T_target_frame = se3_multiply_func(T_target_world, T_world_frame)

    jlog = se3_jlog_func(T_target_frame, wp.float32(1e-4))

    T_frame_world = se3_inverse_func(T_world_frame)

    base_col_offset = 6 if has_floating_base else 0

    spatial_col = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    for joint_idx in range(num_joints):
        if ancestor_mask[frame_index, joint_idx]:
            weight = joints_to_actuated[joint_idx, actuated_idx]
            if weight != 0.0:
                for row in range(6):
                    spatial_col[row] += S_world[instance, row, joint_idx] * weight

    body_col = se3_adjoint_func(T_frame_world) * spatial_col

    jlog_times_body_col = wp.mul(jlog, body_col)

    for row in range(6):
        out_jacobian[instance, row_offset + row, col_offset + base_col_offset + actuated_idx] = (
            -residual_weight[row] * jlog_times_body_col[row]
        )

    if has_floating_base and actuated_idx < 6:
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[actuated_idx] = 1.0

        # right perturbation gives de/dxi = -Jlog(T_target_frame) * Ad(T_frame_base)
        T_frame_base = se3_multiply_func(T_frame_world, T_world_base[instance])
        result_col = se3_adjoint_func(T_frame_base) * unit_vec

        jlog_times_result_col = wp.mul(jlog, result_col)

        for row in range(6):
            out_jacobian[instance, row_offset + row, col_offset + actuated_idx] = (
                -residual_weight[row] * jlog_times_result_col[row]
            )


@wp.kernel
def _terminal_frame_kernel(
    S_world: wp.array3d(dtype=wp.float32),  # [instances, 6, num_joints], with jacobian
    T_world_link: wp.array2d(dtype=wp_vec7),  # [instances, num_links]
    T_world_target: wp.array2d(dtype=wp_vec7),  # [batch_size, 1]
    frame_index: int,
    ancestor_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),  # (6,)
    stride: int,  # poses per trajectory: evaluate each one's last
    jacobian: int,
    tangent_dim: int,  # Jacobian values per residual row
    row_offset: int,
    nnz_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),  # [trajectories, total residual rows]
    out_values: wp.array2d(dtype=wp.float32),  # [trajectories, total sparse values]
):
    """Compute terminal pose residuals and sparse Jacobian values in one launch.

    Thread outputs:
    - Thread 0: residual.
    - Each actuated-column thread: that Jacobian column.
    """
    b, actuated_idx = wp.tid()  # pyright: ignore
    instance = b * stride + stride - 1
    T_world_frame = T_world_link[instance, frame_index]
    T_world_goal = T_world_target[instance // wp.max(1, T_world_link.shape[0] // T_world_target.shape[0]), 0]
    error = se3_log_map_func(se3_multiply_func(se3_inverse_func(T_world_frame), T_world_goal), wp.float32(1e-4))
    if actuated_idx == 0:
        for row in range(6):
            out_residual[b, row_offset + row] = residual_weight[row] * error[row]
    if jacobian == 0:
        return
    jlog = se3_jlog_func(se3_multiply_func(se3_inverse_func(T_world_goal), T_world_frame), wp.float32(1e-4))
    spatial_col = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for joint_idx in range(S_world.shape[2]):
        if ancestor_mask[frame_index, joint_idx]:
            weight = joints_to_actuated[joint_idx, actuated_idx]
            if weight != 0.0:
                for row in range(6):
                    spatial_col[row] += S_world[instance, row, joint_idx] * weight
    column = wp.mul(jlog, se3_adjoint_func(se3_inverse_func(T_world_frame)) * spatial_col)
    for row in range(6):
        out_values[b, nnz_offset + row * tangent_dim + actuated_idx] = -residual_weight[row] * column[row]


@wp.kernel
def compute_frame_task_weighted_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),  # (num_instances, num_links)
    frame_index: int,
    T_world_target: wp.array2d(dtype=wp_vec7),  # (batch_size, num_frames)
    residual_weight: wp.array1d(dtype=wp.float32),  # (6,)
    row_offset: int,
    stride: int,  # poses per trajectory: evaluate each one's last
    out_residual: wp.array2d(dtype=wp.float32),  # (num_instances, total_residual_dim)
):
    idx = wp.tid() * stride + stride - 1  # pyright: ignore[reportOperatorIssue]

    n_repeat = wp.max(1, T_world_link.shape[0] // T_world_target.shape[0])
    target_batch_idx = idx // n_repeat

    T_world_frame = T_world_link[idx, frame_index]
    T_frame_world = se3_inverse_func(T_world_frame)
    T_frame_target = se3_multiply_func(T_frame_world, T_world_target[target_batch_idx, 0])
    error_in_frame = se3_log_map_func(T_frame_target, wp.float32(1e-4))

    for i in range(6):
        out_residual[idx, row_offset + i] = residual_weight[i] * error_in_frame[i]


@wp.kernel
def compute_multi_frame_task_weighted_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),  # (num_instances, num_links)
    frame_indices: wp.array1d(dtype=wp.int32),  # (num_frames,)
    T_world_target: wp.array2d(dtype=wp_vec7),  # (batch_size, num_frames)
    num_frames: int,
    residual_weight: wp.array1d(dtype=wp.float32),  # (num_frames * 6,) per-frame weights
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),  # (num_instances, total_residual_dim)
):
    idx = wp.tid()
    n_repeat = wp.max(1, T_world_link.shape[0] // T_world_target.shape[0])
    target_batch_idx = idx // n_repeat  # pyright: ignore[reportOperatorIssue]

    for frame_num in range(num_frames):
        frame_index = frame_indices[frame_num]

        T_world_frame = T_world_link[idx, frame_index]
        T_frame_world = se3_inverse_func(T_world_frame)
        T_frame_target = se3_multiply_func(T_frame_world, T_world_target[target_batch_idx, frame_num])
        error_in_frame = se3_log_map_func(T_frame_target, wp.float32(1e-4))

        residual_offset = row_offset + frame_num * 6
        weight_offset = frame_num * 6
        for i in range(6):
            out_residual[idx, residual_offset + i] = residual_weight[weight_offset + i] * error_in_frame[i]


@wp.kernel
def compute_multi_frame_task_weighted_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),  # [batch, 6, num_joints]
    T_world_link: wp.array2d(dtype=wp_vec7),  # [batch, num_links]
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    T_world_target: wp.array2d(dtype=wp_vec7),  # [batch_size, num_frames]
    frame_indices: wp.array1d(dtype=wp.int32),  # [num_frames]
    num_frames: int,
    ancestor_mask: wp.array2d(dtype=wp.bool),  # [num_links, num_joints]
    joints_to_actuated: wp.array2d(dtype=wp.float32),  # [num_joints, num_actuated]
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),  # (num_frames * 6,) per-frame weights
    row_offset: int,
    col_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    instance, actuated_idx = wp.tid()  # pyright: ignore

    num_joints = S_world.shape[2]
    n_repeat = wp.max(1, T_world_link.shape[0] // T_world_target.shape[0])
    target_batch_idx = instance // n_repeat
    base_col_offset = 6 if has_floating_base else 0

    for frame_num in range(num_frames):
        frame_index = frame_indices[frame_num]
        residual_offset = row_offset + frame_num * 6
        weight_offset = frame_num * 6

        T_world_frame = T_world_link[instance, frame_index]
        T_target_world = se3_inverse_func(T_world_target[target_batch_idx, frame_num])
        T_target_frame = se3_multiply_func(T_target_world, T_world_frame)

        jlog = se3_jlog_func(T_target_frame, wp.float32(1e-4))
        T_frame_world = se3_inverse_func(T_world_frame)

        spatial_col = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for joint_idx in range(num_joints):
            if ancestor_mask[frame_index, joint_idx]:
                weight = joints_to_actuated[joint_idx, actuated_idx]
                if weight != 0.0:
                    for row in range(6):
                        spatial_col[row] += S_world[instance, row, joint_idx] * weight

        body_col = se3_adjoint_func(T_frame_world) * spatial_col
        jlog_times_body_col = wp.mul(jlog, body_col)

        for row in range(6):
            out_jacobian[instance, residual_offset + row, col_offset + base_col_offset + actuated_idx] = (
                -residual_weight[weight_offset + row] * jlog_times_body_col[row]
            )

        if has_floating_base and actuated_idx < 6:
            unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            unit_vec[actuated_idx] = 1.0

            # right perturbation gives de/dxi = -Jlog(T_target_frame) * Ad(T_frame_base)
            T_frame_base = se3_multiply_func(T_frame_world, T_world_base[instance])
            result_col = se3_adjoint_func(T_frame_base) * unit_vec
            jlog_times_result_col = wp.mul(jlog, result_col)

            for row in range(6):
                out_jacobian[instance, residual_offset + row, col_offset + actuated_idx] = (
                    -residual_weight[weight_offset + row] * jlog_times_result_col[row]
                )
