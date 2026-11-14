# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false
import math
from typing import Any, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np
import warp as wp

from robokit.lie.se3_kernels import se3_adjoint_func
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import Context, ResidualTask
from robokit.utils.warp_utils import pose_array, wp_vec6, wp_vec7
from robokit.xform.warp.rotation_conversions import (
    quaternion_apply_func,
    quaternion_invert_func,
    quaternion_multiply_func,
    quaternion_to_axis_angle_func,
)


# --- tasks ------------------------------------------------------------------
class RotationTask(RobotTask, ResidualTask):
    """Match the world rotations of one or more link frames.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.opt.var_values import VarKey, VarValues
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("ur10_description"))
        >>> target_link_index = robot.link_names.index("ee_link")
        >>> state = robot.state(q=robot.zero_q)
        >>> target_link_pose = robot.forward_kinematics(state).get_T_world_link(target_link_index).reshape((1, 1))
        >>> rotation_task = RotationTask(robot, target_link_index, target_link_pose, weight=1.0)
        >>> random_q = wp.from_numpy(np.array([[-0.5, 0.3, -0.2, 0.4, -0.1, 0.25]], dtype=np.float32), dtype=wp.float32)
        >>> state = robot.state(q=random_q)
        >>> robot.forward_kinematics(state)
        >>> robot.compute_motion_subspace(state)
        >>> weighted_residual = rotation_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))
        >>> weighted_jacobian = rotation_task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state}))
        >>> weighted_residual.numpy().shape[1] == 3
        True
        >>> weighted_jacobian.numpy().shape[1] == 3
        True
    """

    def __init__(
        self,
        robot: Optional[Robot] = None,
        frame_index: Optional[Union[int, Sequence[int], Context]] = None,
        T_world_target: Optional[Union[wp.array, np.ndarray, Context]] = None,
        weight: Union[float, Sequence[float], np.ndarray, Mapping[str, float]] = 10.0,
        fixed_target: bool = False,
        name: str = "",
    ):
        self.name = name
        self.weight = weight
        self.fixed_target = fixed_target
        self.robot: Optional[Robot] = None
        self.frame_index_arg, self.T_world_target_arg = frame_index, T_world_target
        if (
            frame_index is None
            or T_world_target is None
            or any(isinstance(a, Context) for a in (frame_index, T_world_target))
        ):
            return  # config-time stub; `resolve` or `set_frames` completes it
        T_world_target = pose_array(T_world_target)

        frame_indices_list = [frame_index] if isinstance(frame_index, int) else list(frame_index)
        if len(frame_indices_list) != T_world_target.shape[1]:
            raise ValueError(
                f"Number of frame indices ({len(frame_indices_list)}) must match "
                f"number of targets ({T_world_target.shape[1]})"
            )

        self.num_frames = len(frame_indices_list)
        self.device = T_world_target.device

        self.frame_indices = wp.from_numpy(
            np.array(frame_indices_list, dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self.T_world_target = wp.clone(T_world_target)
        if isinstance(weight, (int, float)):
            weight_np = np.full(self.num_frames * 3, weight, dtype=np.float32)
        else:
            weight_np = np.repeat(np.asarray(weight, dtype=np.float32), 3)
        self.residual_weight = wp.from_numpy(weight_np, dtype=wp.float32, device=self.device)
        if robot is not None:
            self.set_robot(robot)

    def set_robot(self, robot: Robot):
        self.robot = robot

    def set_weight(self, weight: Union[float, Sequence[float]]):
        """Update residual weights in-place. Accepts scalar or per-frame sequence."""
        if isinstance(weight, (int, float)):
            arr = np.full(self.num_frames * 3, weight, dtype=np.float32)
        else:
            arr = np.repeat(weight, 3).astype(np.float32)
        wp.copy(self.residual_weight, wp.from_numpy(arr, dtype=wp.float32, device="cpu"))

    def set_frames(
        self, robot: Robot, frame_index: Union[int, Sequence[int]], T_world_target: Union[wp.array, np.ndarray]
    ):
        """Give a config-time stub its robot, link indices, and targets, keeping the configured weights."""
        RotationTask.__init__(self, robot, frame_index, T_world_target, self.weight, self.fixed_target, self.name)

    def set_target(self, T_world_target: Union[wp.array, np.ndarray]):
        """Update target poses in-place. Batch size must match the original."""
        wp.copy(self.T_world_target, pose_array(T_world_target, device=self.device))

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Complete a stub with the helper's links and target buffer (not copied); named weights zero unlisted links."""
        super().resolve(var_values, context)
        if self.frame_index_arg is None or self.T_world_target_arg is None:
            raise ValueError(f"{type(self).__name__} needs frame_index and T_world_target, e.g. a helper's Context.")
        if isinstance(self.weight, Mapping):
            links = np.atleast_1d(np.asarray(self.frame_index_arg, dtype=np.int32))
            self.weight = [self.weight.get(cast(Robot, self.robot).link_names[link], 0.0) for link in links]
        target = cast(wp.array, self.T_world_target_arg)
        self.set_frames(cast(Robot, self.robot), cast(Union[int, Sequence[int]], self.frame_index_arg), target)
        self.T_world_target = target

    @property
    def residual_dim(self) -> int:
        return 3 * self.num_frames

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

        wp.launch(
            kernel=compute_rotation_task_weighted_residual_kernel,
            dim=(batch_size, self.num_frames),
            inputs=[
                var.T_world_link,
                self.frame_indices,
                self.T_world_target,
                self.residual_weight,
                row_offset,
            ],
            outputs=[out_residual],
            device=kernel_device,
        )
        return out_residual

    def compute_weighted_jacobian_analytic(
        self,
        var_values: VarValues,
        out_jacobian: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
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

        wp.launch(
            kernel=compute_rotation_task_weighted_jacobian_kernel,
            dim=(batch_size, self.num_frames, num_actuated),
            inputs=[
                var.S_world,
                var.T_world_base,
                self.frame_indices,
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


class AxisLimitTask(RobotTask, ResidualTask):
    """Limit the angle between a frame-local axis and a world-frame axis."""

    def __init__(
        self,
        robot: Optional[Robot] = None,
        frame_index: Optional[int] = None,
        local_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        world_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        min_angle: Optional[float] = None,
        max_angle: Optional[float] = None,
        weight: float = 1.0,
    ):
        if min_angle is None and max_angle is None:
            raise ValueError("At least one angle limit is required.")
        if min_angle is not None and not 0.0 <= min_angle <= math.pi:
            raise ValueError("min_angle must be in [0, pi].")
        if max_angle is not None and not 0.0 <= max_angle <= math.pi:
            raise ValueError("max_angle must be in [0, pi].")
        if min_angle is not None and max_angle is not None and min_angle > max_angle:
            raise ValueError("min_angle must not exceed max_angle.")

        local = np.asarray(local_axis, dtype=np.float32)
        world = np.asarray(world_axis, dtype=np.float32)
        if np.linalg.norm(local) == 0.0 or np.linalg.norm(world) == 0.0:
            raise ValueError("Axes must be nonzero.")
        local /= np.linalg.norm(local)
        world /= np.linalg.norm(world)

        self.robot = robot
        self.frame_index = frame_index
        self.local_axis = wp.vec3(*local)
        self.world_axis = wp.vec3(*world)
        self.cos_min = math.cos(min_angle) if min_angle is not None else 0.0
        self.cos_max = math.cos(max_angle) if max_angle is not None else 0.0
        self.has_min = min_angle is not None
        self.has_max = max_angle is not None
        self.weight = float(weight)
        self.residual_weight = self.weight

    def set_robot(self, robot: Robot):
        self.robot = robot

    def set_weight(self, weight: float):
        self.weight = float(weight)
        self.residual_weight = self.weight

    @property
    def residual_dim(self) -> int:
        return 1

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        device = out_residual.device if out_residual is not None else var.q.device
        if out_residual is None:
            out_residual = wp.empty((var.batch_size, 1), dtype=wp.float32, device=device)
            row_offset = 0
        wp.launch(
            kernel=_axis_limit_residual_kernel,
            dim=var.batch_size,
            inputs=[
                var.T_world_link,
                self.frame_index,
                self.local_axis,
                self.world_axis,
                self.cos_min,
                self.cos_max,
                self.has_min,
                self.has_max,
                self.weight,
                row_offset,
            ],
            outputs=[out_residual],
            device=device,
        )
        return out_residual

    def compute_weighted_jacobian_analytic(
        self,
        var_values: VarValues,
        out_jacobian: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        col_offset = var_values.tangent_offset(self.var_key)
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)
        device = out_jacobian.device if out_jacobian is not None else var.q.device
        if out_jacobian is None:
            out_jacobian = wp.zeros((var.batch_size, 1, var.tangent_dim), dtype=wp.float32, device=device)
            row_offset = 0
            col_offset = 0
        wp.launch(
            kernel=_axis_limit_jacobian_kernel,
            dim=(var.batch_size, var.tangent_dim),
            inputs=[
                var.T_world_link,
                var.T_world_base,
                var.S_world,
                var.spec_tensors.link_ancestor_joints_mask,
                var.spec_tensors.joints_to_actuated_mapping,
                var.has_floating_base,
                self.frame_index,
                self.local_axis,
                self.world_axis,
                self.cos_min,
                self.cos_max,
                self.has_min,
                self.has_max,
                self.weight,
                row_offset,
                col_offset,
            ],
            outputs=[out_jacobian],
            device=device,
        )
        return out_jacobian


# --- kernels ----------------------------------------------------------------
@wp.kernel
def compute_rotation_task_weighted_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),  # (num_instances, num_links)
    frame_indices: wp.array1d(dtype=wp.int32),  # (num_frames,)
    T_world_target: wp.array2d(dtype=wp_vec7),  # (num_instances, num_frames)
    residual_weight: wp.array1d(dtype=wp.float32),  # (num_frames * 3,)
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),  # (num_instances, total_residual_dim)
):
    idx, frame_num = wp.tid()  # pyright: ignore

    frame_index = frame_indices[frame_num]
    T_world_frame = T_world_link[idx, frame_index]
    n_repeat = wp.max(1, T_world_link.shape[0] // T_world_target.shape[0])
    target = T_world_target[idx // n_repeat, frame_num]

    q_world_frame = wp.vec4(T_world_frame[3], T_world_frame[4], T_world_frame[5], T_world_frame[6])
    q_world_target = wp.vec4(target[3], target[4], target[5], target[6])

    q_target_inv = quaternion_invert_func(q_world_target)
    q_err = quaternion_multiply_func(q_world_frame, q_target_inv)
    error = quaternion_to_axis_angle_func(q_err)

    residual_offset = row_offset + frame_num * 3
    weight_offset = frame_num * 3
    for i in range(3):
        residual_buffer[idx, residual_offset + i] = residual_weight[weight_offset + i] * error[i]


@wp.kernel
def compute_rotation_task_weighted_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),  # [batch, 6, num_joints]
    T_world_base: wp.array1d(dtype=wp_vec7),  # [batch]
    frame_indices: wp.array1d(dtype=wp.int32),  # [num_frames]
    ancestor_mask: wp.array2d(dtype=wp.bool),  # [num_links, num_joints]
    joints_to_actuated: wp.array2d(dtype=wp.float32),  # [num_joints, num_actuated]
    has_floating_base: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),  # (num_frames * 3,)
    row_offset: int,
    col_offset: int,
    jacobian_buffer: wp.array3d(dtype=wp.float32),  # [batch, total_residual_dim, total_dofs]
):
    instance, frame_num, actuated_idx = wp.tid()  # pyright: ignore

    num_joints = S_world.shape[2]
    base_col_offset = 6 if has_floating_base else 0

    frame_index = frame_indices[frame_num]
    residual_offset = row_offset + frame_num * 3
    weight_offset = frame_num * 3

    # accumulate the angular motion-subspace column
    omega_col = wp.vec3(0.0, 0.0, 0.0)
    for joint_idx in range(num_joints):
        if ancestor_mask[frame_index, joint_idx]:
            weight = joints_to_actuated[joint_idx, actuated_idx]
            if weight != 0.0:
                omega_col += weight * wp.vec3(
                    S_world[instance, 3, joint_idx],
                    S_world[instance, 4, joint_idx],
                    S_world[instance, 5, joint_idx],
                )

    # angular geometric Jacobian
    for row in range(3):
        jacobian_buffer[instance, residual_offset + row, col_offset + base_col_offset + actuated_idx] = (
            residual_weight[weight_offset + row] * omega_col[row]
        )

    if has_floating_base and actuated_idx < 6:
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[actuated_idx] = 1.0
        spatial_twist = se3_adjoint_func(T_world_base[instance]) * unit_vec
        world_ang_fb = wp.vec3(spatial_twist[3], spatial_twist[4], spatial_twist[5])

        for row in range(3):
            jacobian_buffer[instance, residual_offset + row, col_offset + actuated_idx] = (
                residual_weight[weight_offset + row] * world_ang_fb[row]
            )


@wp.kernel
def _axis_limit_residual_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    frame_index: int,
    local_axis: wp.vec3,
    world_axis: wp.vec3,
    cos_min: float,
    cos_max: float,
    has_min: wp.bool,
    has_max: wp.bool,
    weight: float,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    T = T_world_link[batch_idx, frame_index]
    axis_world = quaternion_apply_func(wp.vec4(T[3], T[4], T[5], T[6]), local_axis)
    cosine = wp.dot(axis_world, world_axis)  # pyright: ignore[reportCallIssue]
    residual = wp.float32(0.0)
    if has_min:
        residual += wp.max(cosine - cos_min, wp.float32(0.0))
    if has_max:
        residual += wp.max(cos_max - cosine, wp.float32(0.0))
    out_residual[batch_idx, row_offset] = weight * residual


@wp.kernel
def _axis_limit_jacobian_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    T_world_base: wp.array1d(dtype=wp_vec7),
    S_world: wp.array3d(dtype=wp.float32),
    ancestor_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    has_floating_base: wp.bool,
    frame_index: int,
    local_axis: wp.vec3,
    world_axis: wp.vec3,
    cos_min: float,
    cos_max: float,
    has_min: wp.bool,
    has_max: wp.bool,
    weight: float,
    row_offset: int,
    col_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),
):
    batch_idx, col_idx = wp.tid()  # pyright: ignore
    T = T_world_link[batch_idx, frame_index]
    axis_world = quaternion_apply_func(wp.vec4(T[3], T[4], T[5], T[6]), local_axis)
    cosine = wp.dot(axis_world, world_axis)  # pyright: ignore[reportCallIssue]
    scale = wp.float32(0.0)
    if has_min and cosine > cos_min:
        scale += wp.float32(1.0)
    if has_max and cosine < cos_max:
        scale -= wp.float32(1.0)
    if scale == wp.float32(0.0):
        return

    base_dofs = 6 if has_floating_base else 0
    omega = wp.vec3(0.0, 0.0, 0.0)
    if has_floating_base and col_idx < 6:
        unit = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit[col_idx] = wp.float32(1.0)
        twist = se3_adjoint_func(T_world_base[batch_idx]) * unit
        omega = wp.vec3(twist[3], twist[4], twist[5])
    else:
        actuated_idx = col_idx - base_dofs
        num_joints = S_world.shape[2]
        for joint_idx in range(num_joints):
            if ancestor_mask[frame_index, joint_idx]:
                joint_weight = joints_to_actuated[joint_idx, actuated_idx]
                if joint_weight != wp.float32(0.0):
                    omega += joint_weight * wp.vec3(
                        S_world[batch_idx, 3, joint_idx],
                        S_world[batch_idx, 4, joint_idx],
                        S_world[batch_idx, 5, joint_idx],
                    )
    derivative = wp.dot(world_axis, wp.cross(omega, axis_world))  # pyright: ignore[reportCallIssue]
    out_jacobian[batch_idx, row_offset, col_offset + col_idx] = weight * scale * derivative
