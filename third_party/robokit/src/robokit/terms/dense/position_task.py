# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false
from typing import Any, Mapping, Optional, Sequence, Union, cast

import numpy as np
import warp as wp

from robokit.lie.se3_kernels import se3_adjoint_func
from robokit.opt.var_values import VarValues
from robokit.robo.robot import Robot
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import Context, ResidualTask
from robokit.utils.warp_utils import pose_array, wp_vec6, wp_vec7


# --- task -------------------------------------------------------------------
class PositionTask(RobotTask, ResidualTask):
    """Match the world positions of one or more link frames.

    Example:
        >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
        >>> from robokit.opt.var_values import VarKey, VarValues
        >>> from robokit.robo.robot import Robot
        >>> robot = Robot.load(load_robot_description("ur10_description"))
        >>> target_link_index = robot.link_names.index("ee_link")
        >>> state = robot.state(q=robot.zero_q)
        >>> target_link_pose = robot.forward_kinematics(state).get_T_world_link(target_link_index).reshape((1, 1))
        >>> position_task = PositionTask(robot, target_link_index, target_link_pose, weight=1.0)
        >>> random_q = wp.from_numpy(np.array([[-0.5, 0.3, -0.2, 0.4, -0.1, 0.25]], dtype=np.float32), dtype=wp.float32)
        >>> state = robot.state(q=random_q)
        >>> robot.forward_kinematics(state)
        >>> robot.compute_motion_subspace(state)
        >>> weighted_residual = position_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))
        >>> weighted_jacobian = position_task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state}))
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
        weight: Union[float, Sequence[float]] = 20.0,
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
            weight_np = np.repeat(weight, 3).astype(np.float32)
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
        PositionTask.__init__(self, robot, frame_index, T_world_target, self.weight, self.fixed_target, self.name)

    def set_target(self, T_world_target: Union[wp.array, np.ndarray]):
        """Update target poses in-place. Batch size must match the original."""
        wp.copy(self.T_world_target, pose_array(T_world_target, device=self.device))

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Complete a stub with the helper's links and target buffer, which it reads directly."""
        super().resolve(var_values, context)
        if self.frame_index_arg is None or self.T_world_target_arg is None:
            raise ValueError(f"{type(self).__name__} needs frame_index and T_world_target, e.g. a helper's Context.")
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
            kernel=compute_position_task_weighted_residual_kernel,
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
            kernel=compute_position_task_weighted_jacobian_kernel,
            dim=(batch_size, self.num_frames, num_actuated),
            inputs=[
                var.S_world,
                var.T_world_link,
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


# --- kernels ----------------------------------------------------------------
@wp.kernel
def compute_position_task_weighted_residual_kernel(
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

    # world-frame position error
    ee_pos = wp.vec3(T_world_frame[0], T_world_frame[1], T_world_frame[2])
    target_pos = wp.vec3(target[0], target[1], target[2])
    error = target_pos - ee_pos

    residual_offset = row_offset + frame_num * 3
    weight_offset = frame_num * 3
    for i in range(3):
        residual_buffer[idx, residual_offset + i] = residual_weight[weight_offset + i] * error[i]


@wp.kernel
def compute_position_task_weighted_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),  # [batch, 6, num_joints]
    T_world_link: wp.array2d(dtype=wp_vec7),  # [batch, num_links]
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

    # link position in the world frame
    T_world_frame = T_world_link[instance, frame_index]
    ee_pos = wp.vec3(T_world_frame[0], T_world_frame[1], T_world_frame[2])

    # accumulate the motion-subspace column
    spatial_col = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for joint_idx in range(num_joints):
        if ancestor_mask[frame_index, joint_idx]:
            weight = joints_to_actuated[joint_idx, actuated_idx]
            if weight != 0.0:
                for row in range(6):
                    spatial_col[row] += S_world[instance, row, joint_idx] * weight

    # point Jacobian column
    v = wp.vec3(spatial_col[0], spatial_col[1], spatial_col[2])
    omega = wp.vec3(spatial_col[3], spatial_col[4], spatial_col[5])
    v_ee = v + wp.cross(omega, ee_pos)

    for row in range(3):
        jacobian_buffer[instance, residual_offset + row, col_offset + base_col_offset + actuated_idx] = (
            -residual_weight[weight_offset + row] * v_ee[row]
        )

    if has_floating_base and actuated_idx < 6:
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[actuated_idx] = 1.0
        spatial_twist = se3_adjoint_func(T_world_base[instance]) * unit_vec
        v_fb = wp.vec3(spatial_twist[0], spatial_twist[1], spatial_twist[2])
        omega_fb = wp.vec3(spatial_twist[3], spatial_twist[4], spatial_twist[5])
        v_ee_fb = v_fb + wp.cross(omega_fb, ee_pos)

        for row in range(3):
            jacobian_buffer[instance, residual_offset + row, col_offset + actuated_idx] = (
                -residual_weight[weight_offset + row] * v_ee_fb[row]
            )
