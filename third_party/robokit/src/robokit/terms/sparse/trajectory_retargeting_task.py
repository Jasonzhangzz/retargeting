# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportUndefinedVariable=false
# pyright: reportCallIssue=false
"""Sparse pairwise retargeting residuals over a trajectory."""

from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.lie.se3_kernels import se3_adjoint_func
from robokit.opt.var_values import VarKey
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import Context, SparseTask, SparsityPattern
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


# --- task -------------------------------------------------------------------
class TrajectoryRetargetingTask(RobotTask, SparseTask):
    """Pairwise position and direction residuals for trajectory retargeting."""

    num_pairs = 1  # a template counts as active until `resolve` sizes it

    def __init__(
        self,
        robot: Optional["Robot"] = None,  # noqa: F821
        target_keypoints: Optional[Union[np.ndarray, wp.array, Context]] = None,  # [batch, frames, keypoints, 3]
        robot_link_indices: Optional[Union[List[int], Context]] = None,
        target_joint_indices: Optional[Union[List[int], Context]] = None,
        pair_indices: Optional[Union[Sequence[Sequence[int]], np.ndarray, Context]] = None,
        position_weight: float = 1.0,
        angle_weight: float = 1.0,
        target_scale: Union[float, wp.array, Context] = 1.0,
        name: str = "",
    ):
        self.name = name
        self.robot = robot
        self.robot_link_indices_list = robot_link_indices
        self.target_joint_indices_list = target_joint_indices
        self.position_weight = position_weight
        self.angle_weight = angle_weight
        self.target_scale = target_scale
        self.target_keypoints_input = target_keypoints
        self.pair_indices_arg = pair_indices

        if robot is None or robot_link_indices is None or target_joint_indices is None or pair_indices is None:
            return

        self.num_selected = len(robot_link_indices)
        assert len(target_joint_indices) == self.num_selected
        pair_indices_array = np.asarray(pair_indices, dtype=np.int32)
        if pair_indices_array.ndim != 2 or pair_indices_array.shape[1] != 2:
            raise ValueError("pair_indices must have shape [num_pairs, 2]")
        pair_rows_input = pair_indices_array[:, 0]
        pair_cols_input = pair_indices_array[:, 1]
        num_pairs = int(pair_indices_array.shape[0])

        self.device: Optional[wp_device_type] = None
        self.target_keypoints: Optional[wp.array] = None
        self.target_scales: Optional[wp.array] = None
        self.robot_link_indices: Optional[wp.array] = None
        self.target_joint_indices: Optional[wp.array] = None
        self.pair_rows: Optional[wp.array] = None
        self.pair_cols: Optional[wp.array] = None
        self.pair_rows_input = pair_rows_input
        self.pair_cols_input = pair_cols_input
        self.num_pairs = num_pairs

        # ancestor mask for each selected link
        self.link_ancestor_masks: Optional[wp.array] = None
        self.num_unique_dofs: int = 0

        # active DOFs shared by all selected links
        self.all_ancestor_dof_indices: Optional[wp.array] = None

        self.num_frames: int = 0

    @property
    def active(self) -> bool:
        return bool(self.position_weight or self.angle_weight) and self.num_pairs > 0

    def resolve(self, var_values: "VarValues", context: Mapping[Context, Any]):  # noqa: F821
        """Build from the helper's points, links, pairs and scales; Warp points and scales are read directly."""
        super().resolve(var_values, context)
        TrajectoryRetargetingTask.__init__(
            self,
            self.robot,
            self.target_keypoints_input,
            self.robot_link_indices_list,
            self.target_joint_indices_list,
            self.pair_indices_arg,
            self.position_weight,
            self.angle_weight,
            self.target_scale,
            self.name,
        )
        self.init_buffers(var_values.get(self.var_key).q.device)

    def init_buffers(self, device: wp_device_type):
        """Build target, pair, and sparse ancestor buffers.

        Lifecycle:
            1. Upload targets and pair indices.
            2. Build selected-link ancestor masks.
            3. Collect the active actuated columns.
        """
        self.device = device

        if self.target_keypoints is None:
            if isinstance(self.target_keypoints_input, np.ndarray):
                target_np = self.target_keypoints_input
                if target_np.ndim == 3:
                    target_np = target_np[None, ...]
                self.target_keypoints = wp.from_numpy(target_np.astype(np.float32), dtype=wp.float32, device=device)
            else:
                self.target_keypoints = self.target_keypoints_input.to(device)

        self.num_frames = self.target_keypoints.shape[1]
        if self.target_scales is None:
            self.target_scales = (
                self.target_scale.to(device)
                if isinstance(self.target_scale, wp.array)
                else wp.array([self.target_scale], dtype=wp.float32, device=device)
            )
            if (
                self.target_scales.dtype != wp.float32
                or self.target_scales.ndim != 1
                or self.target_scales.shape[0] not in (1, self.target_keypoints.shape[0])
            ):
                raise ValueError("target_scale must be a scalar or one float32 value per target trajectory.")

        self.robot_link_indices = wp.from_numpy(
            np.array(self.robot_link_indices_list, dtype=np.int32), dtype=wp.int32, device=device
        )
        self.target_joint_indices = wp.from_numpy(
            np.array(self.target_joint_indices_list, dtype=np.int32), dtype=wp.int32, device=device
        )
        pair_rows = self.pair_rows_input
        pair_cols = self.pair_cols_input
        self.num_pairs = int(pair_rows.shape[0])
        self.pair_rows = wp.from_numpy(pair_rows, dtype=wp.int32, device=device)
        self.pair_cols = wp.from_numpy(pair_cols, dtype=wp.int32, device=device)

        # build selected-link ancestor masks
        num_joints = self.robot.spec.num_joints

        link_ancestor_masks_np = np.zeros((self.num_selected, num_joints), dtype=np.bool_)
        for idx, link_idx in enumerate(self.robot_link_indices_list):
            link_ancestor_masks_np[idx] = self.robot.spec.link_ancestor_joints_mask[link_idx]

        self.link_ancestor_masks = wp.from_numpy(link_ancestor_masks_np, dtype=wp.bool, device=device)

        # collect actuated columns that control selected links
        joints_to_actuated = self.robot.spec.joints_to_actuated_mapping
        all_ancestor_mask = np.any(link_ancestor_masks_np, axis=0)
        ancestor_indices = np.where(all_ancestor_mask)[0]

        unique_dofs = set()
        for joint_idx in ancestor_indices:
            for actuated_idx in range(joints_to_actuated.shape[1]):
                if joints_to_actuated[joint_idx, actuated_idx] != 0.0:
                    unique_dofs.add(actuated_idx)

        unique_dofs_arr = np.array(sorted(unique_dofs), dtype=np.int32)
        self.num_unique_dofs = len(unique_dofs_arr)
        self.all_ancestor_dof_indices = wp.from_numpy(unique_dofs_arr, dtype=wp.int32, device=device)

    @property
    def residual_dim(self) -> int:
        if self.num_frames == 0 and self.target_keypoints_input is not None:
            if isinstance(self.target_keypoints_input, np.ndarray):
                self.num_frames = (
                    self.target_keypoints_input.shape[1]
                    if self.target_keypoints_input.ndim == 4
                    else self.target_keypoints_input.shape[0]
                )
            else:
                self.num_frames = self.target_keypoints_input.shape[1]
        return self.num_frames * self.num_pairs * 4

    def compute_weighted_residual(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        robot_state = var_values.get(VarKey.ROBOT)
        if self.device is None:
            self.init_buffers(robot_state.q.device)

        if not robot_state.is_fk_computed:
            self.robot.forward_kinematics(robot_state)

        kernel_device = out_residual.device if out_residual is not None else self.device

        if out_residual is None:
            out_residual = wp.empty((robot_state.q.shape[0], self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        num_pairs = self.num_pairs

        wp.launch(
            kernel=compute_trajectory_retargeting_residual_kernel,
            dim=(robot_state.q.shape[0], self.num_frames, num_pairs),
            inputs=[
                robot_state.T_world_link,
                self.target_keypoints,
                self.robot_link_indices,
                self.target_joint_indices,
                self.pair_rows,
                self.pair_cols,
                self.target_scales,
                self.position_weight,
                self.angle_weight,
                self._get_frame_mask(),
                self.use_mask,
                num_pairs,
                row_offset,
            ],
            outputs=[out_residual],
            device=kernel_device,
        )

        return out_residual

    def compute_sparse_jacobian_pattern(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        offset: int = 0,
        out_row_indices: Optional[wp.array] = None,
        out_col_indices: Optional[wp.array] = None,
        nnz_offset: int = 0,
        **kwargs: object,
    ) -> SparsityPattern:
        robot_state = var_values.get(VarKey.ROBOT)
        if self.device is None:
            self.init_buffers(robot_state.q.device)

        kernel_device = robot_state.q.device
        num_dofs = self.robot.spec.num_actuated_joints
        base_dim = 6 if bool(robot_state.has_floating_base) else 0
        single_tangent_dim = base_dim + num_dofs
        num_pairs = self.num_pairs
        cols_per_residual = self.num_unique_dofs + base_dim
        nnz_per_pair = 4 * cols_per_residual
        nnz = self.num_frames * num_pairs * nnz_per_pair

        row_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_row_indices is None else out_row_indices
        )
        col_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_col_indices is None else out_col_indices
        )

        wp.launch(
            kernel=compute_trajectory_retargeting_jacobian_sparse_pattern_kernel,
            dim=(self.num_frames, num_pairs),
            inputs=[
                self.all_ancestor_dof_indices,
                self.num_unique_dofs,
                single_tangent_dim,
                base_dim,
                num_pairs,
                offset,
                nnz_offset,
            ],
            outputs=[row_indices, col_indices],
            device=kernel_device,
        )

        pattern = SparsityPattern()
        pattern.row_indices = row_indices
        pattern.col_indices = col_indices
        return pattern

    def compute_weighted_sparse_jacobian_values(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        out_jacobian_values: Optional[wp.array] = None,
        offset: int = 0,
        nnz_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        robot_state = var_values.get(VarKey.ROBOT)
        if self.device is None:
            self.init_buffers(robot_state.q.device)

        if not robot_state.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(robot_state)

        kernel_device = robot_state.q.device
        base_dim = 6 if bool(robot_state.has_floating_base) else 0
        num_pairs = self.num_pairs
        cols_per_residual = self.num_unique_dofs + base_dim
        nnz_per_pair = 4 * cols_per_residual
        nnz = self.num_frames * num_pairs * nnz_per_pair

        if out_jacobian_values is None:
            out_jacobian_values = wp.empty((robot_state.q.shape[0], nnz), dtype=wp.float32, device=kernel_device)

        spec_tensors = robot_state.spec_tensors

        wp.launch(
            kernel=compute_trajectory_retargeting_jacobian_sparse_values_kernel,
            dim=(robot_state.q.shape[0], self.num_frames, num_pairs),
            inputs=[
                robot_state.S_world,
                robot_state.T_world_link,
                robot_state.T_world_base,
                self.target_keypoints,
                self.robot_link_indices,
                self.target_joint_indices,
                self.pair_rows,
                self.pair_cols,
                self.all_ancestor_dof_indices,
                self.link_ancestor_masks,
                spec_tensors.joints_to_actuated_mapping,
                self.num_unique_dofs,
                self.target_scales,
                self.position_weight,
                self.angle_weight,
                self._get_frame_mask(),
                self.use_mask,
                base_dim,
                num_pairs,
                nnz_offset,
            ],
            outputs=[out_jacobian_values],
            device=kernel_device,
        )

        return out_jacobian_values


# --- kernels ----------------------------------------------------------------
@wp.kernel
def compute_trajectory_retargeting_residual_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),  # [batch, frames, num_links]
    target_keypoints: wp.array4d(dtype=wp.float32),  # [batch, frames, num_keypoints, 3]
    robot_link_indices: wp.array1d(dtype=wp.int32),
    target_joint_indices: wp.array1d(dtype=wp.int32),
    pair_rows: wp.array1d(dtype=wp.int32),
    pair_cols: wp.array1d(dtype=wp.int32),
    target_scales: wp.array1d(dtype=wp.float32),
    position_weight: float,
    angle_weight: float,
    frame_mask: wp.array2d(dtype=wp.uint8),
    use_mask: bool,
    num_pairs: int,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),  # [batch, residual_dim]
):
    """Compute pairwise position and direction residuals."""
    batch_idx, frame_idx, pair_idx = wp.tid()  # pyright: ignore[reportAssignmentType]

    row = pair_rows[pair_idx]
    col = pair_cols[pair_idx]

    # frame and pair row offset
    frame_residual_offset = frame_idx * num_pairs * 4
    pair_residual_offset = pair_idx * 4
    base_idx = row_offset + frame_residual_offset + pair_residual_offset

    # diagonal pairs have zero residual
    if row == col or (use_mask and frame_mask[batch_idx, frame_idx] == wp.uint8(0)):
        out_residual[batch_idx, base_idx + 0] = 0.0
        out_residual[batch_idx, base_idx + 1] = 0.0
        out_residual[batch_idx, base_idx + 2] = 0.0
        out_residual[batch_idx, base_idx + 3] = 0.0
        return

    link_idx_i = robot_link_indices[row]
    link_idx_j = robot_link_indices[col]
    joint_idx_i = target_joint_indices[row]
    joint_idx_j = target_joint_indices[col]

    # robot pair vector
    t_i = T_world_link[batch_idx, frame_idx, link_idx_i]
    t_j = T_world_link[batch_idx, frame_idx, link_idx_j]
    pos_robot_i = wp.vec3(t_i[0], t_i[1], t_i[2])
    pos_robot_j = wp.vec3(t_j[0], t_j[1], t_j[2])
    delta_robot = pos_robot_i - pos_robot_j

    # target pair vector
    tgt_idx = batch_idx // wp.max(1, T_world_link.shape[0] // target_keypoints.shape[0])
    pos_target_i = wp.vec3(
        target_keypoints[tgt_idx, frame_idx, joint_idx_i, 0],
        target_keypoints[tgt_idx, frame_idx, joint_idx_i, 1],
        target_keypoints[tgt_idx, frame_idx, joint_idx_i, 2],
    )
    pos_target_j = wp.vec3(
        target_keypoints[tgt_idx, frame_idx, joint_idx_j, 0],
        target_keypoints[tgt_idx, frame_idx, joint_idx_j, 1],
        target_keypoints[tgt_idx, frame_idx, joint_idx_j, 2],
    )
    delta_target = target_scales[wp.min(tgt_idx, target_scales.shape[0] - 1)] * (pos_target_i - pos_target_j)

    diff = delta_target - delta_robot
    out_residual[batch_idx, base_idx + 0] = diff[0] * position_weight
    out_residual[batch_idx, base_idx + 1] = diff[1] * position_weight
    out_residual[batch_idx, base_idx + 2] = diff[2] * position_weight

    eps = 1e-6
    len_robot = wp.length(delta_robot) + eps
    len_target = wp.length(delta_target) + eps
    dir_robot = delta_robot / len_robot
    dir_target = delta_target / len_target
    angle_err = 1.0 - wp.dot(dir_robot, dir_target)
    out_residual[batch_idx, base_idx + 3] = angle_err * angle_weight


@wp.kernel
def compute_trajectory_retargeting_jacobian_sparse_pattern_kernel(
    all_ancestor_dof_indices: wp.array1d(dtype=wp.int32),
    num_unique_dofs: int,
    single_tangent_dim: int,
    base_dim: int,
    num_pairs: int,
    row_offset: int,
    nnz_offset: int,
    row_indices: wp.array1d(dtype=wp.int32),
    col_indices: wp.array1d(dtype=wp.int32),
):
    """Build the sparse Jacobian pattern."""
    frame_idx, pair_idx = wp.tid()  # pyright: ignore[reportAssignmentType]

    # frame and pair row offset
    frame_residual_offset = frame_idx * num_pairs * 4
    pair_residual_offset = pair_idx * 4

    cols_per_residual = num_unique_dofs + base_dim
    nnz_per_pair = 4 * cols_per_residual

    base_nnz_idx = nnz_offset + (frame_idx * num_pairs + pair_idx) * nnz_per_pair

    # process three position rows and one direction row
    for r in range(4):
        res_idx = row_offset + frame_residual_offset + pair_residual_offset + r
        res_nnz_base = base_nnz_idx + r * cols_per_residual

        # joint columns
        for d in range(num_unique_dofs):
            dof_idx = all_ancestor_dof_indices[d]
            col_idx = frame_idx * single_tangent_dim + base_dim + dof_idx
            row_indices[res_nnz_base + d] = res_idx
            col_indices[res_nnz_base + d] = col_idx

        # floating-base columns
        for b in range(base_dim):
            col_idx = frame_idx * single_tangent_dim + b
            row_indices[res_nnz_base + num_unique_dofs + b] = res_idx
            col_indices[res_nnz_base + num_unique_dofs + b] = col_idx


@wp.kernel
def compute_trajectory_retargeting_jacobian_sparse_values_kernel(
    S_world: wp.array4d(dtype=wp.float32),  # [batch, frames, 6, num_joints]
    T_world_link: wp.array3d(dtype=wp_vec7),  # [batch, frames, num_links]
    T_world_base: wp.array2d(dtype=wp_vec7),  # [batch, frames]
    target_keypoints: wp.array4d(dtype=wp.float32),  # [batch, frames, num_keypoints, 3]
    robot_link_indices: wp.array1d(dtype=wp.int32),
    target_joint_indices: wp.array1d(dtype=wp.int32),
    pair_rows: wp.array1d(dtype=wp.int32),
    pair_cols: wp.array1d(dtype=wp.int32),
    all_ancestor_dof_indices: wp.array1d(dtype=wp.int32),
    link_ancestor_masks: wp.array2d(dtype=wp.bool),  # [num_selected, num_joints]
    joints_to_actuated: wp.array2d(dtype=wp.float32),  # [num_joints, num_actuated]
    num_unique_dofs: int,
    target_scales: wp.array1d(dtype=wp.float32),
    position_weight: float,
    angle_weight: float,
    frame_mask: wp.array2d(dtype=wp.uint8),
    use_mask: bool,
    base_dim: int,
    num_pairs: int,
    nnz_offset: int,
    values: wp.array2d(dtype=wp.float32),  # [batch, nnz]
):
    batch_idx, frame_idx, pair_idx = wp.tid()  # pyright: ignore[reportAssignmentType]

    row = pair_rows[pair_idx]
    col = pair_cols[pair_idx]

    cols_per_residual = num_unique_dofs + base_dim
    nnz_per_pair = 4 * cols_per_residual

    base_nnz_idx = nnz_offset + (frame_idx * num_pairs + pair_idx) * nnz_per_pair

    if row == col or (use_mask and frame_mask[batch_idx, frame_idx] == wp.uint8(0)):
        for i in range(nnz_per_pair):
            values[batch_idx, base_nnz_idx + i] = 0.0
        return

    link_idx_i = robot_link_indices[row]
    link_idx_j = robot_link_indices[col]
    joint_idx_i = target_joint_indices[row]
    joint_idx_j = target_joint_indices[col]

    t_i = T_world_link[batch_idx, frame_idx, link_idx_i]
    t_j = T_world_link[batch_idx, frame_idx, link_idx_j]
    p_i = wp.vec3(t_i[0], t_i[1], t_i[2])
    p_j = wp.vec3(t_j[0], t_j[1], t_j[2])
    delta_robot = p_i - p_j

    tgt_idx = batch_idx // wp.max(1, T_world_link.shape[0] // target_keypoints.shape[0])
    pos_target_i = wp.vec3(
        target_keypoints[tgt_idx, frame_idx, joint_idx_i, 0],
        target_keypoints[tgt_idx, frame_idx, joint_idx_i, 1],
        target_keypoints[tgt_idx, frame_idx, joint_idx_i, 2],
    )
    pos_target_j = wp.vec3(
        target_keypoints[tgt_idx, frame_idx, joint_idx_j, 0],
        target_keypoints[tgt_idx, frame_idx, joint_idx_j, 1],
        target_keypoints[tgt_idx, frame_idx, joint_idx_j, 2],
    )
    delta_target = target_scales[wp.min(tgt_idx, target_scales.shape[0] - 1)] * (pos_target_i - pos_target_j)

    eps = 1e-6
    len_robot = wp.length(delta_robot) + eps
    len_target = wp.length(delta_target) + eps
    dir_robot = delta_robot / len_robot
    dir_target = delta_target / len_target

    dot_val = wp.dot(dir_target, dir_robot)
    d_angle_d_delta = -(dir_target - dot_val * dir_robot) / len_robot

    # direction residual follows the three position rows
    ang_base_offset = 3 * cols_per_residual
    num_joints = joints_to_actuated.shape[0]

    for d in range(num_unique_dofs):
        actuated_idx = all_ancestor_dof_indices[d]
        d_delta_dq = wp.vec3(0.0, 0.0, 0.0)

        for joint_idx in range(num_joints):
            affects_i = link_ancestor_masks[row, joint_idx]
            affects_j = link_ancestor_masks[col, joint_idx]

            if not affects_i and not affects_j:
                continue

            weight_val = joints_to_actuated[joint_idx, actuated_idx]
            if weight_val == 0.0:
                continue

            v = wp.vec3(
                S_world[batch_idx, frame_idx, 0, joint_idx],
                S_world[batch_idx, frame_idx, 1, joint_idx],
                S_world[batch_idx, frame_idx, 2, joint_idx],
            )
            omega = wp.vec3(
                S_world[batch_idx, frame_idx, 3, joint_idx],
                S_world[batch_idx, frame_idx, 4, joint_idx],
                S_world[batch_idx, frame_idx, 5, joint_idx],
            )

            dp_i_dq = wp.vec3(0.0, 0.0, 0.0)
            dp_j_dq = wp.vec3(0.0, 0.0, 0.0)

            if affects_i:
                dp_i_dq = weight_val * (v + wp.cross(omega, p_i))
            if affects_j:
                dp_j_dq = weight_val * (v + wp.cross(omega, p_j))

            d_delta_dq = d_delta_dq + (dp_i_dq - dp_j_dq)

        angle_jac = wp.dot(d_angle_d_delta, d_delta_dq)

        values[batch_idx, base_nnz_idx + 0 * cols_per_residual + d] = -d_delta_dq[0] * position_weight
        values[batch_idx, base_nnz_idx + 1 * cols_per_residual + d] = -d_delta_dq[1] * position_weight
        values[batch_idx, base_nnz_idx + 2 * cols_per_residual + d] = -d_delta_dq[2] * position_weight
        values[batch_idx, base_nnz_idx + ang_base_offset + d] = angle_jac * angle_weight

    for b in range(base_dim):
        v = wp.vec3(0.0, 0.0, 0.0)
        omega = wp.vec3(0.0, 0.0, 0.0)

        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[b] = 1.0
        base_twist = se3_adjoint_func(T_world_base[batch_idx, frame_idx]) * unit_vec
        v = wp.vec3(base_twist[0], base_twist[1], base_twist[2])
        omega = wp.vec3(base_twist[3], base_twist[4], base_twist[5])

        dp_i_dq = v + wp.cross(omega, p_i)
        dp_j_dq = v + wp.cross(omega, p_j)
        d_delta_dq = dp_i_dq - dp_j_dq

        idx_offset = num_unique_dofs + b

        values[batch_idx, base_nnz_idx + 0 * cols_per_residual + idx_offset] = -d_delta_dq[0] * position_weight
        values[batch_idx, base_nnz_idx + 1 * cols_per_residual + idx_offset] = -d_delta_dq[1] * position_weight
        values[batch_idx, base_nnz_idx + 2 * cols_per_residual + idx_offset] = -d_delta_dq[2] * position_weight

        angle_jac = wp.dot(d_angle_d_delta, d_delta_dq)
        values[batch_idx, base_nnz_idx + ang_base_offset + idx_offset] = angle_jac * angle_weight


if TYPE_CHECKING:
    from robokit.opt.var_values import VarValues
    from robokit.robo import Robot
