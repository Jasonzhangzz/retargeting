# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportUndefinedVariable=false
# pyright: reportCallIssue=false
"""Sparse trajectory position and contact residuals."""

from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.lie.se3_kernels import se3_adjoint_func
from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import Context, SparseTask, SparsityPattern
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


# --- position task ----------------------------------------------------------
class TrajectoryPositionTask(RobotTask, SparseTask):
    """Align robot-link positions to optionally indexed world-frame points."""

    num_links = 1  # a template counts as active until `resolve` sizes it

    def __init__(
        self,
        robot: Optional["Robot"] = None,  # noqa: F821
        frame_index: Optional[Union[int, Sequence[int], Context]] = None,
        target_positions: Optional[Union[np.ndarray, wp.array, Context]] = None,  # [batch, frames, points, 3]
        weight: Union[float, Sequence[float], np.ndarray, Mapping[str, float]] = 1.0,  # scalar, per link, or by name
        target_point_indices: Optional[Union[Sequence[int], Context]] = None,
        name: str = "",
    ):
        self.name = name
        self.robot = robot
        self.weight = weight
        self.frame_index_arg, self.target_positions_arg = frame_index, target_positions
        self.target_point_indices_arg = target_point_indices
        if robot is None or frame_index is None or target_positions is None:
            return  # a template; `resolve` completes it
        self.frame_indices_list = [frame_index] if isinstance(frame_index, int) else list(frame_index)
        self.num_links = len(self.frame_indices_list)
        self._uses_target_point_indices = target_point_indices is not None
        self._target_point_indices_np = (
            np.arange(self.num_links, dtype=np.int32)
            if target_point_indices is None
            else np.atleast_1d(np.asarray(target_point_indices, dtype=np.int32))
        )
        num_target_point_indices = self._target_point_indices_np.size
        if self._target_point_indices_np.ndim != 1 or num_target_point_indices != self.num_links:
            raise ValueError(f"expected {self.num_links} target point indices, got {num_target_point_indices}")
        if np.any(self._target_point_indices_np < 0):
            raise ValueError("target_point_indices must be nonnegative")
        self._weight_np = (
            np.full(self.num_links, weight, dtype=np.float32)
            if isinstance(weight, (int, float))
            else np.asarray(weight, dtype=np.float32)
        )
        if len(self._weight_np) != self.num_links:
            raise ValueError(f"expected {self.num_links} weights, got {len(self._weight_np)}")
        self.weights: Optional[wp.array] = None

        self.target_positions_input = target_positions

        self.device: Optional[wp_device_type] = None
        self.target_positions: Optional[wp.array] = None
        self.target_point_indices: Optional[wp.array] = None
        self.frame_indices: Optional[wp.array] = None
        self.ancestor_dof_indices: Optional[wp.array] = None
        self.dof_offsets: Optional[wp.array] = None
        self.nnz_starts: Optional[wp.array] = None
        self._dof_counts_np: np.ndarray = np.zeros(0, dtype=np.int32)
        self._nnz_total: int = 0
        self._nnz_starts_base_dim: int = -1
        self.num_frames: int = 0

    @property
    def active(self) -> bool:
        weight = list(self.weight.values()) if isinstance(self.weight, Mapping) else self.weight
        return bool(np.any(weight)) and self.num_links > 0

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Fill a template with helper links and targets, reading a Warp target directly; unlisted links weigh zero."""
        super().resolve(var_values, context)
        links = np.atleast_1d(np.asarray(self.frame_index_arg, dtype=np.int32))
        weight = self.weight
        if isinstance(weight, Mapping):
            weight = [weight.get(self.robot.link_names[link], 0.0) for link in links]
        TrajectoryPositionTask.__init__(
            self,
            self.robot,
            links.tolist(),
            self.target_positions_arg,
            weight,
            self.target_point_indices_arg,
            self.name,
        )
        self.init_buffers(var_values.get(self.var_key).device)

    def init_buffers(self, device: wp_device_type):
        self.device = device

        target_positions = self.target_positions
        if target_positions is None:
            if isinstance(self.target_positions_input, np.ndarray):
                target_np = self.target_positions_input
                if target_np.ndim == 2:
                    target_np = target_np[None, ...]
                target_positions = wp.from_numpy(target_np.astype(np.float32), dtype=wp.float32, device=device)
            else:
                target_positions = self.target_positions_input.to(device)
        if target_positions.ndim == 3:
            if self.num_links != 1:
                raise ValueError(f"targets must be [batch, frames, {self.num_links}, 3] for {self.num_links} links")
            target_positions = target_positions.reshape((target_positions.shape[0], target_positions.shape[1], 1, 3))
        if target_positions.ndim != 4 or target_positions.shape[3] != 3:
            raise ValueError("targets must have shape [batch, frames, points, 3]")
        if not self._uses_target_point_indices and target_positions.shape[2] != self.num_links:
            raise ValueError(f"targets have {target_positions.shape[2]} links, expected {self.num_links}")
        if self._target_point_indices_np.size and self._target_point_indices_np.max() >= target_positions.shape[2]:
            raise ValueError(f"target_point_indices must be less than {target_positions.shape[2]}")
        self.target_positions = target_positions

        self.num_frames = self.target_positions.shape[1]

        # store each link's active ancestor DOFs in CSR form
        joints_to_actuated = self.robot.spec.joints_to_actuated_mapping
        per_link_dofs = []
        for link_index in self.frame_indices_list:
            ancestor_indices = np.where(self.robot.spec.link_ancestor_joints_mask[link_index])[0]
            unique_dofs = set()
            for joint_idx in ancestor_indices:
                for actuated_idx in range(joints_to_actuated.shape[1]):
                    if joints_to_actuated[joint_idx, actuated_idx] != 0.0:
                        unique_dofs.add(actuated_idx)
            per_link_dofs.append(np.array(sorted(unique_dofs), dtype=np.int32))

        dof_counts = np.array([len(dofs) for dofs in per_link_dofs], dtype=np.int32)
        self._dof_counts_np = dof_counts
        self.dof_offsets = wp.from_numpy(
            np.concatenate([[0], np.cumsum(dof_counts)]).astype(np.int32), dtype=wp.int32, device=device
        )
        self.ancestor_dof_indices = wp.from_numpy(
            np.concatenate(per_link_dofs).astype(np.int32) if per_link_dofs else np.zeros(0, np.int32),
            dtype=wp.int32,
            device=device,
        )
        self.frame_indices = wp.from_numpy(
            np.array(self.frame_indices_list, dtype=np.int32), dtype=wp.int32, device=device
        )
        self.target_point_indices = wp.from_numpy(self._target_point_indices_np, dtype=wp.int32, device=device)
        self.weights = wp.from_numpy(self._weight_np, dtype=wp.float32, device=device)

    def _nnz_layout(self, base_dim: int, device: wp_device_type) -> int:
        """Cache ragged per-link nonzero offsets and return the total count."""
        if self._nnz_starts_base_dim != base_dim or self.nnz_starts is None:
            nnz_per_frame = 3 * (self._dof_counts_np + base_dim)
            starts = self.num_frames * np.concatenate([[0], np.cumsum(nnz_per_frame)[:-1]])
            self.nnz_starts = wp.from_numpy(starts.astype(np.int32), dtype=wp.int32, device=device)
            self._nnz_total = int(self.num_frames * nnz_per_frame.sum())
            self._nnz_starts_base_dim = base_dim
        return self._nnz_total

    @property
    def residual_dim(self) -> int:
        if self.num_frames == 0 and self.target_positions_input is not None:
            shape = self.target_positions_input.shape
            self.num_frames = shape[1] if len(shape) >= 3 else shape[0]
        return self.num_links * self.num_frames * 3

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

        wp.launch(
            kernel=compute_trajectory_position_residual_kernel,
            dim=(robot_state.q.shape[0], self.num_frames, self.num_links),
            inputs=[
                robot_state.T_world_link,
                self.frame_indices,
                self.target_positions,
                self.target_point_indices,
                self.weights,
                row_offset,
                self.use_mask,
                self._get_frame_mask(),
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
        nnz = self._nnz_layout(base_dim, kernel_device)

        row_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_row_indices is None else out_row_indices
        )
        col_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_col_indices is None else out_col_indices
        )

        wp.launch(
            kernel=compute_trajectory_position_jacobian_sparse_pattern_kernel,
            dim=(self.num_frames, self.num_links),
            inputs=[
                self.ancestor_dof_indices,
                self.dof_offsets,
                self.nnz_starts,
                self.num_frames,
                single_tangent_dim,
                base_dim,
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
        nnz = self._nnz_layout(base_dim, kernel_device)

        if out_jacobian_values is None:
            out_jacobian_values = wp.empty((robot_state.q.shape[0], nnz), dtype=wp.float32, device=kernel_device)

        spec_tensors = robot_state.spec_tensors

        wp.launch(
            kernel=compute_trajectory_position_jacobian_sparse_values_kernel,
            dim=(robot_state.q.shape[0], self.num_frames, self.num_links),
            inputs=[
                robot_state.S_world,
                robot_state.T_world_link,
                robot_state.T_world_base,
                self.frame_indices,
                self.ancestor_dof_indices,
                self.dof_offsets,
                self.nnz_starts,
                spec_tensors.link_ancestor_joints_mask,
                spec_tensors.joints_to_actuated_mapping,
                self.weights,
                base_dim,
                nnz_offset,
                self.use_mask,
                self._get_frame_mask(),
            ],
            outputs=[out_jacobian_values],
            device=kernel_device,
        )

        return out_jacobian_values


# --- position kernels -------------------------------------------------------
@wp.kernel
def compute_trajectory_position_residual_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),  # [batch, frames, num_links]
    frame_indices: wp.array1d(dtype=wp.int32),  # [links]
    target_positions: wp.array4d(dtype=wp.float32),  # [batch, frames, points, 3]
    target_point_indices: wp.array1d(dtype=wp.int32),  # [links]
    weights: wp.array1d(dtype=wp.float32),  # [links]
    row_offset: int,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    out_residual: wp.array2d(dtype=wp.float32),  # [batch, residual_dim]
):
    batch_idx, frame_idx, slot = wp.tid()  # pyright: ignore[reportAssignmentType]

    T_world_link_i = T_world_link[batch_idx, frame_idx, frame_indices[slot]]
    link_pos = wp.vec3(T_world_link_i[0], T_world_link_i[1], T_world_link_i[2])
    target_point_index = target_point_indices[slot]

    tgt_idx = batch_idx // wp.max(1, T_world_link.shape[0] // target_positions.shape[0])
    target_pos_vec = wp.vec3(
        target_positions[tgt_idx, frame_idx, target_point_index, 0],
        target_positions[tgt_idx, frame_idx, target_point_index, 1],
        target_positions[tgt_idx, frame_idx, target_point_index, 2],
    )

    diff = link_pos - target_pos_vec
    mask = wp.float32(1.0)
    if use_mask:
        mask = wp.float32(frame_mask[batch_idx, frame_idx])
    w = weights[slot] * mask

    res_base_idx = (slot * target_positions.shape[1] + frame_idx) * 3

    out_residual[batch_idx, row_offset + res_base_idx + 0] = diff[0] * w
    out_residual[batch_idx, row_offset + res_base_idx + 1] = diff[1] * w
    out_residual[batch_idx, row_offset + res_base_idx + 2] = diff[2] * w


@wp.kernel
def compute_trajectory_position_jacobian_sparse_pattern_kernel(
    ancestor_dof_indices: wp.array1d(dtype=wp.int32),  # concatenated per-link dof lists
    dof_offsets: wp.array1d(dtype=wp.int32),  # [links + 1] where each link's dofs begin
    nnz_starts: wp.array1d(dtype=wp.int32),  # [links] where each link's nnz block begins
    num_frames: int,
    single_tangent_dim: int,
    base_dim: int,
    row_offset: int,
    nnz_offset: int,
    row_indices: wp.array1d(dtype=wp.int32),
    col_indices: wp.array1d(dtype=wp.int32),
):
    frame_idx, slot = wp.tid()  # pyright: ignore[reportAssignmentType]

    dof_begin = dof_offsets[slot]
    num_unique_dofs = dof_offsets[slot + 1] - dof_begin
    cols_per_residual = num_unique_dofs + base_dim
    nnz_per_frame = 3 * cols_per_residual

    base_nnz_idx = nnz_offset + nnz_starts[slot] + frame_idx * nnz_per_frame

    # one row per position component
    for r in range(3):
        res_idx = row_offset + (slot * num_frames + frame_idx) * 3 + r
        res_nnz_base = base_nnz_idx + r * cols_per_residual

        # joint columns
        for d in range(num_unique_dofs):
            dof_idx = ancestor_dof_indices[dof_begin + d]
            col_idx = frame_idx * single_tangent_dim + base_dim + dof_idx
            row_indices[res_nnz_base + d] = res_idx
            col_indices[res_nnz_base + d] = col_idx

        # floating-base columns
        for b in range(base_dim):
            col_idx = frame_idx * single_tangent_dim + b
            row_indices[res_nnz_base + num_unique_dofs + b] = res_idx
            col_indices[res_nnz_base + num_unique_dofs + b] = col_idx


@wp.kernel
def compute_trajectory_position_jacobian_sparse_values_kernel(
    S_world: wp.array4d(dtype=wp.float32),  # [batch, frames, 6, num_joints]
    T_world_link: wp.array3d(dtype=wp_vec7),  # [batch, frames, num_links]
    T_world_base: wp.array2d(dtype=wp_vec7),  # [batch, frames]
    frame_indices: wp.array1d(dtype=wp.int32),  # [links]
    ancestor_dof_indices: wp.array1d(dtype=wp.int32),  # concatenated per-link dof lists
    dof_offsets: wp.array1d(dtype=wp.int32),  # [links + 1]
    nnz_starts: wp.array1d(dtype=wp.int32),  # [links]
    link_ancestor_mask: wp.array2d(dtype=wp.bool),  # [num_links, num_joints]
    joints_to_actuated: wp.array2d(dtype=wp.float32),  # [num_joints, num_actuated]
    weights: wp.array1d(dtype=wp.float32),  # [links]
    base_dim: int,
    nnz_offset: int,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    values: wp.array2d(dtype=wp.float32),  # [batch, nnz]
):
    batch_idx, frame_idx, slot = wp.tid()  # pyright: ignore[reportAssignmentType]

    mask = wp.float32(1.0)
    if use_mask:
        mask = wp.float32(frame_mask[batch_idx, frame_idx])
    w = weights[slot] * mask
    frame_index = frame_indices[slot]
    dof_begin = dof_offsets[slot]
    num_unique_dofs = dof_offsets[slot + 1] - dof_begin
    cols_per_residual = num_unique_dofs + base_dim
    nnz_per_frame = 3 * cols_per_residual
    base_nnz_idx = nnz_offset + nnz_starts[slot] + frame_idx * nnz_per_frame

    T_world_link_i = T_world_link[batch_idx, frame_idx, frame_index]
    p_link = wp.vec3(T_world_link_i[0], T_world_link_i[1], T_world_link_i[2])

    num_joints = joints_to_actuated.shape[0]

    for d in range(num_unique_dofs):
        actuated_idx = ancestor_dof_indices[dof_begin + d]
        dp_dq = wp.vec3(0.0, 0.0, 0.0)

        for joint_idx in range(num_joints):
            if not link_ancestor_mask[frame_index, joint_idx]:
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
            dp_dq = dp_dq + weight_val * (v + wp.cross(omega, p_link))

        values[batch_idx, base_nnz_idx + 0 * cols_per_residual + d] = dp_dq[0] * w
        values[batch_idx, base_nnz_idx + 1 * cols_per_residual + d] = dp_dq[1] * w
        values[batch_idx, base_nnz_idx + 2 * cols_per_residual + d] = dp_dq[2] * w

    for b in range(base_dim):
        v = wp.vec3(0.0, 0.0, 0.0)
        omega = wp.vec3(0.0, 0.0, 0.0)

        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[b] = 1.0
        base_twist = se3_adjoint_func(T_world_base[batch_idx, frame_idx]) * unit_vec
        v = wp.vec3(base_twist[0], base_twist[1], base_twist[2])
        omega = wp.vec3(base_twist[3], base_twist[4], base_twist[5])

        dp_dq = v + wp.cross(omega, p_link)

        idx_offset = num_unique_dofs + b
        values[batch_idx, base_nnz_idx + 0 * cols_per_residual + idx_offset] = dp_dq[0] * w
        values[batch_idx, base_nnz_idx + 1 * cols_per_residual + idx_offset] = dp_dq[1] * w
        values[batch_idx, base_nnz_idx + 2 * cols_per_residual + idx_offset] = dp_dq[2] * w


# --- contact kernels --------------------------------------------------------
@wp.func
def _segment_target_func(contact_pt: wp.vec3, axis: wp.vec3, point: wp.vec3) -> wp.vec4:
    """The point of the segment `contact_pt + t axis`, `t` in [0, 1], nearest `point`, and `t`."""
    length_sq = wp.dot(axis, axis)
    t = wp.float32(0.0)
    if length_sq > wp.float32(0.0):
        t = wp.clamp(wp.dot(point - contact_pt, axis) / length_sq, wp.float32(0.0), wp.float32(1.0))
    target = contact_pt + t * axis
    return wp.vec4(target[0], target[1], target[2], t)


@wp.kernel
def _compute_trajectory_contact_residual_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),  # [batch, frames, num_links]
    contact_points: wp.array4d(dtype=wp.float32),  # [batch, frames, max_contacts, 3]
    contact_axes: wp.array4d(dtype=wp.float32),  # [batch, frames, max_contacts, 3]
    local_contact_points: wp.array4d(dtype=wp.float32),  # [batch, frames, max_contacts, 3]
    contact_link_indices: wp.array2d(dtype=wp.int32),  # [frames, max_contacts]
    contact_mask: wp.array3d(dtype=wp.bool),  # [batch, frames, max_contacts]
    max_contacts: int,
    contact_margin: float,
    weight: float,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),  # [batch, residual_dim]
):
    """Compute elementwise contact residuals outside the deadband."""
    batch_idx, frame_idx, contact_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    tgt_idx = batch_idx // wp.max(1, T_world_link.shape[0] // contact_mask.shape[0])

    base_res_idx = row_offset + (frame_idx * max_contacts + contact_idx) * 3

    if not contact_mask[tgt_idx, frame_idx, contact_idx]:
        out_residual[batch_idx, base_res_idx + 0] = wp.float32(0.0)
        out_residual[batch_idx, base_res_idx + 1] = wp.float32(0.0)
        out_residual[batch_idx, base_res_idx + 2] = wp.float32(0.0)
        return

    link_idx = contact_link_indices[frame_idx, contact_idx]
    T_world_link_i = T_world_link[batch_idx, frame_idx, link_idx]
    link_pos = wp.vec3(T_world_link_i[0], T_world_link_i[1], T_world_link_i[2])
    q_wxyz_world_link = wp.quat(T_world_link_i[4], T_world_link_i[5], T_world_link_i[6], T_world_link_i[3])
    local_point = wp.vec3(
        local_contact_points[tgt_idx, frame_idx, contact_idx, 0],
        local_contact_points[tgt_idx, frame_idx, contact_idx, 1],
        local_contact_points[tgt_idx, frame_idx, contact_idx, 2],
    )
    link_point = link_pos + wp.quat_rotate(q_wxyz_world_link, local_point)

    contact_pt = wp.vec3(
        contact_points[tgt_idx, frame_idx, contact_idx, 0],
        contact_points[tgt_idx, frame_idx, contact_idx, 1],
        contact_points[tgt_idx, frame_idx, contact_idx, 2],
    )
    axis = wp.vec3(
        contact_axes[tgt_idx, frame_idx, contact_idx, 0],
        contact_axes[tgt_idx, frame_idx, contact_idx, 1],
        contact_axes[tgt_idx, frame_idx, contact_idx, 2],
    )
    target = _segment_target_func(contact_pt, axis, link_point)

    raw_diff = wp.vec3(target[0], target[1], target[2]) - link_point
    margin = wp.float32(contact_margin)

    p0 = wp.max(wp.abs(raw_diff[0]) - margin, wp.float32(0.0))
    p1 = wp.max(wp.abs(raw_diff[1]) - margin, wp.float32(0.0))
    p2 = wp.max(wp.abs(raw_diff[2]) - margin, wp.float32(0.0))

    w = wp.float32(weight)
    out_residual[batch_idx, base_res_idx + 0] = p0 * w
    out_residual[batch_idx, base_res_idx + 1] = p1 * w
    out_residual[batch_idx, base_res_idx + 2] = p2 * w


@wp.kernel
def _compute_trajectory_contact_jacobian_sparse_pattern_kernel(
    contact_link_indices: wp.array2d(dtype=wp.int32),  # [frames, max_contacts]
    link_ancestor_dofs: wp.array2d(dtype=wp.int32),  # [num_links, max_ancestor_dofs]
    link_num_ancestor_dofs: wp.array1d(dtype=wp.int32),  # [num_links]
    max_ancestor_dofs: int,
    max_contacts: int,
    single_tangent_dim: int,
    base_dim: int,
    row_offset: int,
    nnz_offset: int,
    row_indices: wp.array1d(dtype=wp.int32),
    col_indices: wp.array1d(dtype=wp.int32),
):
    frame_idx, contact_idx = wp.tid()  # pyright: ignore[reportAssignmentType]

    cols_per_residual = max_ancestor_dofs + base_dim
    nnz_per_contact = 3 * cols_per_residual
    base_nnz = nnz_offset + (frame_idx * max_contacts + contact_idx) * nnz_per_contact
    dummy_col = frame_idx * single_tangent_dim + base_dim

    for xyz in range(3):
        res_row = row_offset + (frame_idx * max_contacts + contact_idx) * 3 + xyz
        nnz_xyz_base = base_nnz + xyz * cols_per_residual

        link_idx = contact_link_indices[frame_idx, contact_idx]
        num_ancestor = link_num_ancestor_dofs[link_idx]
        for d in range(max_ancestor_dofs):
            col = dummy_col
            if d < num_ancestor:
                col = frame_idx * single_tangent_dim + base_dim + link_ancestor_dofs[link_idx, d]
            row_indices[nnz_xyz_base + d] = res_row
            col_indices[nnz_xyz_base + d] = col

        for b in range(base_dim):
            row_indices[nnz_xyz_base + max_ancestor_dofs + b] = res_row
            col_indices[nnz_xyz_base + max_ancestor_dofs + b] = frame_idx * single_tangent_dim + b


@wp.kernel
def _compute_trajectory_contact_jacobian_sparse_values_kernel(
    S_world: wp.array4d(dtype=wp.float32),  # [batch, frames, 6, num_joints]
    T_world_link: wp.array3d(dtype=wp_vec7),  # [batch, frames, num_links]
    T_world_base: wp.array2d(dtype=wp_vec7),  # [batch, frames]
    contact_points: wp.array4d(dtype=wp.float32),  # [batch, frames, max_contacts, 3]
    contact_axes: wp.array4d(dtype=wp.float32),  # [batch, frames, max_contacts, 3]
    local_contact_points: wp.array4d(dtype=wp.float32),  # [batch, frames, max_contacts, 3]
    contact_link_indices: wp.array2d(dtype=wp.int32),  # [frames, max_contacts]
    contact_mask: wp.array3d(dtype=wp.bool),  # [batch, frames, max_contacts]
    link_ancestor_dofs: wp.array2d(dtype=wp.int32),  # [num_links, max_ancestor_dofs]
    link_num_ancestor_dofs: wp.array1d(dtype=wp.int32),  # [num_links]
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),  # [num_links, num_joints]
    joints_to_actuated: wp.array2d(dtype=wp.float32),  # [num_joints, num_actuated]
    max_ancestor_dofs: int,
    max_contacts: int,
    contact_margin: float,
    weight: float,
    base_dim: int,
    nnz_offset: int,
    values: wp.array2d(dtype=wp.float32),  # [batch, nnz]
):
    batch_idx, frame_idx, contact_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    tgt_idx = batch_idx // wp.max(1, T_world_link.shape[0] // contact_mask.shape[0])

    cols_per_residual = max_ancestor_dofs + base_dim
    nnz_per_contact = 3 * cols_per_residual
    base_nnz = nnz_offset + (frame_idx * max_contacts + contact_idx) * nnz_per_contact

    if not contact_mask[tgt_idx, frame_idx, contact_idx]:
        for d in range(3 * cols_per_residual):
            values[batch_idx, base_nnz + d] = wp.float32(0.0)
        return

    link_idx = contact_link_indices[frame_idx, contact_idx]
    link_pose = T_world_link[batch_idx, frame_idx, link_idx]
    p_link = wp.vec3(link_pose[0], link_pose[1], link_pose[2])
    q_wxyz_world_link = wp.quat(link_pose[4], link_pose[5], link_pose[6], link_pose[3])
    local_point = wp.vec3(
        local_contact_points[tgt_idx, frame_idx, contact_idx, 0],
        local_contact_points[tgt_idx, frame_idx, contact_idx, 1],
        local_contact_points[tgt_idx, frame_idx, contact_idx, 2],
    )
    p_contact = p_link + wp.quat_rotate(q_wxyz_world_link, local_point)

    contact_pt = wp.vec3(
        contact_points[tgt_idx, frame_idx, contact_idx, 0],
        contact_points[tgt_idx, frame_idx, contact_idx, 1],
        contact_points[tgt_idx, frame_idx, contact_idx, 2],
    )
    axis = wp.vec3(
        contact_axes[tgt_idx, frame_idx, contact_idx, 0],
        contact_axes[tgt_idx, frame_idx, contact_idx, 1],
        contact_axes[tgt_idx, frame_idx, contact_idx, 2],
    )
    target = _segment_target_func(contact_pt, axis, p_contact)
    raw_diff = wp.vec3(target[0], target[1], target[2]) - p_contact
    margin = wp.float32(contact_margin)
    # inside the segment the target slides with the point: only motion across the axis counts
    slide = wp.vec3(0.0, 0.0, 0.0)
    if target[3] > wp.float32(0.0) and target[3] < wp.float32(1.0):
        slide = wp.normalize(axis)

    # activate components outside the deadband
    act_x = wp.float32(0.0)
    act_y = wp.float32(0.0)
    act_z = wp.float32(0.0)
    if wp.abs(raw_diff[0]) > margin:
        act_x = wp.sign(raw_diff[0])
    if wp.abs(raw_diff[1]) > margin:
        act_y = wp.sign(raw_diff[1])
    if wp.abs(raw_diff[2]) > margin:
        act_z = wp.sign(raw_diff[2])

    num_joints = joints_to_actuated.shape[0]
    num_ancestor = link_num_ancestor_dofs[link_idx]
    w = wp.float32(weight)

    # joint columns
    for d in range(max_ancestor_dofs):
        val_x = wp.float32(0.0)
        val_y = wp.float32(0.0)
        val_z = wp.float32(0.0)

        if d < num_ancestor:
            actuated_idx = link_ancestor_dofs[link_idx, d]
            dp_dq = wp.vec3(0.0, 0.0, 0.0)

            for joint_idx in range(num_joints):
                if not link_ancestor_joints_mask[link_idx, joint_idx]:
                    continue
                w_jt = joints_to_actuated[joint_idx, actuated_idx]
                if w_jt == wp.float32(0.0):
                    continue
                v = wp.vec3(
                    S_world[batch_idx, frame_idx, 0, joint_idx],
                    S_world[batch_idx, frame_idx, 1, joint_idx],
                    S_world[batch_idx, frame_idx, 2, joint_idx],
                )
                om = wp.vec3(
                    S_world[batch_idx, frame_idx, 3, joint_idx],
                    S_world[batch_idx, frame_idx, 4, joint_idx],
                    S_world[batch_idx, frame_idx, 5, joint_idx],
                )
                dp_dq = dp_dq + w_jt * (v + wp.cross(om, p_contact))
            dp_dq = dp_dq - wp.dot(dp_dq, slide) * slide

            # differentiate the active residual components
            val_x = -act_x * w * dp_dq[0]
            val_y = -act_y * w * dp_dq[1]
            val_z = -act_z * w * dp_dq[2]

        values[batch_idx, base_nnz + 0 * cols_per_residual + d] = val_x
        values[batch_idx, base_nnz + 1 * cols_per_residual + d] = val_y
        values[batch_idx, base_nnz + 2 * cols_per_residual + d] = val_z

    # floating-base columns
    for b in range(base_dim):
        unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit_vec[b] = wp.float32(1.0)
        base_twist = se3_adjoint_func(T_world_base[batch_idx, frame_idx]) * unit_vec
        v = wp.vec3(base_twist[0], base_twist[1], base_twist[2])
        om = wp.vec3(base_twist[3], base_twist[4], base_twist[5])
        dp_dq = v + wp.cross(om, p_contact)
        dp_dq = dp_dq - wp.dot(dp_dq, slide) * slide

        idx = max_ancestor_dofs + b
        values[batch_idx, base_nnz + 0 * cols_per_residual + idx] = -act_x * w * dp_dq[0]
        values[batch_idx, base_nnz + 1 * cols_per_residual + idx] = -act_y * w * dp_dq[1]
        values[batch_idx, base_nnz + 2 * cols_per_residual + idx] = -act_z * w * dp_dq[2]


# --- contact task -----------------------------------------------------------
class TrajectoryContactTask(RobotTask, SparseTask):
    """Keep selected link points within a deadband of trajectory contacts: points, or segments along `contact_axes`.

    Args:
        robot: Robot with FK support.
        contact_points: Shared `[T, C, 3]` or batched `[B, T, C, 3]` world targets.
        local_contact_points: Optional link-local points. Defaults to each link origin.
        contact_link_indices: Padded `[T, C]` robot link indices.
        contact_mask: Shared `[T, C]` or batched `[B, T, C]` validity mask.
        contact_margin: Deadband margin; no penalty when distance is within this threshold.
        weight: Residual weight.
    """

    max_contacts = 1  # a template counts as active until `resolve` sizes it

    def __init__(
        self,
        robot: Optional["Robot"] = None,
        contact_points: Optional[Union[np.ndarray, wp.array, Context]] = None,
        contact_link_indices: Optional[Union[np.ndarray, Sequence[int], Context]] = None,
        contact_mask: Optional[Union[np.ndarray, wp.array, Context]] = None,
        contact_margin: float = 0.01,
        weight: float = 1.0,
        local_contact_points: Optional[Union[np.ndarray, wp.array, Context]] = None,
        name: str = "",
    ):
        self.name = name
        self.robot = robot
        self.contact_margin = contact_margin
        self.weight = weight
        self.contact_points_arg, self.contact_link_indices_arg = contact_points, contact_link_indices
        self.contact_mask_arg, self.local_contact_points_arg = contact_mask, local_contact_points
        if robot is None or contact_points is None or contact_link_indices is None or contact_mask is None:
            return

        contact_points_np = np.asarray(contact_points, dtype=np.float32)
        if contact_points_np.ndim == 3:
            contact_points_np = contact_points_np[None]
        if contact_points_np.ndim != 4:
            raise ValueError("contact_points must have shape [T, C, 3] or [B, T, C, 3].")
        contact_batch = int(contact_points_np.shape[0])
        self.num_frames = int(contact_points_np.shape[1])
        self.max_contacts = int(contact_points_np.shape[2])
        expected_points_shape = (contact_batch, self.num_frames, self.max_contacts, 3)
        if contact_points_np.shape != expected_points_shape:
            raise ValueError(f"contact_points must have shape {expected_points_shape}.")

        if local_contact_points is None:
            local_contact_points_np = np.zeros(expected_points_shape, dtype=np.float32)
        else:
            local_contact_points_np = np.asarray(local_contact_points, dtype=np.float32)
            if local_contact_points_np.ndim == 3:
                local_contact_points_np = np.broadcast_to(local_contact_points_np[None], expected_points_shape).copy()
            if local_contact_points_np.shape != expected_points_shape:
                raise ValueError(
                    f"local_contact_points must have shape {expected_points_shape} or {expected_points_shape[1:]}."
                )

        contact_link_indices_np = np.asarray(contact_link_indices, dtype=np.int32)
        if contact_link_indices_np.ndim == 1:
            contact_link_indices_np = np.broadcast_to(
                contact_link_indices_np[None], (self.num_frames, self.max_contacts)
            ).copy()
        if contact_link_indices_np.shape != (self.num_frames, self.max_contacts):
            raise ValueError(f"contact_link_indices must have shape {(self.num_frames, self.max_contacts)}.")

        contact_mask_np = np.asarray(contact_mask, dtype=np.bool_)
        if contact_mask_np.ndim == 2:
            contact_mask_np = np.broadcast_to(contact_mask_np[None], (contact_batch,) + contact_mask_np.shape).copy()
        expected_mask_shape = (contact_batch, self.num_frames, self.max_contacts)
        if contact_mask_np.shape != expected_mask_shape:
            raise ValueError(f"contact_mask must have shape {expected_mask_shape} or {expected_mask_shape[1:]}.")

        self.contact_points_input = contact_points_np
        self.local_contact_points_input = local_contact_points_np
        self.contact_link_indices_input = contact_link_indices_np
        self.contact_mask_input = contact_mask_np

        self.device: Optional[wp_device_type] = None
        self.contact_points_wp: Optional[wp.array] = None
        self.local_contact_points_wp: Optional[wp.array] = None
        self.contact_axes_wp: Optional[wp.array] = None
        self.contact_link_indices_wp: Optional[wp.array] = None
        self.contact_mask_wp: Optional[wp.array] = None
        self.link_ancestor_dofs_wp: Optional[wp.array] = None
        self.link_num_ancestor_dofs_wp: Optional[wp.array] = None
        self.max_ancestor_dofs: int = 0

    @property
    def active(self) -> bool:
        return bool(np.any(self.weight)) and self.max_contacts > 0

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Build from the helper's contact links; its Warp contact buffers are read directly."""
        super().resolve(var_values, context)
        points, mask, local = self.contact_points_arg, self.contact_mask_arg, self.local_contact_points_arg
        TrajectoryContactTask.__init__(
            self,
            self.robot,
            np.zeros(points.shape, np.float32),
            self.contact_link_indices_arg,
            np.zeros(mask.shape, np.bool_),
            self.contact_margin,
            self.weight,
            name=self.name,
        )
        self._init_buffers(points.device)
        self.contact_points_wp, self.contact_mask_wp = points, mask
        if local is not None:
            self.local_contact_points_wp = local

    def _init_buffers(self, device: wp_device_type):
        self.device = device
        self.contact_points_wp = wp.from_numpy(self.contact_points_input, dtype=wp.float32, device=device)
        self.local_contact_points_wp = wp.from_numpy(self.local_contact_points_input, dtype=wp.float32, device=device)
        self.contact_axes_wp = wp.zeros(self.contact_points_input.shape, dtype=wp.float32, device=device)
        self.contact_link_indices_wp = wp.from_numpy(self.contact_link_indices_input, dtype=wp.int32, device=device)
        self.contact_mask_wp = wp.from_numpy(self.contact_mask_input, dtype=wp.bool, device=device)

        robot_spec = self.robot.spec
        num_links = robot_spec.num_links
        joints_to_actuated = robot_spec.joints_to_actuated_mapping
        contact_links = np.unique(self.contact_link_indices_input)
        link_dofs = {
            int(link): np.flatnonzero(
                np.any(joints_to_actuated[robot_spec.link_ancestor_joints_mask[link]] != 0.0, axis=0)
            )
            for link in contact_links
        }
        self.max_ancestor_dofs = max((len(dofs) for dofs in link_dofs.values()), default=1)

        link_ancestor_dofs_np = np.zeros((num_links, self.max_ancestor_dofs), dtype=np.int32)
        link_num_ancestor_dofs_np = np.zeros(num_links, dtype=np.int32)
        for link, dofs in link_dofs.items():
            link_ancestor_dofs_np[link, : len(dofs)] = dofs
            link_num_ancestor_dofs_np[link] = len(dofs)

        self.link_ancestor_dofs_wp = wp.from_numpy(link_ancestor_dofs_np, dtype=wp.int32, device=device)
        self.link_num_ancestor_dofs_wp = wp.from_numpy(link_num_ancestor_dofs_np, dtype=wp.int32, device=device)

    def set_contacts(
        self,
        contact_points: wp.array,
        contact_mask: wp.array,
        local_contact_points: Optional[wp.array] = None,
        contact_axes: Optional[wp.array] = None,
    ):
        """Copy batched runtime contact data into graph-stable device buffers; omitted points and axes are zero. A
        nonzero axis makes its contact the segment from the point along it: the link point may sit anywhere on it, e.g.
        a straight move at its own pace."""
        if self.device is None:
            self._init_buffers(contact_points.device)
        if contact_points.ndim == 3:
            contact_points = contact_points.reshape((1,) + contact_points.shape)
        if contact_mask.ndim == 2:
            contact_mask = contact_mask.reshape((1,) + contact_mask.shape)
        contact_batch = contact_points.shape[0]
        if contact_batch != self.contact_points_wp.shape[0]:
            # rebatching replaces the stable buffers; only safe before graph capture
            expected = (contact_batch, self.num_frames, self.max_contacts, 3)
            self.contact_points_wp = wp.zeros(expected, dtype=wp.float32, device=self.device)
            self.local_contact_points_wp = wp.zeros(expected, dtype=wp.float32, device=self.device)
            self.contact_axes_wp = wp.zeros(expected, dtype=wp.float32, device=self.device)
            self.contact_mask_wp = wp.zeros(expected[:-1], dtype=wp.bool, device=self.device)
        expected_points_shape = (contact_batch, self.num_frames, self.max_contacts, 3)
        expected_mask_shape = expected_points_shape[:-1]
        if contact_points.shape != expected_points_shape:
            raise ValueError(f"contact_points must have shape {expected_points_shape}.")
        if contact_mask.shape != expected_mask_shape:
            raise ValueError(f"contact_mask must have shape {expected_mask_shape}.")
        if contact_points.dtype != wp.float32 or contact_mask.dtype != wp.bool:
            raise TypeError("contact_points must be wp.float32 and contact_mask must be wp.bool.")
        optional = [
            a.reshape((1,) + a.shape) if a is not None and a.ndim == 3 else a
            for a in (local_contact_points, contact_axes)
        ]
        if any(a is not None and (a.shape != expected_points_shape or a.dtype != wp.float32) for a in optional):
            raise ValueError(
                f"local_contact_points and contact_axes must be wp.float32 of shape {expected_points_shape}."
            )
        if any(a is not None and a.device != self.device for a in (contact_points, contact_mask, *optional)):
            raise ValueError(f"contact inputs must be on {self.device}.")
        wp.copy(self.contact_points_wp, contact_points)
        wp.copy(self.contact_mask_wp, contact_mask)
        for buffer, value in zip((self.local_contact_points_wp, self.contact_axes_wp), optional):
            if value is None:
                buffer.zero_()
            else:
                wp.copy(buffer, value)

    @property
    def residual_dim(self) -> int:
        return self.num_frames * self.max_contacts * 3

    def compute_weighted_residual(
        self,
        var_values: "VarValues",
        *args: object,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        robot_state = var_values.get(VarKey.ROBOT)
        if self.device is None:
            self._init_buffers(robot_state.q.device)

        if not robot_state.is_fk_computed:
            self.robot.forward_kinematics(robot_state)

        kernel_device = out_residual.device if out_residual is not None else self.device

        if out_residual is None:
            out_residual = wp.empty((robot_state.q.shape[0], self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        wp.launch(
            kernel=_compute_trajectory_contact_residual_kernel,
            dim=(robot_state.q.shape[0], self.num_frames, self.max_contacts),
            inputs=[
                robot_state.T_world_link,
                self.contact_points_wp,
                self.contact_axes_wp,
                self.local_contact_points_wp,
                self.contact_link_indices_wp,
                self.contact_mask_wp,
                self.max_contacts,
                self.contact_margin,
                self.weight,
                row_offset,
            ],
            outputs=[out_residual],
            device=kernel_device,
        )

        return out_residual

    def compute_sparse_jacobian_pattern(
        self,
        var_values: "VarValues",
        *args: object,
        offset: int = 0,
        out_row_indices: Optional[wp.array] = None,
        out_col_indices: Optional[wp.array] = None,
        nnz_offset: int = 0,
        **kwargs: object,
    ) -> SparsityPattern:
        robot_state = var_values.get(VarKey.ROBOT)
        if self.device is None:
            self._init_buffers(robot_state.q.device)

        kernel_device = robot_state.q.device
        num_dofs = self.robot.spec.num_actuated_joints
        base_dim = 6 if bool(robot_state.has_floating_base) else 0
        single_tangent_dim = base_dim + num_dofs

        cols_per_residual = self.max_ancestor_dofs + base_dim
        nnz = self.num_frames * self.max_contacts * 3 * cols_per_residual

        row_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_row_indices is None else out_row_indices
        )
        col_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_col_indices is None else out_col_indices
        )

        wp.launch(
            kernel=_compute_trajectory_contact_jacobian_sparse_pattern_kernel,
            dim=(self.num_frames, self.max_contacts),
            inputs=[
                self.contact_link_indices_wp,
                self.link_ancestor_dofs_wp,
                self.link_num_ancestor_dofs_wp,
                self.max_ancestor_dofs,
                self.max_contacts,
                single_tangent_dim,
                base_dim,
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
        var_values: "VarValues",
        *args: object,
        out_jacobian_values: Optional[wp.array] = None,
        offset: int = 0,
        nnz_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        robot_state = var_values.get(VarKey.ROBOT)
        if self.device is None:
            self._init_buffers(robot_state.q.device)

        if not robot_state.is_fk_computed:
            self.robot.forward_kinematics(robot_state)
        if not robot_state.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(robot_state)

        kernel_device = robot_state.q.device
        base_dim = 6 if bool(robot_state.has_floating_base) else 0

        cols_per_residual = self.max_ancestor_dofs + base_dim
        nnz = self.num_frames * self.max_contacts * 3 * cols_per_residual

        if out_jacobian_values is None:
            out_jacobian_values = wp.empty((robot_state.q.shape[0], nnz), dtype=wp.float32, device=kernel_device)

        spec_tensors = robot_state.spec_tensors

        wp.launch(
            kernel=_compute_trajectory_contact_jacobian_sparse_values_kernel,
            dim=(robot_state.q.shape[0], self.num_frames, self.max_contacts),
            inputs=[
                robot_state.S_world,
                robot_state.T_world_link,
                robot_state.T_world_base,
                self.contact_points_wp,
                self.contact_axes_wp,
                self.local_contact_points_wp,
                self.contact_link_indices_wp,
                self.contact_mask_wp,
                self.link_ancestor_dofs_wp,
                self.link_num_ancestor_dofs_wp,
                spec_tensors.link_ancestor_joints_mask,
                spec_tensors.joints_to_actuated_mapping,
                self.max_ancestor_dofs,
                self.max_contacts,
                self.contact_margin,
                self.weight,
                base_dim,
                nnz_offset,
            ],
            outputs=[out_jacobian_values],
            device=kernel_device,
        )

        return out_jacobian_values


if TYPE_CHECKING:
    from robokit.opt.var_values import VarValues
    from robokit.robo import Robot
