# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false, reportCallIssue=false, reportReturnType=false
# pyright: reportOperatorIssue=false
# pyright: reportUndefinedVariable=false
"""Sparse sphere self-collision over a trajectory."""

from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, cast

import numpy as np
import warp as wp

from robokit.opt.var_values import VarKey, VarValues
from robokit.robo.robot_state import RobotState
from robokit.terms.dense.self_collision_task import compute_active_collision_pairs
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import Context, GradientTask, SparseTask, SparsityPattern, sparse_cost_and_gradient
from robokit.utils.warp_utils import wp_device_type


# --- task -------------------------------------------------------------------
class TrajectorySelfCollisionTask(RobotTask, SparseTask, GradientTask):
    """Penalize self-collisions between robot collision spheres at all frames."""

    compute_weighted_cost_and_gradient = sparse_cost_and_gradient
    precompute_collision_geometry = "sphere"  # pairs read world sphere centers, computed once

    def __init__(
        self,
        robot: Optional["Robot"] = None,  # noqa: F821
        num_frames: Optional[int] = None,
        weight: float = 10.0,
        margin: float = 0.02,
        penalty: Literal["squared", "max_linear"] = "squared",
        name: str = "",
    ):
        self.name = name
        self.robot = robot
        if num_frames is not None:
            self.num_frames = num_frames
        self.weight = float(weight)
        self.margin = margin
        # max_linear: per frame, only the deepest pair, with cost weight * penetration
        self.penalty = penalty
        self._argmax: Optional[wp.array] = None
        if robot is None:  # an unbuilt stub: the trajectory optimizer builds the real task
            return
        if not robot.spec.has_collision_spheres:
            raise RuntimeError("Robot has no collision spheres. Load with load_collision_spheres=True.")
        buffers = robot.spec.self_collision_link_buffers  # the robot asset's sphere inflation, for this check only
        self._radii_np = np.asarray(
            [
                radius + buffers.get(robot.link_names[link], 0.0)
                for radius, link in zip(robot.spec.collision_sphere_radii, robot.spec.collision_spheres_link_indices)
            ],
            dtype=np.float32,
        )

        self._active_pairs_np = compute_active_collision_pairs(robot.spec)
        ignored = {(int(a), int(b)) for a, b in robot.spec.self_collision_ignored_pairs}
        if ignored and len(self._active_pairs_np) > 0:
            ignored_both = ignored | {(j, i) for i, j in ignored}
            geom_link = robot.spec.collision_spheres_link_indices
            keep = [(int(geom_link[i]), int(geom_link[j])) not in ignored_both for i, j in self._active_pairs_np]
            self._active_pairs_np = self._active_pairs_np[np.asarray(keep, dtype=bool)]
        self.num_pairs = len(self._active_pairs_np)
        self._slots = 1 if penalty == "max_linear" else max(1, self.num_pairs)  # residuals per frame

        self.device: Optional[wp_device_type] = None
        self._active_pairs_wp: Optional[wp.array] = None
        self.sphere_mask: Optional[wp.array] = None  # (frames, spheres): 0 drops a sphere's pairs at a frame
        self.pair_mask: Optional[wp.array] = None  # (pairs,): 0 drops a pair

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Build from the robot and horizon of the trajectory `var_values` holds."""
        super().resolve(var_values, context)
        state = cast(RobotState, var_values.get(self.var_key))
        TrajectorySelfCollisionTask.__init__(
            self,
            state.robot,
            state.q.shape[1],
            weight=self.weight,
            margin=self.margin,
            penalty=self.penalty,
            name=self.name,
        )

    def init_buffers(self, device: wp_device_type):
        self.device = device
        self._radii_wp = wp.from_numpy(self._radii_np, dtype=wp.float32, device=device)
        self._no_argmax = wp.full((1, 1), -1, dtype=wp.int32, device=device)
        if self.sphere_mask is None:
            self.sphere_mask = wp.ones((self.num_frames, len(self._radii_np)), dtype=wp.int32, device=device)
        if self.pair_mask is None:
            self.pair_mask = wp.ones(max(1, self.num_pairs), dtype=wp.int32, device=device)
        if self.num_pairs > 0:
            self._active_pairs_wp = wp.from_numpy(self._active_pairs_np, dtype=wp.int32, device=device)
        else:
            self._active_pairs_wp = wp.zeros((1, 2), dtype=wp.int32, device=device)

    @property
    def residual_dim(self) -> int:
        return self.num_frames * self._slots

    def _accumulate_max_residual(self, var_values: "VarValues", out: wp.array):  # noqa: F821
        """Accumulate positive residual maxima without storing every frame/pair residual."""
        state = var_values.get(VarKey.ROBOT)
        if self.device is None:
            self.init_buffers(state.q.device)
        if self.num_pairs == 0:
            return
        if self.use_mask:
            self._build_frame_mask(var_values.device, var_values.batch_size)
        if not state.is_collision_spheres_computed:
            self.robot.transform_collision_spheres(state)
        wp.launch_tiled(
            _max_residual_kernel,
            dim=[state.q.shape[0] * self.num_frames],
            inputs=[
                state.collision_sphere_centers_world,
                self._radii_wp,
                self._active_pairs_wp,
                self.margin,
                self.num_frames,
                self.weight,
                self.use_mask,
                self._get_frame_mask(),
                self.sphere_mask,
                self.pair_mask,
                self.penalty == "max_linear",
            ],
            outputs=[out],
            block_dim=_PAIR_THREADS,
            device=out.device,
        )

    def compute_weighted_residual(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
        out_jacobian: Optional[wp.array] = None,  # max_linear: also write the selected pairs' Jacobian values
        nnz_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        robot_state = var_values.get(VarKey.ROBOT)
        if self.device is None:
            self.init_buffers(robot_state.q.device)

        if not robot_state.is_collision_spheres_computed:
            self.robot.transform_collision_spheres(robot_state)

        kernel_device = out_residual.device if out_residual is not None else self.device

        if out_residual is None:
            out_residual = wp.zeros((robot_state.q.shape[0], self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if self.num_pairs == 0:
            return out_residual

        if self.penalty == "max_linear":
            shape = (robot_state.q.shape[0], self.num_frames)
            if self._argmax is None or self._argmax.shape[0] < shape[0]:  # line searches alternate batch sizes
                self._argmax = wp.full(shape, -1, dtype=wp.int32, device=kernel_device)
            # evaluate pair distances in parallel and keep each frame's deepest collision
            wp.launch_tiled(
                _deepest_pair_kernel,
                dim=[shape[0] * shape[1]],
                inputs=[
                    robot_state.collision_sphere_centers_world,
                    self._radii_wp,
                    self._active_pairs_wp,
                    self.margin,
                    self.num_frames,
                    self.weight,
                    row_offset,
                    self.use_mask,
                    self._get_frame_mask(),
                    self.sphere_mask,
                    self.pair_mask,
                ],
                outputs=[out_residual, self._argmax],
                block_dim=_PAIR_THREADS,
                device=kernel_device,
            )
            if out_jacobian is not None:
                self.compute_weighted_sparse_jacobian_values(
                    var_values, out_jacobian_values=out_jacobian, nnz_offset=nnz_offset
                )
            return out_residual
        wp.launch(
            kernel=compute_self_collision_residual_kernel,
            dim=(robot_state.q.shape[0], self.num_frames, self.num_pairs),
            inputs=[
                robot_state.collision_sphere_centers_world,
                self._radii_wp,
                self._active_pairs_wp,
                self.margin,
                self.weight,
                row_offset,
                self.num_pairs,
                self.use_mask,
                self._get_frame_mask(),
                self.sphere_mask,
                self.pair_mask,
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
        num_actuated = self.robot.spec.num_actuated_joints
        base_dim = 6 if bool(robot_state.has_floating_base) else 0
        single_tangent_dim = base_dim + num_actuated

        effective_num_pairs = self._slots
        cols_per_residual = num_actuated + base_dim
        nnz = self.num_frames * effective_num_pairs * cols_per_residual

        row_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_row_indices is None else out_row_indices
        )
        col_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_col_indices is None else out_col_indices
        )

        wp.launch(
            kernel=compute_self_collision_jacobian_pattern_kernel,
            dim=(self.num_frames, effective_num_pairs, cols_per_residual),
            inputs=[
                effective_num_pairs,
                num_actuated,
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

        if not robot_state.is_collision_spheres_computed:
            self.robot.transform_collision_spheres(robot_state)
        if not robot_state.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(robot_state)

        kernel_device = robot_state.q.device
        spec_tensors = robot_state.spec_tensors
        num_actuated = self.robot.spec.num_actuated_joints
        base_dim = 6 if bool(robot_state.has_floating_base) else 0
        effective_num_pairs = self._slots
        cols_per_residual = num_actuated + base_dim
        nnz = self.num_frames * effective_num_pairs * cols_per_residual

        if out_jacobian_values is None:
            out_jacobian_values = wp.zeros((robot_state.q.shape[0], nnz), dtype=wp.float32, device=kernel_device)

        if self.num_pairs == 0:
            return out_jacobian_values

        wp.launch(
            kernel=compute_self_collision_jacobian_values_kernel,
            dim=(robot_state.q.shape[0], self.num_frames, self._slots, cols_per_residual),
            inputs=[
                robot_state.S_world,
                robot_state.collision_sphere_centers_world,
                self._radii_wp,
                spec_tensors.collision_spheres_link_indices,
                spec_tensors.link_ancestor_joints_mask,
                spec_tensors.joints_to_actuated_mapping,
                self._active_pairs_wp,
                self.margin,
                self.weight,
                self._slots,
                num_actuated,
                base_dim,
                nnz_offset,
                self.use_mask,
                self._get_frame_mask(),
                self.sphere_mask,
                self.pair_mask,
                self._argmax if self._argmax is not None else self._no_argmax,
                self.penalty == "max_linear",
            ],
            outputs=[out_jacobian_values],
            device=kernel_device,
        )

        return out_jacobian_values


# --- device code ------------------------------------------------------------
@wp.kernel
def compute_self_collision_residual_kernel(
    sphere_centers: wp.array3d(dtype=wp.vec3),  # [batch, frames, num_spheres], world
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    active_pairs: wp.array2d(dtype=wp.int32),  # [num_pairs, 2]
    margin: float,
    weight: float,
    row_offset: int,
    num_pairs: int,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    sphere_mask: wp.array2d(dtype=wp.int32),
    pair_mask: wp.array1d(dtype=wp.int32),
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx, frame_idx, pair_idx = wp.tid()  # pyright: ignore[reportAssignmentType]

    sphere_i = active_pairs[pair_idx, 0]
    sphere_j = active_pairs[pair_idx, 1]

    diff = sphere_centers[batch_idx, frame_idx, sphere_i] - sphere_centers[batch_idx, frame_idx, sphere_j]
    surface_dist = wp.length(diff) - collision_sphere_radii[sphere_i] - collision_sphere_radii[sphere_j]
    penetration = wp.max(0.0, margin - surface_dist)
    mask = wp.float32(sphere_mask[frame_idx, sphere_i] * sphere_mask[frame_idx, sphere_j] * pair_mask[pair_idx])
    if use_mask:
        mask *= wp.float32(frame_mask[batch_idx, frame_idx])

    residual_idx = frame_idx * num_pairs + pair_idx
    out_residual[batch_idx, row_offset + residual_idx] = weight * penetration * mask


_PAIR_THREADS = 256


@wp.kernel
def _deepest_pair_kernel(
    sphere_centers: wp.array3d(dtype=wp.vec3),
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    active_pairs: wp.array2d(dtype=wp.int32),
    margin: float,
    num_frames: int,
    weight: float,
    row_offset: int,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    sphere_mask: wp.array2d(dtype=wp.int32),
    pair_mask: wp.array1d(dtype=wp.int32),
    out_residual: wp.array2d(dtype=wp.float32),
    argmax: wp.array2d(dtype=wp.int32),
):
    """Reduce each frame's deepest pair to sqrt(2 w p); threads scan strided pairs (one thread on CPU)."""
    row, lane = wp.tid()  # pyright: ignore[reportAssignmentType]
    batch_idx = row // num_frames
    frame_idx = row % num_frames
    deepest = float(-1.0e30)
    best = int(-1)
    for pair in range(lane, active_pairs.shape[0], wp.block_dim()):
        sphere_i = active_pairs[pair, 0]
        sphere_j = active_pairs[pair, 1]
        diff = sphere_centers[batch_idx, frame_idx, sphere_i] - sphere_centers[batch_idx, frame_idx, sphere_j]
        penetration = margin - (wp.length(diff) - collision_sphere_radii[sphere_i] - collision_sphere_radii[sphere_j])
        on = sphere_mask[frame_idx, sphere_i] * sphere_mask[frame_idx, sphere_j] * pair_mask[pair]
        if penetration > deepest and on != 0:
            deepest = penetration
            best = pair
    lane_deepest = wp.tile(deepest)
    winner = wp.tile_extract(wp.tile_argmax(lane_deepest), 0)
    best = wp.tile_extract(wp.tile(best), winner)
    deepest = wp.tile_extract(lane_deepest, winner)
    if use_mask:  # Warp boolean expressions evaluate both operands; the unused mask may be only (1, 1).
        if frame_mask[batch_idx, frame_idx] == wp.uint8(0):
            deepest = 0.0
    if deepest <= 0.0:
        best = -1
        deepest = 0.0
    if lane == 0:
        argmax[batch_idx, frame_idx] = best
        out_residual[batch_idx, row_offset + frame_idx] = wp.sqrt(2.0 * weight * deepest)


@wp.kernel(enable_backward=False)
def _max_residual_kernel(
    sphere_centers: wp.array3d(dtype=wp.vec3),
    radii: wp.array1d(dtype=wp.float32),
    active_pairs: wp.array2d(dtype=wp.int32),
    margin: float,
    num_frames: int,
    weight: float,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    sphere_mask: wp.array2d(dtype=wp.int32),
    pair_mask: wp.array1d(dtype=wp.int32),
    max_linear: wp.bool,
    out: wp.array1d(dtype=wp.float32),
):
    row, lane = wp.tid()  # pyright: ignore[reportAssignmentType]
    batch, frame = row // num_frames, row % num_frames
    mask = float(1.0)
    if use_mask:
        mask = wp.float32(frame_mask[batch, frame])
        if mask == 0.0:
            return
    deepest = float(-1.0e30)
    for pair in range(lane, active_pairs.shape[0], wp.block_dim()):
        i, j = active_pairs[pair, 0], active_pairs[pair, 1]
        diff = sphere_centers[batch, frame, i] - sphere_centers[batch, frame, j]
        penetration = margin - (wp.length(diff) - radii[i] - radii[j])
        if sphere_mask[frame, i] * sphere_mask[frame, j] * pair_mask[pair] != 0:
            deepest = wp.where(penetration > deepest, penetration, deepest)
    deepest = wp.max(0.0, wp.tile_extract(wp.tile_max(wp.tile(deepest)), 0))
    if lane == 0:
        value = weight * deepest * mask
        if max_linear:
            value = wp.sqrt(2.0 * weight * deepest)
        if value > 0.0:
            wp.atomic_max(out, batch, value)


@wp.kernel
def compute_self_collision_jacobian_pattern_kernel(
    num_pairs: int,
    num_actuated: int,
    single_tangent_dim: int,
    base_dim: int,
    row_offset: int,
    nnz_offset: int,
    row_indices: wp.array1d(dtype=wp.int32),
    col_indices: wp.array1d(dtype=wp.int32),
):
    frame_idx, pair_idx, col_local_idx = wp.tid()  # pyright: ignore[reportAssignmentType]

    residual_idx = row_offset + frame_idx * num_pairs + pair_idx
    cols_per_residual = num_actuated + base_dim
    nnz_idx = nnz_offset + (frame_idx * num_pairs + pair_idx) * cols_per_residual + col_local_idx

    row_indices[nnz_idx] = residual_idx

    if col_local_idx < num_actuated:
        col_indices[nnz_idx] = frame_idx * single_tangent_dim + base_dim + col_local_idx
    else:
        base_col = col_local_idx - num_actuated
        col_indices[nnz_idx] = frame_idx * single_tangent_dim + base_col


@wp.func
def _pair_jacobian_value_func(
    S_world: wp.array4d(dtype=wp.float32),
    sphere_centers: wp.array3d(dtype=wp.vec3),
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    sphere_i: int,
    sphere_j: int,
    margin: float,
    weight: float,
    max_linear: wp.bool,
    batch_idx: int,
    frame_idx: int,
    col_local_idx: int,
    num_actuated: int,
) -> float:
    """One pair residual's derivative along one Jacobian column."""
    link_i = collision_spheres_link_indices[sphere_i]
    link_j = collision_spheres_link_indices[sphere_j]
    world_center_i = sphere_centers[batch_idx, frame_idx, sphere_i]
    world_center_j = sphere_centers[batch_idx, frame_idx, sphere_j]

    diff = world_center_i - world_center_j
    dist = wp.length(diff)
    surface_dist = dist - collision_sphere_radii[sphere_i] - collision_sphere_radii[sphere_j]

    dres_ddist = float(0.0)
    if (margin - surface_dist) > 0.0:
        dres_ddist = -weight
        if max_linear:  # d sqrt(2 w p) / d distance = -w / sqrt(2 w p)
            dres_ddist = -weight / wp.max(wp.sqrt(2.0 * weight * (margin - surface_dist)), 1.0e-12)

    denom = wp.max(wp.float32(1e-8), dist)
    normal = diff / denom

    pv_i_x = float(0.0)
    pv_i_y = float(0.0)
    pv_i_z = float(0.0)
    pv_j_x = float(0.0)
    pv_j_y = float(0.0)
    pv_j_z = float(0.0)

    if col_local_idx < num_actuated:
        actuated_idx = col_local_idx
        num_joints = S_world.shape[3]

        for joint_idx in range(num_joints):
            jta_weight = joints_to_actuated[joint_idx, actuated_idx]
            if jta_weight != 0.0:
                if link_ancestor_joints_mask[link_i, joint_idx]:
                    v_x = S_world[batch_idx, frame_idx, 0, joint_idx] * jta_weight
                    v_y = S_world[batch_idx, frame_idx, 1, joint_idx] * jta_weight
                    v_z = S_world[batch_idx, frame_idx, 2, joint_idx] * jta_weight
                    w_x = S_world[batch_idx, frame_idx, 3, joint_idx] * jta_weight
                    w_y = S_world[batch_idx, frame_idx, 4, joint_idx] * jta_weight
                    w_z = S_world[batch_idx, frame_idx, 5, joint_idx] * jta_weight
                    pv_i_x = pv_i_x + v_x + w_y * world_center_i[2] - w_z * world_center_i[1]
                    pv_i_y = pv_i_y + v_y + w_z * world_center_i[0] - w_x * world_center_i[2]
                    pv_i_z = pv_i_z + v_z + w_x * world_center_i[1] - w_y * world_center_i[0]

                if link_ancestor_joints_mask[link_j, joint_idx]:
                    v_x = S_world[batch_idx, frame_idx, 0, joint_idx] * jta_weight
                    v_y = S_world[batch_idx, frame_idx, 1, joint_idx] * jta_weight
                    v_z = S_world[batch_idx, frame_idx, 2, joint_idx] * jta_weight
                    w_x = S_world[batch_idx, frame_idx, 3, joint_idx] * jta_weight
                    w_y = S_world[batch_idx, frame_idx, 4, joint_idx] * jta_weight
                    w_z = S_world[batch_idx, frame_idx, 5, joint_idx] * jta_weight
                    pv_j_x = pv_j_x + v_x + w_y * world_center_j[2] - w_z * world_center_j[1]
                    pv_j_y = pv_j_y + v_y + w_z * world_center_j[0] - w_x * world_center_j[2]
                    pv_j_z = pv_j_z + v_z + w_x * world_center_j[1] - w_y * world_center_j[0]
    else:
        base_col = col_local_idx - num_actuated
        if base_col < 3:
            if base_col == 0:
                pv_i_x = 1.0
                pv_j_x = 1.0
            elif base_col == 1:
                pv_i_y = 1.0
                pv_j_y = 1.0
            else:
                pv_i_z = 1.0
                pv_j_z = 1.0
        else:
            axis = base_col - 3
            if axis == 0:
                pv_i_y = world_center_i[2]
                pv_i_z = -world_center_i[1]
                pv_j_y = world_center_j[2]
                pv_j_z = -world_center_j[1]
            elif axis == 1:
                pv_i_x = -world_center_i[2]
                pv_i_z = world_center_i[0]
                pv_j_x = -world_center_j[2]
                pv_j_z = world_center_j[0]
            else:
                pv_i_x = world_center_i[1]
                pv_i_y = -world_center_i[0]
                pv_j_x = world_center_j[1]
                pv_j_y = -world_center_j[0]

    rel_vel_x = pv_i_x - pv_j_x
    rel_vel_y = pv_i_y - pv_j_y
    rel_vel_z = pv_i_z - pv_j_z

    ddist_dq = normal[0] * rel_vel_x + normal[1] * rel_vel_y + normal[2] * rel_vel_z
    return dres_ddist * ddist_dq


@wp.kernel
def compute_self_collision_jacobian_values_kernel(
    S_world: wp.array4d(dtype=wp.float32),  # [batch, frames, 6, num_joints]
    sphere_centers: wp.array3d(dtype=wp.vec3),  # [batch, frames, num_spheres], world
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joints_to_actuated: wp.array2d(dtype=wp.float32),
    active_pairs: wp.array2d(dtype=wp.int32),
    margin: float,
    weight: float,
    num_pairs: int,
    num_actuated: int,
    base_dim: int,
    nnz_offset: int,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    sphere_mask: wp.array2d(dtype=wp.int32),
    pair_mask: wp.array1d(dtype=wp.int32),
    argmax: wp.array2d(dtype=wp.int32),
    max_linear: wp.bool,
    values: wp.array2d(dtype=wp.float32),
):
    batch_idx, frame_idx, pair_idx, col_local_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    cols_per_residual = num_actuated + base_dim
    nnz_idx = nnz_offset + (frame_idx * num_pairs + pair_idx) * cols_per_residual + col_local_idx
    pair = pair_idx
    if max_linear:  # one residual per frame, differentiating its deepest pair
        pair = argmax[batch_idx, frame_idx]
        if pair < 0:
            values[batch_idx, nnz_idx] = 0.0
            return

    sphere_i = active_pairs[pair, 0]
    sphere_j = active_pairs[pair, 1]

    mask = wp.float32(sphere_mask[frame_idx, sphere_i] * sphere_mask[frame_idx, sphere_j] * pair_mask[pair])
    if use_mask:
        mask *= wp.float32(frame_mask[batch_idx, frame_idx])
    values[batch_idx, nnz_idx] = mask * _pair_jacobian_value_func(
        S_world,
        sphere_centers,
        collision_sphere_radii,
        collision_spheres_link_indices,
        link_ancestor_joints_mask,
        joints_to_actuated,
        sphere_i,
        sphere_j,
        margin,
        weight,
        max_linear,
        batch_idx,
        frame_idx,
        col_local_idx,
        num_actuated,
    )


if TYPE_CHECKING:
    from robokit.opt.var_values import VarValues
    from robokit.robo import Robot
