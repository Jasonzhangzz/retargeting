# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportCallIssue=false
# pyright: reportUndefinedVariable=false
# pyright: reportOptionalSubscript=false
"""Sparse trajectory extension of `SceneCollisionTask` with optional swept sampling."""

from functools import lru_cache
from typing import TYPE_CHECKING, Any, List, Literal, Mapping, Optional, Sequence, Union, cast

import numpy as np
import warp as wp

from robokit.geom import MeshGeom, VolumeGeom
from robokit.geom.sdf_kernels import (
    PRIM_BOX,
    SdfHit,
    VolumeObservations,
    closest_on_meshes,
    closest_on_meshes_with_sdf,
    closest_on_primitives,
    closest_on_volumes,
)
from robokit.lie.se3_kernels import se3_adjoint_func
from robokit.opt.var_values import VarKey
from robokit.terms.robot_task import RobotTask
from robokit.terms.sphere_collision_kernels import (
    PENALTY_PLAIN,
    PENALTY_SMOOTH_ROOT,
    PENALTY_SURFACE_DISTANCE,
    _compute_collision_penalty_func,
)
from robokit.terms.task import (
    Context,
    GradientTask,
    SparseTask,
    SparsityPattern,
    _sparse_cost_deterministic_kernel,
    _sparse_gradient_csc_kernel,
    sparse_cost_and_gradient,
)
from robokit.utils.warp_utils import group_sum, row_sum32, warp_shuffle, wp_device_type, wp_vec6, wp_vec7


if TYPE_CHECKING:
    from robokit.geom import BaseGeom, WarpScene
    from robokit.opt.var_values import VarValues
    from robokit.robo import Robot, RobotState


# --- device code ------------------------------------------------------------
@wp.func
def _transform_point_se3_func(T_world_link: wp_vec7, local_point: wp.vec3) -> wp.vec3:
    pos = wp.vec3(T_world_link[0], T_world_link[1], T_world_link[2])
    quat = wp.quat(T_world_link[4], T_world_link[5], T_world_link[6], T_world_link[3])
    return pos + wp.quat_rotate(quat, local_point)


@wp.func
def _compute_cached_gradient_func(
    world_center: wp.vec3, hit_dist: float, hit_closest: wp.vec3, hit_normal: wp.vec3
) -> wp.vec3:
    """Compute the signed closest-point direction, with the hit normal as fallback."""
    diff = world_center - hit_closest
    dlen = wp.length(diff)
    if dlen > 1.0e-8:
        if hit_dist < 0.0:
            return diff / dlen * (-1.0)
        return diff / dlen
    return hit_normal


@wp.func
def _frame_sphere_weight_func(
    frame_sphere_mask: wp.array3d(dtype=wp.bool),
    batch_idx: int,
    frame_idx: int,
    sphere_idx: int,
    solve_batch_size: int,
) -> float:
    mask_batch_idx = batch_idx // (solve_batch_size // frame_sphere_mask.shape[0])
    mask_sphere_idx = wp.min(sphere_idx, frame_sphere_mask.shape[2] - 1)
    return float(frame_sphere_mask[mask_batch_idx, frame_idx, mask_sphere_idx])


@wp.func
def _scene_index_func(
    scene_indices: wp.array2d(dtype=wp.int32),  # (scenes, frames) or (scenes, 1); empty: one scene
    solve_batch_size: int,
    batch_idx: int,
    frame_idx: int,
) -> int:
    scenes = scene_indices.shape[0]
    if scenes == 0:
        return 0
    source_batch_idx = batch_idx // wp.max(1, solve_batch_size // scenes)  # a batch narrower than the scenes
    return scene_indices[source_batch_idx, wp.where(scene_indices.shape[1] > 1, frame_idx, 0)]


@wp.func
def _margin_func(
    goal_distance: wp.array1d(dtype=wp.float32),
    goal_radius: float,
    sphere_margin: wp.array1d(dtype=wp.float32),
    batch_idx: int,
    batch_size: int,
    sphere_local_idx: int,
) -> float:
    # the sphere's own margin, tapered inside `goal_radius` of each query's joint goal
    margin = sphere_margin[sphere_local_idx]
    if goal_radius > 0.0:
        distance = goal_distance[batch_idx // wp.max(1, batch_size // goal_distance.shape[0])]
        margin = margin * wp.min(distance / wp.max(goal_radius, 1.0e-8), 1.0)
    return margin


@wp.kernel
def zero_collision_rows_kernel(
    offset: int,
    values: wp.array2d(dtype=wp.float32),
):
    batch_idx, column_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    values[batch_idx, offset + column_idx] = 0.0


@wp.struct
class _SceneQuery:
    """One geometry's query arrays; fields its kind does not read stay unset."""

    prim_type: wp.int32
    params: wp.array1d(dtype=wp.vec4)
    mesh_ids: wp.array1d(dtype=wp.uint64)
    volume_ids: wp.array1d(dtype=wp.uint64)
    aabb_min: wp.array1d(dtype=wp.vec3)
    aabb_max: wp.array1d(dtype=wp.vec3)
    paddings: wp.array1d(dtype=wp.float32)
    observations: wp.array1d(dtype=VolumeObservations)
    refine_band: float
    max_dist: float
    inv_poses: wp.array1d(dtype=wp.mat44)
    enable_inv_poses: bool
    scales: wp.array1d(dtype=wp.float32)
    enable_scales: bool
    offsets: wp.array1d(dtype=wp.int32)


def _build_query_kernels(kind: str):
    """Build the residual and Jacobian-cache kernels of one geometry kind; `wp.static` compiles out the others."""

    @wp.func
    def closest(q: _SceneQuery, p: wp.vec3, si: int, want_normal: bool) -> SdfHit:
        start, end = q.offsets[si], q.offsets[si + 1]
        if wp.static(kind == "primitive"):
            return closest_on_primitives(
                p, 1e6, q.prim_type, q.params, q.inv_poses, q.enable_inv_poses, q.scales, q.enable_scales,
                start, end, want_normal,
            )  # fmt: skip
        elif wp.static(kind == "mesh"):
            return closest_on_meshes(
                p, 1e6, q.mesh_ids, q.aabb_min, q.aabb_max, q.inv_poses, q.enable_inv_poses, q.scales,
                q.enable_scales, start, end, q.max_dist, want_normal,
            )  # fmt: skip
        elif wp.static(kind == "volume"):
            return closest_on_volumes(
                p, 1e6, q.volume_ids, q.aabb_min, q.aabb_max, q.paddings, q.observations, q.inv_poses, q.enable_inv_poses,
                q.scales, q.enable_scales, start, end, want_normal,
            )  # fmt: skip
        else:
            return closest_on_meshes_with_sdf(
                p, 1e6, q.mesh_ids, q.aabb_min, q.aabb_max, q.volume_ids, q.paddings, q.refine_band, q.max_dist,
                q.inv_poses, q.enable_inv_poses, q.scales, q.enable_scales, start, end, want_normal,
            )  # fmt: skip

    @wp.kernel(module="unique")
    def residual_kernel(
        T_world_link: wp.array3d(dtype=wp_vec7),
        local_collision_sphere_centers: wp.array1d(dtype=wp.vec3),
        collision_sphere_radii: wp.array1d(dtype=wp.float32),
        collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
        sphere_indices: wp.array1d(dtype=wp.int32),
        query: _SceneQuery,
        scene_indices: wp.array2d(dtype=wp.int32),  # (scenes, frames) or (scenes, 1); empty: one scene
        goal_distance: wp.array1d(dtype=wp.float32),
        goal_radius: float,
        sphere_margin: wp.array1d(dtype=wp.float32),
        residual_weight: wp.array1d(dtype=wp.float32),
        frame_sphere_mask: wp.array3d(dtype=wp.bool),
        sweep_steps: int,
        penalty_mode: wp.int32,
        row_offset: int,
        num_spheres: int,
        out_residual: wp.array2d(dtype=wp.float32),
    ):
        batch_idx, frame_idx, sphere_local_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
        residual_idx = frame_idx * num_spheres + sphere_local_idx
        scale = residual_weight[residual_idx] * _frame_sphere_weight_func(
            frame_sphere_mask, batch_idx, frame_idx, sphere_local_idx, T_world_link.shape[0]
        )
        if scale == 0.0:  # a masked or weightless row skips its scene query
            out_residual[batch_idx, row_offset + residual_idx] = 0.0
            return
        sphere_global_idx = sphere_indices[sphere_local_idx]
        link_idx = collision_spheres_link_indices[sphere_global_idx]
        local_center = local_collision_sphere_centers[sphere_global_idx]
        radius = collision_sphere_radii[sphere_global_idx]
        center = _transform_point_se3_func(T_world_link[batch_idx, frame_idx, link_idx], local_center)
        prev_center = center
        steps = wp.where(wp.static(kind == "primitive"), sweep_steps, 0)  # only primitive scenes sweep
        if steps > 0:
            prev_frame = wp.max(frame_idx - 1, 0)
            prev_center = _transform_point_se3_func(T_world_link[batch_idx, prev_frame, link_idx], local_center)

        si = _scene_index_func(scene_indices, T_world_link.shape[0], batch_idx, frame_idx)
        eta = _margin_func(
            goal_distance, goal_radius, sphere_margin, batch_idx, T_world_link.shape[0], sphere_local_idx
        )
        # keep the largest residual across interpolated positions; without a sweep, the frame's own position
        residual = float(0.0)
        for s in range(steps + 1):
            t = float(s + 1) / float(steps + 1)
            hit = closest(query, prev_center + t * (center - prev_center), si, False)
            residual = wp.max(residual, _compute_collision_penalty_func(hit.dist - radius, eta, penalty_mode)[0])
        # This row scores the whole segment into `frame_idx`, so zeroing the last row drops the segment.
        out_residual[batch_idx, row_offset + residual_idx] = scale * residual

    @wp.kernel(module="unique")
    def jacobian_kernel(
        T_world_link: wp.array3d(dtype=wp_vec7),
        local_collision_sphere_centers: wp.array1d(dtype=wp.vec3),
        collision_sphere_radii: wp.array1d(dtype=wp.float32),
        collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
        sphere_indices: wp.array1d(dtype=wp.int32),
        query: _SceneQuery,
        scene_indices: wp.array2d(dtype=wp.int32),  # (scenes, frames) or (scenes, 1); empty: one scene
        num_spheres: int,
        goal_distance: wp.array1d(dtype=wp.float32),
        goal_radius: float,
        sphere_margin: wp.array1d(dtype=wp.float32),
        sweep_steps: int,
        penalty_mode: int,
        frame_sphere_mask: wp.array3d(dtype=wp.bool),
        cache_world_center: wp.array1d(dtype=wp.vec3),
        cache_signed_dist_normal: wp.array1d(dtype=wp.vec4),
        cache_sweep_fraction: wp.array1d(dtype=wp.float32),
    ):
        batch_idx, frame_idx, sphere_local_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
        sphere_global_idx = sphere_indices[sphere_local_idx]
        link_idx = collision_spheres_link_indices[sphere_global_idx]
        local_center = local_collision_sphere_centers[sphere_global_idx]
        world_center = _transform_point_se3_func(T_world_link[batch_idx, frame_idx, link_idx], local_center)
        cache_idx = (batch_idx * T_world_link.shape[1] + frame_idx) * num_spheres + sphere_local_idx
        cache_world_center[cache_idx] = world_center  # a neighbouring frame's sweep may still start here
        mask = _frame_sphere_weight_func(
            frame_sphere_mask, batch_idx, frame_idx, sphere_local_idx, T_world_link.shape[0]
        )
        if mask == 0.0:  # a masked sphere skips its scene query: far from everything, no cost or gradient
            cache_signed_dist_normal[cache_idx] = wp.vec4(1.0e6, 0.0, 0.0, 0.0)
            cache_sweep_fraction[cache_idx] = 1.0
            return

        si = _scene_index_func(scene_indices, T_world_link.shape[0], batch_idx, frame_idx)
        hit = closest(query, world_center, si, True)
        # cache the largest-residual sample on the segment into this frame and where it lies on the segment
        sample = world_center
        fraction = float(1.0)
        if wp.static(kind == "primitive") and sweep_steps > 0 and frame_idx > 0:  # only primitive scenes sweep
            prev_center = _transform_point_se3_func(T_world_link[batch_idx, frame_idx - 1, link_idx], local_center)
            radius = collision_sphere_radii[sphere_global_idx]
            eta = _margin_func(
                goal_distance, goal_radius, sphere_margin, batch_idx, T_world_link.shape[0], sphere_local_idx
            )
            # sweep samples lie within the segment length of this frame, so a frame that far from the scene skips them
            reach = radius + eta + wp.length(world_center - prev_center)
            if penalty_mode == PENALTY_SURFACE_DISTANCE or hit.dist < reach:
                best = _compute_collision_penalty_func(hit.dist - radius, eta, penalty_mode)[0]
                for step in range(sweep_steps):
                    t = float(step + 1) / float(sweep_steps + 1)
                    p = prev_center + t * (world_center - prev_center)
                    candidate = closest(query, p, si, True)
                    residual = _compute_collision_penalty_func(candidate.dist - radius, eta, penalty_mode)[0]
                    if residual > best:
                        best = residual
                        hit = candidate
                        sample = p
                        fraction = t
        grad = _compute_cached_gradient_func(sample, hit.dist, hit.closest, hit.normal)
        cache_signed_dist_normal[cache_idx] = wp.vec4(hit.dist, grad[0], grad[1], grad[2])
        cache_sweep_fraction[cache_idx] = fraction

    return residual_kernel, jacobian_kernel


_QUERY_KERNELS = {kind: _build_query_kernels(kind) for kind in ("primitive", "mesh", "volume", "hybrid")}


@wp.func
def _eta_metric_func(sph_dist: float, eta: float, delta: wp.vec3) -> wp.vec4:
    """Cost and gradient of an eta-inflated sphere's penetration `sph_dist`: quadratic inside eta, then linear."""
    if sph_dist <= 0.0:
        return wp.vec4(0.0)
    if sph_dist > eta:
        return wp.vec4(sph_dist - 0.5 * eta, delta[0], delta[1], delta[2])
    scale = sph_dist / eta
    return wp.vec4(0.5 / eta * sph_dist * sph_dist, scale * delta[0], scale * delta[1], scale * delta[2])


@wp.func
def _box_hit_func(
    p: wp.vec3,
    prim_type: wp.int32,
    primitive_params: wp.array1d(dtype=wp.vec4),
    primitive_inv_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_poses: bool,
    primitive_scales: wp.array1d(dtype=wp.float32),
    enable_scales: bool,
    box: int,
) -> SdfHit:
    return closest_on_primitives(
        p,
        float(1e6),
        prim_type,
        primitive_params,
        primitive_inv_poses,
        enable_inv_poses,
        primitive_scales,
        enable_scales,
        box,
        box + 1,
        True,
    )


@wp.func
def _box_distance_func(
    p: wp.vec3,
    prim_type: wp.int32,
    primitive_params: wp.array1d(dtype=wp.vec4),
    primitive_inv_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_poses: bool,
    primitive_scales: wp.array1d(dtype=wp.float32),
    enable_scales: bool,
    box: int,
) -> float:
    """A box's signed distance alone, without its closest point; other primitives return -inf, so none is skipped."""
    if prim_type != PRIM_BOX:
        return -1.0e30
    q = p
    if enable_inv_poses:
        q = wp.transform_point(primitive_inv_poses[box], q)
    if enable_scales:
        q = q / primitive_scales[box]
    prm = primitive_params[box]
    d = wp.vec3(wp.abs(q[0]) - prm[0], wp.abs(q[1]) - prm[1], wp.abs(q[2]) - prm[2])
    inside = wp.min(wp.max(wp.max(d[0], d[1]), d[2]), 0.0)
    dist = wp.length(wp.vec3(wp.max(d[0], 0.0), wp.max(d[1], 0.0), wp.max(d[2], 0.0))) + inside
    if enable_scales:
        dist *= primitive_scales[box]
    return dist


@wp.func
def _box_sample_func(p: wp.vec3, w: float, eta: float, hit: SdfHit) -> wp.vec4:
    """One box, one sample: (cost, gradient) of a sphere of radius `w` at `p`, given the box's closest point."""
    grad_sd = _compute_cached_gradient_func(p, hit.dist, hit.closest, hit.normal)
    return _eta_metric_func(w - hit.dist, eta, -grad_sd)


@wp.kernel
def compute_traj_collision_jacobian_pattern_kernel(
    num_spheres: int,
    num_actuated: int,
    single_tangent_dim: int,
    base_dim: int,
    blocks: int,
    row_offset: int,
    nnz_offset: int,
    row_indices: wp.array1d(dtype=wp.int32),
    col_indices: wp.array1d(dtype=wp.int32),
):
    frame_idx, sphere_local_idx, col_grid_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    endpoint = col_grid_idx // single_tangent_dim
    col_local_idx = col_grid_idx % single_tangent_dim
    if frame_idx == 0 and endpoint > 0:
        return
    source_frame = frame_idx - endpoint

    residual_idx = row_offset + frame_idx * num_spheres + sphere_local_idx
    frame_block = frame_idx * blocks - int(blocks == 2 and frame_idx > 0)
    row_blocks = blocks if frame_idx > 0 else 1
    nnz_idx = (
        nnz_offset
        + (frame_block * num_spheres + sphere_local_idx * row_blocks + endpoint) * single_tangent_dim
        + col_local_idx
    )
    row_indices[nnz_idx] = residual_idx

    if col_local_idx < num_actuated:
        col_indices[nnz_idx] = source_frame * single_tangent_dim + base_dim + col_local_idx
    else:
        base_col = col_local_idx - num_actuated
        col_indices[nnz_idx] = source_frame * single_tangent_dim + base_col


@wp.func
def _sphere_point_velocity_func(
    T_world_base: wp.array2d(dtype=wp_vec7),
    S_world: wp.array4d(dtype=wp.float32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    col_joint_starts: wp.array1d(dtype=wp.int32),
    col_joint_indices: wp.array1d(dtype=wp.int32),
    col_joint_weights: wp.array1d(dtype=wp.float32),
    batch_idx: int,
    frame_idx: int,
    link_idx: int,
    col_local_idx: int,
    base_col: int,
    world_center: wp.vec3,
) -> wp.vec3:
    """The velocity of a point on `link_idx` at `world_center` per unit motion of one Jacobian column."""
    if base_col >= 0:  # floating base: a body-frame base twist, moved to the world frame
        unit = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        unit[base_col] = 1.0
        twist = se3_adjoint_func(T_world_base[batch_idx, frame_idx]) * unit
        return wp.vec3(twist[0], twist[1], twist[2]) + wp.cross(wp.vec3(twist[3], twist[4], twist[5]), world_center)
    linear = wp.vec3(0.0)
    angular = wp.vec3(0.0)
    # visit only joints mapped to this actuated column
    for k in range(col_joint_starts[col_local_idx], col_joint_starts[col_local_idx + 1]):
        joint_idx = col_joint_indices[k]
        if link_ancestor_joints_mask[link_idx, joint_idx]:
            weight = col_joint_weights[k]
            linear += weight * wp.vec3(
                S_world[batch_idx, frame_idx, 0, joint_idx],
                S_world[batch_idx, frame_idx, 1, joint_idx],
                S_world[batch_idx, frame_idx, 2, joint_idx],
            )
            angular += weight * wp.vec3(
                S_world[batch_idx, frame_idx, 3, joint_idx],
                S_world[batch_idx, frame_idx, 4, joint_idx],
                S_world[batch_idx, frame_idx, 5, joint_idx],
            )
    return linear + wp.cross(angular, world_center)


@wp.kernel
def compute_traj_collision_jacobian_cached_kernel(
    T_world_base: wp.array2d(dtype=wp_vec7),
    S_world: wp.array4d(dtype=wp.float32),
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    col_joint_starts: wp.array1d(dtype=wp.int32),
    col_joint_indices: wp.array1d(dtype=wp.int32),
    col_joint_weights: wp.array1d(dtype=wp.float32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    active_cols: wp.array1d(dtype=wp.int32),
    cache_world_center: wp.array1d(dtype=wp.vec3),
    cache_signed_dist_normal: wp.array1d(dtype=wp.vec4),
    cache_sweep_fraction: wp.array1d(dtype=wp.float32),
    goal_distance: wp.array1d(dtype=wp.float32),
    goal_radius: float,
    sphere_margin: wp.array1d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    frame_sphere_mask: wp.array3d(dtype=wp.bool),
    penalty_mode: wp.int32,
    num_spheres: int,
    num_actuated: int,
    n_active: int,
    base_dim: int,
    blocks: int,
    nnz_offset: int,
    values: wp.array2d(dtype=wp.float32),
):
    batch_idx, frame_idx, sphere_local_idx, col_grid_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    endpoint = col_grid_idx // (n_active + base_dim)
    col_grid_idx = col_grid_idx % (n_active + base_dim)
    if frame_idx == 0 and endpoint > 0:
        return
    source_frame = frame_idx - endpoint

    # map the active-column grid onto real Jacobian columns
    if col_grid_idx < n_active:
        col_local_idx = active_cols[col_grid_idx]
    else:
        col_local_idx = num_actuated + (col_grid_idx - n_active)

    sphere_global_idx = sphere_indices[sphere_local_idx]
    link_idx = collision_spheres_link_indices[sphere_global_idx]
    radius = collision_sphere_radii[sphere_global_idx]

    cache_idx = (batch_idx * S_world.shape[1] + frame_idx) * num_spheres + sphere_local_idx
    world_center = cache_world_center[(batch_idx * S_world.shape[1] + source_frame) * num_spheres + sphere_local_idx]
    signed_dist_normal = cache_signed_dist_normal[cache_idx]
    fraction = float(1.0)
    if blocks == 2:
        fraction = cache_sweep_fraction[cache_idx]
        if endpoint == 1:
            fraction = 1.0 - fraction

    signed_dist = signed_dist_normal[0]
    normal = wp.vec3(signed_dist_normal[1], signed_dist_normal[2], signed_dist_normal[3])

    eta = _margin_func(goal_distance, goal_radius, sphere_margin, batch_idx, S_world.shape[0], sphere_local_idx)
    dres_dsd = _compute_collision_penalty_func(signed_dist - radius, eta, penalty_mode)[1]

    cols_per_residual = num_actuated + base_dim
    frame_block = frame_idx * blocks - int(blocks == 2 and frame_idx > 0)
    row_blocks = blocks if frame_idx > 0 else 1
    nnz_idx = (
        nnz_offset
        + (frame_block * num_spheres + sphere_local_idx * row_blocks + endpoint) * cols_per_residual
        + col_local_idx
    )
    mask = _frame_sphere_weight_func(frame_sphere_mask, batch_idx, frame_idx, sphere_local_idx, S_world.shape[0])
    if dres_dsd == 0.0 or fraction == 0.0 or mask == 0.0:
        values[batch_idx, nnz_idx] = 0.0
        return

    point_velocity = _sphere_point_velocity_func(
        T_world_base,
        S_world,
        link_ancestor_joints_mask,
        col_joint_starts,
        col_joint_indices,
        col_joint_weights,
        batch_idx,
        source_frame,
        link_idx,
        col_local_idx,
        col_local_idx - num_actuated,
        world_center,
    )
    dsd_dv = wp.dot(normal, point_velocity)
    values[batch_idx, nnz_idx] = (
        mask * residual_weight[frame_idx * num_spheres + sphere_local_idx] * dres_dsd * dsd_dv * fraction
    )


_SWEPT_THREADS = 128  # CUDA threads per frame: sphere slots of _SWEPT_LANES lanes
_SWEPT_LANES = 2  # threads per sphere, each taking every other box
_MAX_JACOBIAN_COLS = 31  # the frame cost and every column share one 32-wide row sum
_frame_sums = wp.types.vector(length=32, dtype=wp.float32)


@wp.func
def _swept_sphere_cost_func(
    p0: wp.vec3,  # the sphere's centre at the previous frame, this frame and the next
    p1: wp.vec3,
    p2: wp.vec3,
    w: float,  # the sphere's radius plus the margin
    eta: float,
    back_dist: float,  # the gaps the sweep samples, zero when the neighbour's sphere already overlaps this one
    fwd_dist: float,
    prim_type: wp.int32,
    primitive_params: wp.array1d(dtype=wp.vec4),
    primitive_inv_poses: wp.array1d(dtype=wp.mat44),
    enable_inv_poses: bool,
    primitive_scales: wp.array1d(dtype=wp.float32),
    enable_scales: bool,
    first_box: int,
    end_box: int,
    box_step: int,
    sweep_steps: int,
) -> wp.vec4:
    """The swept cost of one sphere over `first_box` to `end_box`: (cost, gradient at `p1`), unweighted."""
    back_len = back_dist + 2.0 * w
    fwd_len = fwd_dist + 2.0 * w
    total = wp.vec4(0.0)
    # every sample lies within half a segment of p1, and a box at least w away contributes nothing: skip it
    reach = w + 0.5 * wp.max(back_len, fwd_len)
    for box in range(first_box, end_box, box_step):
        if (
            _box_distance_func(
                p1,
                prim_type,
                primitive_params,
                primitive_inv_poses,
                enable_inv_poses,
                primitive_scales,
                enable_scales,
                box,
            )
            >= reach
        ):
            continue
        hit = _box_hit_func(
            p1, prim_type, primitive_params, primitive_inv_poses, enable_inv_poses, primitive_scales, enable_scales, box
        )
        sum_pt = _box_sample_func(p1, w, eta, hit)
        jump_mid = float(0.0)
        if back_dist > 0.0 or fwd_dist > 0.0:
            jump_mid = wp.max(wp.abs(hit.dist) - w, w)
        for direction in range(2):  # sweep backwards, then forwards, preserving accumulation order
            gap = wp.where(direction == 0, back_dist, fwd_dist)
            length = wp.where(direction == 0, back_len, fwd_len)
            endpoint = wp.where(direction == 0, p0, p2)
            if gap > 0.0 and jump_mid < gap / 2.0:
                jump = jump_mid
                for _sample in range(sweep_steps):
                    if jump < length / 2.0:
                        k0 = 1.0 - jump / length
                        pi = k0 * p1 + (1.0 - k0) * endpoint
                        hit_i = _box_hit_func(
                            pi,
                            prim_type,
                            primitive_params,
                            primitive_inv_poses,
                            enable_inv_poses,
                            primitive_scales,
                            enable_scales,
                            box,
                        )
                        sum_pt += _box_sample_func(pi, w, eta, hit_i)
                        jump += wp.max(wp.abs(hit_i.dist), w)
        if sum_pt[0] > 0.0:
            total += sum_pt
    return total


@wp.func
def _speed_metric_func(cost: float, grad: wp.vec3, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, dt: float) -> wp.vec4:
    """CHOMP's speed metric: the cost and gradient along the path, scaled by the sphere's speed."""
    vel = (0.5 / dt) * (p2 - p0)
    speed = wp.length(vel)
    if speed < 0.001:
        return wp.vec4(cost, grad[0], grad[1], grad[2])
    acc = (1.0 / (dt * dt)) * (p0 + p2 - 2.0 * p1)
    v = vel / speed
    curvature = acc / (speed * speed)
    orth_grad = grad - wp.dot(v, grad) * v
    orth_curve = cost * (curvature - wp.dot(v, curvature) * v)
    scaled = speed * (orth_grad - orth_curve)
    return wp.vec4(speed * cost, scaled[0], scaled[1], scaled[2])


@lru_cache(maxsize=None)
def _build_swept_frame_kernel(
    n_cols: int, prim_type: int, enable_inv_poses: bool, enable_scales: bool, n_active: int
) -> wp.Kernel:
    """Compile geometry features and column layout with static accumulator indexing."""
    cols_vector = wp.types.vector(length=max(1, n_cols), dtype=wp.float32)

    @wp.kernel(module="unique")
    def swept_frame_kernel(
        sphere_centers: wp.array3d(dtype=wp.vec3),  # [batch, frames, spheres], world
        collision_sphere_radii: wp.array1d(dtype=wp.float32),
        sphere_indices: wp.array1d(dtype=wp.int32),
        primitive_params: wp.array1d(dtype=wp.vec4),
        primitive_inv_poses: wp.array1d(dtype=wp.mat44),
        primitive_scales: wp.array1d(dtype=wp.float32),
        primitive_offsets: wp.array1d(dtype=wp.int32),
        scene_indices: wp.array2d(dtype=wp.int32),  # (scenes, frames) or (scenes, 1); empty: one scene
        goal_distance: wp.array1d(dtype=wp.float32),
        goal_radius: float,
        sphere_margin: wp.array1d(dtype=wp.float32),
        speed_dt: wp.array1d(dtype=wp.float32),
        sweep_steps: int,
        T_world_base: wp.array2d(dtype=wp_vec7),
        S_world: wp.array4d(dtype=wp.float32),
        collision_spheres_link_indices: wp.array1d(dtype=wp.int32),
        link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
        col_joint_starts: wp.array1d(dtype=wp.int32),
        col_joint_indices: wp.array1d(dtype=wp.int32),
        col_joint_weights: wp.array1d(dtype=wp.float32),
        active_cols: wp.array1d(dtype=wp.int32),
        residual_weight: wp.array1d(dtype=wp.float32),
        frame_sphere_mask: wp.array3d(dtype=wp.bool),
        num_actuated: int,
        threads: int,  # per frame: _SWEPT_THREADS on CUDA, 1 on CPU
        lanes: int,  # per sphere: _SWEPT_LANES on CUDA, 1 on CPU
        row_offset: int,
        nnz_offset: int,
        out_residual: wp.array2d(dtype=wp.float32),
        values: wp.array2d(dtype=wp.float32),
    ):
        """Swept sphere/box costs: one residual per frame, 1 / num_spheres of Jacobian rows.

        Per-frame reduction:
        1. Assign a thread group with sphere slots of `lanes` threads; each thread visits every lanes-th box.
        2. Sum weighted sphere costs and Jacobian slopes, including the speed metric.
        3. Return sqrt(frame cost), matching the per-sphere formulation's cost and gradient.
        """
        tid = wp.tid()
        row, thread = tid // threads, tid % threads
        num_frames = sphere_centers.shape[1]
        num_spheres = sphere_indices.shape[0]
        cols_per_residual = num_actuated + n_cols - n_active  # the actuated joints plus any floating base
        batch_idx, frame_idx = row // num_frames, row % num_frames
        slots = threads // lanes
        slot, sub = thread // lanes, thread % lanes
        si = _scene_index_func(scene_indices, sphere_centers.shape[0], batch_idx, frame_idx)
        frame_cost = float(0.0)
        slopes = cols_vector()
        for first in range(0, num_spheres, slots):
            sphere_local_idx = wp.min(first + slot, num_spheres - 1)
            sphere_global_idx = sphere_indices[sphere_local_idx]
            weight = _frame_sphere_weight_func(
                frame_sphere_mask, batch_idx, frame_idx, sphere_local_idx, sphere_centers.shape[0]
            )
            weight *= residual_weight[frame_idx * num_spheres + sphere_local_idx]
            p1 = sphere_centers[batch_idx, frame_idx, sphere_global_idx]
            eta = _margin_func(
                goal_distance, goal_radius, sphere_margin, batch_idx, sphere_centers.shape[0], sphere_local_idx
            )
            w = collision_sphere_radii[sphere_global_idx] + eta
            p0 = p1
            p2 = p1
            back_dist = float(0.0)
            fwd_dist = float(0.0)
            if frame_idx > 0:
                p0 = sphere_centers[batch_idx, frame_idx - 1, sphere_global_idx]
                back_dist = wp.max(0.0, wp.length(p1 - p0) - 2.0 * w)
            if frame_idx < num_frames - 1:
                p2 = sphere_centers[batch_idx, frame_idx + 1, sphere_global_idx]
                fwd_dist = wp.max(0.0, wp.length(p2 - p1) - 2.0 * w)
            total = wp.vec4(0.0)
            if first + slot < num_spheres and weight != 0.0:
                total = _swept_sphere_cost_func(
                    p0,
                    p1,
                    p2,
                    w,
                    eta,
                    back_dist,
                    fwd_dist,
                    prim_type,
                    primitive_params,
                    primitive_inv_poses,
                    enable_inv_poses,
                    primitive_scales,
                    enable_scales,
                    primitive_offsets[si] + sub,
                    primitive_offsets[si + 1],
                    lanes,
                    sweep_steps,
                )
            # Every lane reaches the shuffle, including masked and padded sphere slots.
            for k in range(4):
                total[k] = group_sum(total[k], lanes)
            if sub != 0 or first + slot >= num_spheres or weight == 0.0:
                continue
            if total[0] > 0.0 and back_dist > 0.0 and fwd_dist > 0.0:
                dt = speed_dt[batch_idx // (wp.max(1, sphere_centers.shape[0] // speed_dt.shape[0]))]
                total = _speed_metric_func(total[0], wp.vec3(total[1], total[2], total[3]), p0, p1, p2, dt)
            grad = wp.vec3(total[1], total[2], total[3])
            cost = total[0]
            frame_cost += weight * weight * wp.max(cost, 0.0)
            if cost > 0.0:
                for col_grid_idx in range(wp.static(n_cols)):
                    col_local_idx = num_actuated + col_grid_idx - n_active
                    if col_grid_idx < n_active:
                        col_local_idx = active_cols[col_grid_idx]
                    point_velocity = _sphere_point_velocity_func(
                        T_world_base,
                        S_world,
                        link_ancestor_joints_mask,
                        col_joint_starts,
                        col_joint_indices,
                        col_joint_weights,
                        batch_idx,
                        frame_idx,
                        collision_spheres_link_indices[sphere_global_idx],
                        col_local_idx,
                        wp.where(col_grid_idx < n_active, -1, col_grid_idx - n_active),
                        p1,
                    )
                    slopes[col_grid_idx] += weight * weight * 0.5 * wp.dot(grad, point_velocity)
        if threads > 1:  # the frame's sums: lane k of every warp holds the sum of value k
            sums = _frame_sums()
            sums[0] = frame_cost
            for k in range(wp.static(n_cols)):
                sums[1 + k] = slopes[k]
            total_k = row_sum32(sums)
            frame_cost = warp_shuffle(total_k, 0)
            for k in range(wp.static(n_cols)):
                slopes[k] = warp_shuffle(total_k, 1 + k)
        if thread != 0:
            return
        residual = wp.sqrt(frame_cost)
        if out_residual.shape[0] > 0:
            out_residual[batch_idx, row_offset + frame_idx] = residual
        for col_grid_idx in range(wp.static(n_cols)):
            col_local_idx = num_actuated + col_grid_idx - n_active
            if col_grid_idx < n_active:
                col_local_idx = active_cols[col_grid_idx]
            values[batch_idx, nnz_offset + frame_idx * cols_per_residual + col_local_idx] = wp.where(
                residual > 0.0, slopes[col_grid_idx] / residual, 0.0
            )

    return cast(wp.Kernel, swept_frame_kernel)


@wp.kernel
def cached_residual_kernel(
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    cache_signed_dist_normal: wp.array1d(dtype=wp.vec4),
    goal_distance: wp.array1d(dtype=wp.float32),
    goal_radius: float,
    sphere_margin: wp.array1d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    frame_sphere_mask: wp.array3d(dtype=wp.bool),
    penalty_mode: wp.int32,
    num_frames: int,
    num_spheres: int,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    # reuse signed distances cached by the Jacobian pass
    batch_idx, frame_idx, sphere_local_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    sphere_global_idx = sphere_indices[sphere_local_idx]
    radius = collision_sphere_radii[sphere_global_idx]
    cache_idx = (batch_idx * num_frames + frame_idx) * num_spheres + sphere_local_idx
    signed_dist_normal = cache_signed_dist_normal[cache_idx]
    eta = _margin_func(goal_distance, goal_radius, sphere_margin, batch_idx, out_residual.shape[0], sphere_local_idx)
    residual = _compute_collision_penalty_func(signed_dist_normal[0] - radius, eta, penalty_mode)[0]
    residual_idx = frame_idx * num_spheres + sphere_local_idx
    mask = _frame_sphere_weight_func(frame_sphere_mask, batch_idx, frame_idx, sphere_local_idx, out_residual.shape[0])
    out_residual[batch_idx, row_offset + residual_idx] = mask * residual_weight[residual_idx] * residual


@wp.kernel
def max_penetration_kernel(
    collision_sphere_radii: wp.array1d(dtype=wp.float32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    sphere_mask: wp.array1d(dtype=wp.float32),
    frame_sphere_mask: wp.array3d(dtype=wp.bool),
    cache_sdf_normal: wp.array1d(dtype=wp.vec4),
    num_frames: int,
    num_spheres: int,
    out: wp.array1d(dtype=wp.float32),
):
    # max-reduce cached penetration after the fixed start while masking root spheres
    batch_idx, interior_idx, sphere_local_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    frame_idx = interior_idx + 1
    radius = collision_sphere_radii[sphere_indices[sphere_local_idx]]
    cache_idx = (batch_idx * num_frames + frame_idx) * num_spheres + sphere_local_idx
    batch_size = cache_sdf_normal.shape[0] // (num_frames * num_spheres)  # the cache's, not `out`'s
    mask = _frame_sphere_weight_func(frame_sphere_mask, batch_idx, frame_idx, sphere_local_idx, batch_size)
    pen = mask * sphere_mask[sphere_local_idx] * (radius - cache_sdf_normal[cache_idx][0])
    if pen > 0.0:
        wp.atomic_max(out, batch_idx, pen)


# --- task -------------------------------------------------------------------
class TrajectoryCollisionTask(RobotTask, SparseTask, GradientTask):
    """Evaluate trajectory-sphere costs against `geometry`, else every scene geometry; one row block per geometry.

    Swept-segment evaluation:
    1. Linearly interpolate sphere centers at `sweep_steps` samples between waypoints.
    2. Differentiate the largest residual through both endpoints.
    3. Include the final segment even if `free_goal_frame` disables the standalone goal waypoint.
    """

    def __init__(
        self,
        robot: Optional["Robot"] = None,  # noqa: F821
        scene: Optional[Union["WarpScene", Context]] = None,  # noqa: F821
        num_frames: Optional[int] = None,
        weight: Union[float, Sequence[float]] = 20.0,
        margin: Union[float, Sequence[float], np.ndarray] = 0.1,
        max_dist: float = 1e6,
        sphere_indices: Optional[Union[Sequence[int], np.ndarray, Context]] = None,
        scene_indices: Optional[Union[wp.array, Context]] = None,
        active_actuated_indices: Optional[Union[Sequence[int], np.ndarray]] = None,
        sweep_steps: int = 0,
        penalty: Literal["smooth", "plain", "surface_distance"] = "plain",
        geometry: Optional["BaseGeom"] = None,
        frame_sphere_mask: Optional[Union[np.ndarray, wp.array, Context]] = None,
        speed_dt: Optional[Union[wp.array, Context]] = None,  # set: scale the swept cost by each sphere's speed
        free_goal_frame: bool = False,
        name: str = "",
        goal_radius: float = 0.0,  # >0: the margin tapers inside this joint distance of each query's goal
        goal_distance: Optional[Union[wp.array, Context]] = None,  # each query's start-to-goal joint distance
    ):
        self.name = name
        self.goal_distance = goal_distance
        self.robot, self._scene = robot, scene
        if num_frames is not None:
            self.num_frames = num_frames
        self.weight, self.margin, self.max_dist = weight, margin, max_dist
        self.sweep_steps, self.penalty, self.free_goal_frame = sweep_steps, penalty, free_goal_frame
        self.speed_dt = speed_dt
        self.goal_radius = goal_radius
        self.scene_indices_arg = scene_indices
        self._geometry = geometry
        self._sphere_indices_np = sphere_indices
        self._active_cols_np = active_actuated_indices
        self._frame_sphere_mask_np = frame_sphere_mask
        if robot is None:
            return
        if scene is None or num_frames is None:
            raise ValueError("A built collision task requires a scene and num_frames.")
        if not robot.spec.has_collision_spheres:
            raise RuntimeError("Robot has no collision spheres. Load with load_collision_spheres=True.")
        if penalty not in {"smooth", "plain", "surface_distance"}:
            raise ValueError(f"Unsupported penalty: {penalty}")
        if goal_radius > 0.0 and not isinstance(goal_distance, wp.array):
            raise ValueError("goal_radius needs goal_distance, each query's start-to-goal joint distance.")
        if geometry is not None and not any(g is geometry for g in scene.geoms):
            raise ValueError(f"geometry {geometry!r} is not attached to the scene.")
        # geoms allocate fixed-capacity arrays at scene.add and update them in place, so pack them once
        self._geometries = [geometry] if geometry is not None else list(scene.geoms)
        self._kinds, self._queries = [], []
        for g in self._geometries:
            kind = "volume" if isinstance(g, VolumeGeom) else "primitive"
            if isinstance(g, MeshGeom):
                kind = "hybrid" if g.enable_sdf else "mesh"
            q = _SceneQuery()
            q.prim_type, q.params = g.TYPE_ID, g.params_wp
            q.mesh_ids, q.volume_ids = g.mesh_ids_wp, g.volume_ids_wp
            q.aabb_min, q.aabb_max = g.aabb_min_wp, g.aabb_max_wp
            q.paddings, q.refine_band = g.paddings_wp, g.refine_band
            if isinstance(g, VolumeGeom):
                q.observations = g.observations_wp
            q.inv_poses, q.enable_inv_poses, q.scales, q.enable_scales = (
                g.inv_poses_wp,
                g.enable_inv_poses,
                g.scales_wp,
                g.enable_scales,
            )
            q.offsets, q.max_dist = g.scene_offsets_wp, max_dist
            self._kinds.append(kind)
            self._queries.append(q)
        # set: the swept-sphere cost (box sum, adaptive sweep, speed metric) is primitive-only; else per-waypoint
        if "primitive" not in self._kinds:
            sweep_steps = self.sweep_steps = 0
            speed_dt = self.speed_dt = None
        elif speed_dt is not None or sweep_steps > 0:
            assert set(self._kinds) <= {"primitive"}, "the swept and speed-metric costs support primitive scenes only"

        num_total_spheres = int(robot.spec.local_collision_sphere_centers.shape[0])
        if sphere_indices is None:
            self._sphere_indices_np = np.arange(num_total_spheres, dtype=np.int32)
        else:
            self._sphere_indices_np = np.asarray(sphere_indices, dtype=np.int32)
        self.num_spheres = len(self._sphere_indices_np)

        # restrict Jacobian work to optimized actuated columns
        num_actuated = robot.spec.num_actuated_joints
        if active_actuated_indices is None:
            self._active_cols_np = np.arange(num_actuated, dtype=np.int32)
        else:
            self._active_cols_np = np.asarray(active_actuated_indices, dtype=np.int32)
        self._n_active = len(self._active_cols_np)

        # a sequence gives each selected sphere its own margin, e.g. 0 for feet standing on the floor
        self._sphere_margin_np = np.broadcast_to(np.asarray(margin, dtype=np.float32), (self.num_spheres,)).copy()
        residual_weight = np.ones(self.num_spheres, dtype=np.float32)
        if isinstance(weight, (float, int)):
            residual_weight[:] = float(weight)
        else:
            residual_weight[:] = np.asarray(weight, dtype=np.float32)
        self._residual_weight_np = np.tile(residual_weight, num_frames)
        if free_goal_frame and sweep_steps == 0:
            self._residual_weight_np[-self.num_spheres :] = 0.0
        mask = np.ones((1, num_frames, self.num_spheres), dtype=np.bool_)
        self._frame_sphere_mask_shared = frame_sphere_mask if isinstance(frame_sphere_mask, wp.array) else None
        if frame_sphere_mask is not None:
            mask = (
                frame_sphere_mask
                if self._frame_sphere_mask_shared is None
                else np.ones(frame_sphere_mask.shape, np.bool_)
            )
            mask = np.asarray(mask, dtype=np.bool_)
            if mask.ndim == 2:
                mask = mask[None]
            if mask.shape[1] != num_frames or mask.shape[2] not in (1, self.num_spheres):
                raise ValueError(
                    f"frame_sphere_mask must end with ({num_frames}, {self.num_spheres} or 1), got {mask.shape}."
                )
            mask = np.broadcast_to(mask, (len(mask), num_frames, self.num_spheres))  # one column covers every sphere
        self._frame_sphere_mask_np = np.ascontiguousarray(mask)
        self._frame_sphere_mask_arg: Optional[wp.array] = None

        self._scene_indices_arg = None if scene_indices is None else self._scene_rows(scene_indices)

        self.device: Optional[wp_device_type] = None
        self._sphere_indices_wp: Optional[wp.array] = None
        self.residual_weight = None
        self.frame_sphere_mask: Optional[wp.array] = None
        self._scene_indices_wp: Optional[wp.array] = None
        self._jac_cache_world_center: Optional[wp.array] = None
        self._jac_cache_signed_dist_normal: Optional[wp.array] = None
        self._jac_cache_sweep_fraction: Optional[wp.array] = None

    @property
    def active(self) -> bool:
        """Whether the term adds cost: nonzero weight, a scene, and at least one selected sphere."""
        spheres = self._sphere_indices_np
        return bool(np.any(self.weight)) and self._scene is not None and (spheres is None or np.size(spheres) > 0)

    def resolve(self, var_values: "VarValues", context: Mapping[Context, Any]):  # noqa: F821
        """Build for the robot and horizon in `var_values`, sharing a Warp `frame_sphere_mask`, unless inactive."""
        super().resolve(var_values, context)
        if not self.active:
            return
        state = var_values.get(self.var_key)
        TrajectoryCollisionTask.__init__(
            self,
            state.robot,
            self._scene,
            state.q.shape[1],
            weight=self.weight,
            margin=self.margin,
            max_dist=self.max_dist,
            sphere_indices=self._sphere_indices_np,
            scene_indices=self.scene_indices_arg,
            active_actuated_indices=self._active_cols_np,
            sweep_steps=self.sweep_steps,
            penalty=self.penalty,
            geometry=self._geometry,
            frame_sphere_mask=self._frame_sphere_mask_np,
            speed_dt=self.speed_dt,
            free_goal_frame=self.free_goal_frame,
            name=self.name,
            goal_radius=self.goal_radius,
            goal_distance=self.goal_distance,
        )

    @property
    def _penalty_mode(self) -> int:
        if self.penalty == "smooth":
            # this task is consumed as a residual, so it needs the root of the potential
            return PENALTY_SMOOTH_ROOT
        if self.penalty == "surface_distance":
            return PENALTY_SURFACE_DISTANCE
        return PENALTY_PLAIN

    def init_buffers(self, device: wp_device_type):
        """Build stable scene/sparsity parameters; query caches grow on first use.

        Lifecycle:
            1. Upload selected sphere and actuated-column indices.
            2. Build the actuated-column joint CSR mapping.
            3. Allocate scene/margin arrays and invalidate lazy query caches.
        """
        self.device = device
        self._sphere_indices_wp = wp.from_numpy(self._sphere_indices_np, dtype=wp.int32, device=device)
        self._active_cols_wp = wp.from_numpy(self._active_cols_np, dtype=wp.int32, device=device)
        # store nonzero joint mappings for each actuated column in CSR form
        jta = self.robot.spec.joints_to_actuated_mapping  # [num_joints, num_actuated], numpy
        starts = [0]
        joint_idx_list: List[int] = []
        weight_list: List[float] = []
        for a in range(jta.shape[1]):
            nz = np.nonzero(jta[:, a])[0]
            joint_idx_list.extend(int(j) for j in nz)
            weight_list.extend(float(jta[j, a]) for j in nz)
            starts.append(len(joint_idx_list))
        self._col_joint_starts_wp = wp.from_numpy(np.asarray(starts, dtype=np.int32), dtype=wp.int32, device=device)
        self._col_joint_indices_wp = wp.from_numpy(
            np.asarray(joint_idx_list, dtype=np.int32), dtype=wp.int32, device=device
        )
        self._col_joint_weights_wp = wp.from_numpy(
            np.asarray(weight_list, dtype=np.float32), dtype=wp.float32, device=device
        )
        self.residual_weight = wp.from_numpy(self._residual_weight_np, dtype=wp.float32, device=device)
        self._sphere_margin_wp = wp.from_numpy(self._sphere_margin_np, dtype=wp.float32, device=device)
        if self._frame_sphere_mask_shared is not None:
            self.frame_sphere_mask = self._frame_sphere_mask_shared
        elif self._frame_sphere_mask_arg is None:
            self.frame_sphere_mask = wp.from_numpy(self._frame_sphere_mask_np, dtype=wp.bool, device=device)
        else:
            if self._frame_sphere_mask_arg.device != wp.get_device(device):
                raise ValueError(
                    f"frame_sphere_mask must be on {wp.get_device(device)}, got {self._frame_sphere_mask_arg.device}."
                )
            self.frame_sphere_mask = wp.clone(self._frame_sphere_mask_arg)
        movable = self.robot.spec.collision_spheres_link_indices[self._sphere_indices_np] > 0
        self._pen_sphere_mask_wp = wp.from_numpy(movable.astype(np.float32), dtype=wp.float32, device=device)
        if self._scene_indices_arg is not None:
            if self._scene_indices_arg.device != wp.get_device(device):
                raise ValueError(
                    f"scene_indices must be on {wp.get_device(device)}, got {self._scene_indices_arg.device}."
                )
            self._scene_indices_wp = self._scene_indices_arg  # the caller's buffer, read directly
        self._cache_size = 0  # query caches grow on first use, to the runtime batch
        self._jac_cache_world_center = self._jac_cache_signed_dist_normal = self._jac_cache_sweep_fraction = None
        # the helper's buffer, read directly; without a taper the kernels never read it
        self._goal_distance_wp = (
            self.goal_distance if self.goal_radius > 0.0 else wp.zeros(1, dtype=wp.float32, device=device)
        )

    def accumulate_max_penetration(self, out: wp.array):
        """Max-accumulate penetration from the most recent Jacobian query cache, over every geometry.

        Runs over the batch the cache holds: `out` may be longer (a score task sized for every draw).
        """
        n = self._cache_size
        for k in range(len(self._kinds)):
            wp.launch(
                kernel=max_penetration_kernel,
                dim=(
                    n // (self.num_frames * self.num_spheres),
                    self.num_frames - 2 if self.free_goal_frame and self.sweep_steps == 0 else self.num_frames - 1,
                    self.num_spheres,
                ),
                inputs=[
                    self.robot.spec.get_tensors(str(self.device)).collision_sphere_radii,
                    self._sphere_indices_wp,
                    self._pen_sphere_mask_wp,
                    self.frame_sphere_mask,
                    self._jac_cache_signed_dist_normal[k * n : (k + 1) * n],
                    self.num_frames,
                    self.num_spheres,
                ],
                outputs=[out],
                device=self.device,
            )

    def set_frame_sphere_mask(self, frame_sphere_mask: wp.array):
        """Set the active `(batch, frame, selected sphere)` collision rows."""
        if frame_sphere_mask.dtype != wp.bool or frame_sphere_mask.ndim != 3:
            raise TypeError("frame_sphere_mask must be a 3D wp.bool array.")
        if frame_sphere_mask.shape[1] != self.num_frames or frame_sphere_mask.shape[2] not in (1, self.num_spheres):
            raise ValueError(
                f"frame_sphere_mask must end with ({self.num_frames}, 1 or {self.num_spheres}), "
                f"got {frame_sphere_mask.shape}."
            )
        self._frame_sphere_mask_arg = frame_sphere_mask
        if self.device is None:
            return
        if frame_sphere_mask.device != wp.get_device(self.device):
            raise ValueError(
                f"frame_sphere_mask must be on {wp.get_device(self.device)}, got {frame_sphere_mask.device}."
            )
        if frame_sphere_mask.shape != self.frame_sphere_mask.shape:
            self.frame_sphere_mask = wp.clone(frame_sphere_mask)
        else:
            wp.copy(self.frame_sphere_mask, frame_sphere_mask)

    @property
    def _jacobian_blocks(self) -> int:
        """Column blocks per swept row: this frame's, plus the segment start's when a plain sweep moves both."""
        return 2 if self.sweep_steps > 0 and self.speed_dt is None else 1

    @property
    def _block_rows(self) -> int:
        # the speed-metric cost is summed per frame: L-BFGS sees the same cost and gradient with fewer rows
        return self.num_frames * (1 if self.speed_dt is not None else self.num_spheres)

    @property
    def residual_dim(self) -> int:
        """One block of rows per geometry; an empty scene keeps one block of zeros."""
        return max(1, len(self._kinds)) * self._block_rows

    def _launch_speed_frames(
        self,
        robot_state: "RobotState",
        out_residual: Optional[wp.array],
        row_offset: int,
        values: Optional[wp.array],
        nnz_offset: int,
    ) -> None:
        """The per-frame speed-metric residual (`out_residual`) and/or Jacobian values (`values`), one launch."""
        if not robot_state.is_collision_spheres_computed:
            self.robot.transform_collision_spheres(robot_state)
        spec_tensors = robot_state.spec_tensors
        base_dim = 6 if bool(robot_state.has_floating_base) else 0
        n_cols = self._n_active + base_dim if values is not None else 0
        if n_cols > _MAX_JACOBIAN_COLS:
            raise ValueError(f"the speed-metric cost supports at most {_MAX_JACOBIAN_COLS} Jacobian columns")
        cuda = robot_state.q.device.is_cuda
        threads = _SWEPT_THREADS if cuda else 1
        block_nnz = self.num_frames * (self.robot.spec.num_actuated_joints + base_dim)
        for k, g in enumerate(self._geometries):
            wp.launch(
                kernel=_build_swept_frame_kernel(
                    n_cols, g.TYPE_ID, g.enable_inv_poses, g.enable_scales, self._n_active
                ),
                dim=int(robot_state.q.shape[0]) * self.num_frames * threads,
                inputs=[
                    robot_state.collision_sphere_centers_world,
                    spec_tensors.collision_sphere_radii,
                    self._sphere_indices_wp,
                    g.params_wp,
                    g.inv_poses_wp,
                    g.scales_wp,
                    g.scene_offsets_wp,
                    self._scene_indices_wp,
                    self._goal_distance_wp,
                    self.goal_radius,
                    self._sphere_margin_wp,
                    self.speed_dt,
                    self.sweep_steps,
                    robot_state.T_world_base,
                    robot_state.S_world if values is not None else None,
                    spec_tensors.collision_spheres_link_indices,
                    spec_tensors.link_ancestor_joints_mask,
                    self._col_joint_starts_wp,
                    self._col_joint_indices_wp,
                    self._col_joint_weights_wp,
                    self._active_cols_wp,
                    self.residual_weight,
                    self.frame_sphere_mask,
                    self.robot.spec.num_actuated_joints,
                    threads,
                    _SWEPT_LANES if cuda else 1,
                    row_offset + k * self._block_rows,
                    nnz_offset + k * block_nnz,
                ],
                outputs=[out_residual, values],
                block_dim=threads if cuda else 256,
                device=robot_state.q.device,
            )

    def _scene_rows(self, scene_indices: wp.array) -> wp.array:
        """Scenes shaped `(batch,)` or `(batch, frames)` as the kernels' `(batch, frames or 1)` rows."""
        if scene_indices.dtype != wp.int32 or scene_indices.ndim not in (1, 2):
            raise TypeError("scene_indices must be a 1D or 2D wp.int32 array.")
        if scene_indices.ndim == 2 and scene_indices.shape[1] != self.num_frames:
            raise ValueError(f"2D scene_indices must have shape (batch, {self.num_frames}).")
        return scene_indices.reshape((scene_indices.shape[0], scene_indices.size // scene_indices.shape[0]))

    def set_scene_indices(self, scene_indices: wp.array):
        """Set scenes shaped `(batch,)` or per-frame scenes shaped `(batch, frames)`."""
        rows = self._scene_rows(scene_indices)
        self._scene_indices_arg = rows
        if self.device is not None:
            if scene_indices.device != wp.get_device(self.device):
                raise ValueError(f"scene_indices must be on {wp.get_device(self.device)}, got {scene_indices.device}.")
            if self._scene_indices_wp is None or rows.shape != self._scene_indices_wp.shape:
                # resizing replaces the stable buffer; only safe before graph capture
                self._scene_indices_wp = wp.clone(rows)
            else:
                wp.copy(self._scene_indices_wp, rows)

    def compute_weighted_cost_and_gradient(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        out_cost: wp.array,
        out_gradient: Optional[wp.array] = None,
        **kwargs: object,
    ):
        """Compute sparse cost and optional gradient with one scene query.

        Lifecycle:
            1. Use the base cost-only path when no gradient is requested.
            2. Compute Jacobian values and cache their scene query.
            3. Rebuild residuals from the cache and reduce cost and gradient.
        """
        if out_gradient is None:
            sparse_cost_and_gradient(self, var_values, *args, out_cost=out_cost, **kwargs)
            return
        self._build_sparse_gradient_buffers(var_values)
        self.compute_weighted_sparse_jacobian_values(
            var_values,
            out_jacobian_values=self._sparse_jac_buf,
            nnz_offset=0,
            query_penetration=False,  # scoring refreshes penetration; the speed-metric gradient does not read it
            out_residual=self._sparse_residual_buf if self.speed_dt is not None else None,
        )
        if not self._kinds:
            self.compute_weighted_residual(var_values, out_residual=self._sparse_residual_buf, row_offset=0)
        elif self.speed_dt is None:
            self._cached_residuals(var_values, self._sparse_residual_buf, 0)
        wp.launch(
            kernel=_sparse_cost_deterministic_kernel,
            dim=var_values.batch_size,
            inputs=[self._sparse_residual_buf, out_cost, self.residual_dim],
            device=var_values.device,
        )
        col_offset = var_values.tangent_offset(self.var_key)
        wp.launch(
            kernel=_sparse_gradient_csc_kernel,
            dim=(var_values.batch_size, self._sparse_num_cols),
            inputs=[
                self._sparse_residual_buf,
                self._sparse_jac_buf,
                self._sparse_col_perm,
                self._sparse_col_nnz_starts,
                self._sparse_pattern.row_indices,
                col_offset,
            ],
            outputs=[out_gradient],
            device=var_values.device,
        )

    def fill_residual_and_jacobian(
        self,
        var_values: "VarValues",  # noqa: F821
        out_residual: wp.array,
        row_offset: int,
        out_jacobian: Optional[wp.array] = None,
        nnz_offset: int = 0,
    ):
        """Fill residuals and sparse values from one shared scene query."""
        if out_jacobian is None or not self._kinds:
            super().fill_residual_and_jacobian(
                var_values, out_residual, row_offset, out_jacobian=out_jacobian, nnz_offset=nnz_offset
            )
            return
        if self.speed_dt is not None:  # the per-frame residual comes with the Jacobian values
            self.compute_weighted_sparse_jacobian_values(
                var_values,
                out_jacobian_values=out_jacobian,
                nnz_offset=nnz_offset,
                query_penetration=False,
                out_residual=out_residual,
                row_offset=row_offset,
            )
            return
        self.compute_weighted_sparse_jacobian_values(
            var_values, out_jacobian_values=out_jacobian, nnz_offset=nnz_offset, query_penetration=False
        )
        self._cached_residuals(var_values, out_residual, row_offset)

    def _cached_residuals(self, var_values: "VarValues", out_residual: wp.array, row_offset: int):  # noqa: F821
        """Every geometry's residual rows from its block of the Jacobian query cache."""
        n = self._cache_size
        for k in range(len(self._kinds)):
            wp.launch(
                kernel=cached_residual_kernel,
                dim=(var_values.batch_size, self.num_frames, self.num_spheres),
                inputs=[
                    var_values.get(VarKey.ROBOT).spec_tensors.collision_sphere_radii,
                    self._sphere_indices_wp,
                    self._jac_cache_signed_dist_normal[k * n : (k + 1) * n],
                    self._goal_distance_wp,
                    self.goal_radius,
                    self._sphere_margin_wp,
                    self.residual_weight,
                    self.frame_sphere_mask,
                    self._penalty_mode,
                    self.num_frames,
                    self.num_spheres,
                    row_offset + k * self._block_rows,
                ],
                outputs=[out_residual],
                device=var_values.device,
            )

    def compute_weighted_residual(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        """Compute trajectory collision residuals.

        Lifecycle:
            1. Prepare forward kinematics and stable runtime buffers.
            2. Run each geometry's primitive, mesh, volume, or hybrid query kernel.
            3. Evaluate every frame and selected sphere.
        """
        robot_state = var_values.get(VarKey.ROBOT)
        if self.device is None:
            self.init_buffers(robot_state.q.device)
        if not robot_state.is_fk_computed:
            self.robot.forward_kinematics(robot_state)

        kernel_device = out_residual.device if out_residual is not None else self.device
        spec_tensors = robot_state.spec_tensors
        actual_batch = int(robot_state.q.shape[0])
        if out_residual is None:
            out_residual = wp.empty((actual_batch, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if not self._kinds:
            wp.launch(
                kernel=zero_collision_rows_kernel,
                dim=(actual_batch, self.residual_dim),
                inputs=[row_offset],
                outputs=[out_residual],
                device=kernel_device,
            )
            return out_residual

        if self.speed_dt is not None:
            self._launch_speed_frames(robot_state, out_residual, row_offset, None, 0)
            return out_residual
        for k, (kind, query) in enumerate(zip(self._kinds, self._queries)):
            wp.launch(
                kernel=_QUERY_KERNELS[kind][0],
                dim=(actual_batch, self.num_frames, self.num_spheres),
                inputs=[
                    robot_state.T_world_link,
                    spec_tensors.local_collision_sphere_centers,
                    spec_tensors.collision_sphere_radii,
                    spec_tensors.collision_spheres_link_indices,
                    self._sphere_indices_wp,
                    query,
                    self._scene_indices_wp,
                    self._goal_distance_wp,
                    self.goal_radius,
                    self._sphere_margin_wp,
                    self.residual_weight,
                    self.frame_sphere_mask,
                    self.sweep_steps,
                    self._penalty_mode,
                    row_offset + k * self._block_rows,
                    self.num_spheres,
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
        cols_per_residual = num_actuated + base_dim
        blocks = self._jacobian_blocks
        rows_per_frame = self._block_rows // self.num_frames
        block_nnz = (1 + (self.num_frames - 1) * blocks) * rows_per_frame * cols_per_residual
        nnz = max(1, len(self._kinds)) * block_nnz

        row_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_row_indices is None else out_row_indices
        )
        col_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_col_indices is None else out_col_indices
        )

        for k in range(max(1, len(self._kinds))):
            wp.launch(
                kernel=compute_traj_collision_jacobian_pattern_kernel,
                dim=(self.num_frames, rows_per_frame, cols_per_residual * blocks),
                inputs=[
                    rows_per_frame,
                    num_actuated,
                    cols_per_residual,
                    base_dim,
                    blocks,
                    offset + k * self._block_rows,
                    nnz_offset + k * block_nnz,
                ],
                outputs=[row_indices, col_indices],
                device=kernel_device,
            )

        pattern = SparsityPattern()
        pattern.row_indices = row_indices
        pattern.col_indices = col_indices
        return pattern

    def _query_collision_distances(self, var_values: "VarValues"):  # noqa: F821
        """Fill each geometry's sphere-distance cache for the Jacobian and `accumulate_max_penetration`."""
        robot_state = var_values.get(VarKey.ROBOT)
        if self.device is None:
            self.init_buffers(robot_state.q.device)
        if not robot_state.is_fk_computed:
            self.robot.forward_kinematics(robot_state)
        spec_tensors = robot_state.spec_tensors
        b, f, s = int(robot_state.q.shape[0]), self.num_frames, self.num_spheres
        n = self._cache_size = b * f * s
        if self._jac_cache_world_center is None or self._jac_cache_world_center.shape[0] < len(self._kinds) * n:
            device = robot_state.q.device
            self._jac_cache_world_center = wp.empty((len(self._kinds) * n,), dtype=wp.vec3, device=device)
            self._jac_cache_signed_dist_normal = wp.empty((len(self._kinds) * n,), dtype=wp.vec4, device=device)
            self._jac_cache_sweep_fraction = wp.empty((len(self._kinds) * n,), dtype=wp.float32, device=device)
        for k, (kind, query) in enumerate(zip(self._kinds, self._queries)):
            wp.launch(
                kernel=_QUERY_KERNELS[kind][1],
                dim=(b, f, s),
                inputs=[
                    robot_state.T_world_link,
                    spec_tensors.local_collision_sphere_centers,
                    spec_tensors.collision_sphere_radii,
                    spec_tensors.collision_spheres_link_indices,
                    self._sphere_indices_wp,
                    query,
                    self._scene_indices_wp,
                    self.num_spheres,
                    self._goal_distance_wp,
                    self.goal_radius,
                    self._sphere_margin_wp,
                    self.sweep_steps,
                    self._penalty_mode,
                    self.frame_sphere_mask,
                ],
                outputs=[
                    c[k * n : (k + 1) * n]
                    for c in (
                        self._jac_cache_world_center,
                        self._jac_cache_signed_dist_normal,
                        self._jac_cache_sweep_fraction,
                    )
                ],
                device=robot_state.q.device,
            )

    def compute_weighted_sparse_jacobian_values(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        out_jacobian_values: Optional[wp.array] = None,
        offset: int = 0,
        nnz_offset: int = 0,
        query_penetration: bool = True,  # False: skip the penetration query only scoring reads
        out_residual: Optional[wp.array] = None,  # speed metric: the residual too, from the same launch
        row_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        """Compute sparse trajectory collision Jacobian values.

        Lifecycle:
            1. Prepare kinematics, motion subspaces, and runtime-sized caches.
            2. Query each geometry and cache its sphere centers and SDF normals.
            3. Assemble active joint and optional floating-base columns.
        """
        robot_state = var_values.get(VarKey.ROBOT)
        if self.device is None:
            self.init_buffers(robot_state.q.device)
        if not robot_state.is_fk_computed:
            self.robot.forward_kinematics(robot_state)
        if not robot_state.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(robot_state)

        kernel_device = robot_state.q.device
        spec_tensors = robot_state.spec_tensors
        num_actuated = self.robot.spec.num_actuated_joints
        base_dim = 6 if bool(robot_state.has_floating_base) else 0
        cols_per_residual = num_actuated + base_dim
        blocks = self._jacobian_blocks
        block_nnz = (1 + (self.num_frames - 1) * blocks) * (self._block_rows // self.num_frames) * cols_per_residual
        nnz = max(1, len(self._kinds)) * block_nnz
        b, f, s = int(robot_state.q.shape[0]), self.num_frames, self.num_spheres

        if out_jacobian_values is None:
            out_jacobian_values = wp.empty((b, nnz), dtype=wp.float32, device=kernel_device)

        if not self._kinds:
            wp.launch(
                kernel=zero_collision_rows_kernel,
                dim=(b, nnz),
                inputs=[nnz_offset],
                outputs=[out_jacobian_values],
                device=kernel_device,
            )
            return out_jacobian_values
        # leave locked columns stale because the optimizer masks them after assembly

        if self.speed_dt is None or query_penetration:
            self._query_collision_distances(var_values)
        if self.speed_dt is not None:
            self._launch_speed_frames(robot_state, out_residual, row_offset, out_jacobian_values, nnz_offset)
            return out_jacobian_values
        n = self._cache_size
        for k in range(len(self._kinds)):
            caches = (self._jac_cache_world_center, self._jac_cache_signed_dist_normal, self._jac_cache_sweep_fraction)
            wp.launch(
                kernel=compute_traj_collision_jacobian_cached_kernel,
                dim=(b, f, s, (self._n_active + base_dim) * blocks),
                inputs=[
                    robot_state.T_world_base,
                    robot_state.S_world,
                    spec_tensors.collision_sphere_radii,
                    spec_tensors.collision_spheres_link_indices,
                    spec_tensors.link_ancestor_joints_mask,
                    self._col_joint_starts_wp,
                    self._col_joint_indices_wp,
                    self._col_joint_weights_wp,
                    self._sphere_indices_wp,
                    self._active_cols_wp,
                    *[c[k * n : (k + 1) * n] for c in caches],
                    self._goal_distance_wp,
                    self.goal_radius,
                    self._sphere_margin_wp,
                    self.residual_weight,
                    self.frame_sphere_mask,
                    self._penalty_mode,
                    self.num_spheres,
                    num_actuated,
                    self._n_active,
                    base_dim,
                    blocks,
                    nnz_offset + k * block_nnz,
                ],
                outputs=[out_jacobian_values],
                device=kernel_device,
            )
        return out_jacobian_values
