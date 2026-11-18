# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
"""Robot self-collision residuals using spheres or capsules."""

from typing import TYPE_CHECKING, Dict, Literal, Optional, Set, Tuple, Union

import numpy as np
import warp as wp

from robokit.geom.sdf_kernels import closest_segment_to_segment_func
from robokit.lie.se3_kernels import se3_adjoint_func
from robokit.opt.var_values import VarValues
from robokit.robo.robot_kernels import transform_link_points_kernel
from robokit.robo.robot_spec import JointType
from robokit.terms.robot_task import RobotTask
from robokit.terms.task import GradientTask, ResidualTask
from robokit.utils.warp_utils import wp_device_type, wp_vec6, wp_vec7


if TYPE_CHECKING:
    from robokit.robo.robot import Robot
    from robokit.robo.robot_spec import RobotSpec
    from robokit.robo.robot_state import RobotState


# --- public API -------------------------------------------------------------
def compute_active_collision_pairs(
    spec: "RobotSpec",
    ignored_link_pairs: Optional[Set[Tuple[int, int]]] = None,
    geom_link_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute non-rigid, non-adjacent collision-geometry pairs.

    Args:
        spec: Robot specification.
        ignored_link_pairs: Explicit link pairs to skip instead of adjacency filtering.
        geom_link_indices: Link index of each geometry. Defaults to the collision spheres' link indices; pass `arange(num_links)` for one-capsule-per-link.

    Returns:
        Geometry index pairs, shape `[num_pairs, 2]`.
    """
    sphere_link = spec.collision_spheres_link_indices if geom_link_indices is None else geom_link_indices
    num_links = len(spec.link_parent_joint_indices)
    if ignored_link_pairs is not None:
        skip = np.eye(num_links, dtype=bool)  # skip[a, b]: never test links a and b against each other
        for la, lb in ignored_link_pairs:
            skip[la, lb] = skip[lb, la] = True
    else:
        joint_types, parent_joints = spec.joint_types, spec.parent_joint_indices

        def _get_nonfixed_ancestor(j: int) -> int:
            while j >= 0 and joint_types[j] == JointType.FIXED:
                j = int(parent_joints[j])
            return j

        # links welded to the same moving body, or on bodies one joint apart, never collide
        body_of = np.array([_get_nonfixed_ancestor(int(j)) for j in spec.link_parent_joint_indices], dtype=np.int32)
        skip = body_of[:, None] == body_of[None, :]
        for j in range(len(joint_types)):
            if joint_types[j] != JointType.FIXED:
                child, parent = body_of == j, body_of == _get_nonfixed_ancestor(int(parent_joints[j]))
                skip |= np.outer(child, parent) | np.outer(parent, child)

    i, j = np.triu_indices(len(sphere_link), k=1)
    keep = ~skip[sphere_link[i], sphere_link[j]]
    return np.stack([i[keep], j[keep]], axis=1).astype(np.int32)


# --- device code ------------------------------------------------------------
@wp.kernel
def _compute_link_pair_penetration_kernel(
    link_bounding_sphere_centers_world: wp.array2d(dtype=wp.vec3),
    link_bounding_sphere_radii: wp.array1d(dtype=wp.float32),
    link_pair_indices: wp.array2d(dtype=wp.int32),
    margin: float,
    penetration_out: wp.array2d(dtype=wp.float32),
):
    instance, link_pair_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    link_a = link_pair_indices[link_pair_idx, 0]
    link_b = link_pair_indices[link_pair_idx, 1]
    center_a = link_bounding_sphere_centers_world[instance, link_a]
    center_b = link_bounding_sphere_centers_world[instance, link_b]
    dist = wp.length(center_a - center_b)
    p = link_bounding_sphere_radii[link_a] + link_bounding_sphere_radii[link_b] + wp.float32(margin) - dist
    penetration_out[instance, link_pair_idx] = p


@wp.kernel
def _reset_selection_kernel(
    max_k: int,
    selected_link_pair_out: wp.array2d(dtype=wp.int32),
    selected_count: wp.array1d(dtype=wp.int32),
):
    # reset one selection slot per thread
    instance, k = wp.tid()  # pyright: ignore[reportAssignmentType]
    selected_link_pair_out[instance, k] = -1
    if k == 0:
        selected_count[instance] = 0


@wp.func
def _compute_pair_shuffle_key_func(penetration: float, pair_idx: int) -> float:
    # keep selection repeatable for the same configuration
    return wp.randf(wp.rand_init(wp.int32(penetration * wp.float32(1000.0)), pair_idx))


@wp.kernel
def _select_active_link_pairs_kernel(
    link_pair_penetration: wp.array2d(dtype=wp.float32),
    max_k: int,
    selected_link_pair_out: wp.array2d(dtype=wp.int32),
    selected_count: wp.array1d(dtype=wp.int32),
):
    # let penetrating pairs claim available slots
    instance, l = wp.tid()  # pyright: ignore[reportAssignmentType]
    if link_pair_penetration[instance, l] <= wp.float32(0.0):
        return
    slot = wp.atomic_add(selected_count, instance, 1)
    if slot < max_k:
        selected_link_pair_out[instance, slot] = l


@wp.kernel
def _compute_selected_pair_keys_kernel(
    link_pair_penetration: wp.array2d(dtype=wp.float32),
    selected_link_pair: wp.array2d(dtype=wp.int32),
    keys: wp.array2d(dtype=wp.float32),
):
    instance, k = wp.tid()  # pyright: ignore[reportAssignmentType]
    l = selected_link_pair[instance, k]
    if l >= 0:
        keys[instance, k] = _compute_pair_shuffle_key_func(link_pair_penetration[instance, l], l)


@wp.kernel
def _sort_selected_link_pairs_kernel(
    selected_link_pair: wp.array2d(dtype=wp.int32),
    keys: wp.array2d(dtype=wp.float32),
    selected_count: wp.array1d(dtype=wp.int32),
    max_k: int,
    sorted_link_pair: wp.array2d(dtype=wp.int32),
):
    # remove nondeterministic atomic claim order: each claimed pair writes itself at its (key, pair) rank
    instance, i = wp.tid()  # pyright: ignore[reportAssignmentType]
    n = wp.min(selected_count[instance], max_k)
    if i >= n:
        sorted_link_pair[instance, i] = -1
        return
    l_i = selected_link_pair[instance, i]
    key_i = keys[instance, i]
    rank = int(0)
    for j in range(n):
        l_j = selected_link_pair[instance, j]
        key_j = keys[instance, j]
        if key_j < key_i or (key_j == key_i and l_j < l_i):
            rank += 1
    sorted_link_pair[instance, rank] = l_i


_PAIR_CHUNKS = 16  # the sphere pairs of one link pair are scanned in this many parallel chunks


@wp.kernel
def _find_chunk_most_penetrating_pair_kernel(
    selected_link_pair: wp.array2d(dtype=wp.int32),
    link_pair_offsets: wp.array1d(dtype=wp.int32),
    sorted_pair_indices: wp.array2d(dtype=wp.int32),
    world_centers: wp.array2d(dtype=wp.vec3),
    radii: wp.array1d(dtype=wp.float32),
    chunk_surface: wp.array3d(dtype=wp.float32),
    chunk_pair: wp.array3d(dtype=wp.int32),
):
    instance, k, c = wp.tid()  # pyright: ignore[reportAssignmentType]
    l = selected_link_pair[instance, k]
    best_surface = wp.float32(1.0e10)
    best_p = int(-1)
    if l >= 0:
        start = link_pair_offsets[l]
        end = link_pair_offsets[l + 1]
        size = (end - start + chunk_surface.shape[2] - 1) // chunk_surface.shape[2]
        for p in range(start + c * size, wp.min(start + (c + 1) * size, end)):
            i = sorted_pair_indices[p, 0]
            j = sorted_pair_indices[p, 1]
            surface = wp.length(world_centers[instance, i] - world_centers[instance, j]) - radii[i] - radii[j]
            if surface < best_surface:
                best_surface = surface
                best_p = p
    chunk_surface[instance, k, c] = best_surface
    chunk_pair[instance, k, c] = best_p


@wp.kernel
def _find_most_penetrating_sphere_pair_kernel(
    selected_link_pair: wp.array2d(dtype=wp.int32),
    link_pair_offsets: wp.array1d(dtype=wp.int32),
    sorted_pair_indices: wp.array2d(dtype=wp.int32),
    chunk_surface: wp.array3d(dtype=wp.float32),
    chunk_pair: wp.array3d(dtype=wp.int32),
    active_sphere_pairs_out: wp.array3d(dtype=wp.int32),
):
    # the first strict minimum over the chunks, in order, is the first strict minimum over all pairs
    instance, k = wp.tid()  # pyright: ignore[reportAssignmentType]
    l = selected_link_pair[instance, k]
    if l < 0:
        active_sphere_pairs_out[instance, k, 0] = -1
        active_sphere_pairs_out[instance, k, 1] = -1
        return
    best_p = link_pair_offsets[l]
    best_surface = wp.float32(1.0e10)
    for c in range(chunk_surface.shape[2]):
        if chunk_surface[instance, k, c] < best_surface:
            best_surface = chunk_surface[instance, k, c]
            best_p = chunk_pair[instance, k, c]
    active_sphere_pairs_out[instance, k, 0] = sorted_pair_indices[best_p, 0]
    active_sphere_pairs_out[instance, k, 1] = sorted_pair_indices[best_p, 1]


@wp.func
def _compute_self_collision_penalty_func(penetration: float, use_sqrt: wp.bool, eps: float) -> wp.vec2:
    residual = wp.max(wp.float32(0.0), penetration)  # pyright: ignore[reportCallIssue]
    derivative = wp.float32(0.0)
    if penetration > wp.float32(0.0):
        derivative = wp.float32(-1.0)
    if use_sqrt:
        residual = wp.sqrt(residual + eps)
        if penetration > wp.float32(0.0):
            derivative = wp.float32(-0.5) / residual
    return wp.vec2(residual, derivative)


@wp.kernel
def _compute_self_collision_residual_kernel(
    active_sphere_pairs: wp.array3d(dtype=wp.int32),
    world_centers: wp.array2d(dtype=wp.vec3),
    radii: wp.array1d(dtype=wp.float32),
    margin: float,
    weight: float,
    use_sqrt: wp.bool,
    eps: float,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    instance, k = wp.tid()  # pyright: ignore[reportAssignmentType]
    idx_i = active_sphere_pairs[instance, k, 0]
    if idx_i < 0:
        out_residual[instance, row_offset + k] = wp.float32(0.0)
        return
    idx_j = active_sphere_pairs[instance, k, 1]
    center_i = world_centers[instance, idx_i]
    center_j = world_centers[instance, idx_j]
    dist = wp.length(center_i - center_j)
    surface_dist = dist - radii[idx_i] - radii[idx_j]
    penalty = _compute_self_collision_penalty_func(wp.float32(margin) - surface_dist, use_sqrt, eps)
    out_residual[instance, row_offset + k] = weight * penalty[0]


@wp.func
def _accumulate_pair_jacobian_func(
    instance: int,
    row: int,
    point_i: wp.vec3,
    point_j: wp.vec3,
    link_i: int,
    link_j: int,
    normal: wp.vec3,
    weighted_scale: float,
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joint_actuated_col: wp.array1d(dtype=wp.int32),
    joint_actuated_weight: wp.array1d(dtype=wp.float32),
    has_floating_base: wp.bool,
    out_jacobian: wp.array3d(dtype=wp.float32),
):
    """Write one sphere or capsule pair's Jacobian row."""
    difference = point_i - point_j

    if has_floating_base:
        adj_mat = se3_adjoint_func(T_world_base[instance])
        for col_idx in range(6):
            unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            unit_vec[col_idx] = wp.float32(1.0)
            twist_world = adj_mat * unit_vec
            angular = wp.vec3(twist_world[3], twist_world[4], twist_world[5])
            relative_column = wp.cross(angular, difference)
            out_jacobian[instance, row, col_idx] = weighted_scale * wp.dot(normal, relative_column)  # pyright: ignore[reportCallIssue]

    base_dofs = int(0)
    if has_floating_base:
        base_dofs = int(6)

    num_joints = S_world.shape[2]
    for joint_idx in range(num_joints):
        actuated_idx = joint_actuated_col[joint_idx]
        if actuated_idx < 0:
            continue
        is_anc_i = link_ancestor_joints_mask[link_i, joint_idx]
        is_anc_j = link_ancestor_joints_mask[link_j, joint_idx]
        if (not is_anc_i) and (not is_anc_j):
            continue

        linear = wp.vec3(
            S_world[instance, 0, joint_idx], S_world[instance, 1, joint_idx], S_world[instance, 2, joint_idx]
        )
        angular = wp.vec3(
            S_world[instance, 3, joint_idx], S_world[instance, 4, joint_idx], S_world[instance, 5, joint_idx]
        )

        column_i = wp.vec3(0.0, 0.0, 0.0)
        column_j = wp.vec3(0.0, 0.0, 0.0)
        if is_anc_i:
            column_i = linear + wp.cross(angular, point_i)
        if is_anc_j:
            column_j = linear + wp.cross(angular, point_j)
        ddistance_dq = wp.dot(normal, column_i - column_j)  # pyright: ignore[reportCallIssue]

        col = base_dofs + actuated_idx
        value = weighted_scale * joint_actuated_weight[joint_idx] * ddistance_dq
        out_jacobian[instance, row, col] += value


@wp.kernel
def _compute_self_collision_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    world_centers: wp.array2d(dtype=wp.vec3),
    radii: wp.array1d(dtype=wp.float32),
    sphere_link_indices: wp.array1d(dtype=wp.int32),
    active_sphere_pairs: wp.array3d(dtype=wp.int32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joint_actuated_col: wp.array1d(dtype=wp.int32),
    joint_actuated_weight: wp.array1d(dtype=wp.float32),
    has_floating_base: wp.bool,
    margin: float,
    weight: float,
    use_sqrt: wp.bool,
    eps: float,
    row_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),
):
    instance, k = wp.tid()  # pyright: ignore[reportAssignmentType]
    idx_i = active_sphere_pairs[instance, k, 0]
    if idx_i < 0:
        return  # buffer is zero-init; nothing to write
    idx_j = active_sphere_pairs[instance, k, 1]

    center_i = world_centers[instance, idx_i]
    center_j = world_centers[instance, idx_j]
    diff = center_i - center_j
    dist = wp.length(diff)
    surface_dist = dist - radii[idx_i] - radii[idx_j]
    penetration = wp.float32(margin) - surface_dist

    if penetration <= wp.float32(0.0):
        return  # buffer is zero-init

    scale = _compute_self_collision_penalty_func(penetration, use_sqrt, eps)[1]

    denom = wp.max(wp.float32(1e-8), dist)
    normal = diff / denom
    weighted_scale = weight * scale

    row = row_offset + k
    link_i = sphere_link_indices[idx_i]
    link_j = sphere_link_indices[idx_j]
    _accumulate_pair_jacobian_func(
        instance,
        row,
        center_i,
        center_j,
        link_i,
        link_j,
        normal,
        weighted_scale,
        S_world,
        T_world_base,
        link_ancestor_joints_mask,
        joint_actuated_col,
        joint_actuated_weight,
        has_floating_base,
        out_jacobian,
    )


@wp.kernel
def _compute_capsule_link_pair_penetration_kernel(
    a_world: wp.array2d(dtype=wp.vec3),
    b_world: wp.array2d(dtype=wp.vec3),
    capsule_radii: wp.array1d(dtype=wp.float32),
    link_pair_indices: wp.array2d(dtype=wp.int32),
    margin: float,
    penetration_out: wp.array2d(dtype=wp.float32),
):
    # broad-phase capsule link pairs with bounding spheres
    instance, lp = wp.tid()  # pyright: ignore[reportAssignmentType]
    li = link_pair_indices[lp, 0]
    lj = link_pair_indices[lp, 1]
    ai = a_world[instance, li]
    bi = b_world[instance, li]
    aj = a_world[instance, lj]
    bj = b_world[instance, lj]
    center_i = wp.float32(0.5) * (ai + bi)
    center_j = wp.float32(0.5) * (aj + bj)
    bound_i = wp.float32(0.5) * wp.length(ai - bi) + capsule_radii[li]
    bound_j = wp.float32(0.5) * wp.length(aj - bj) + capsule_radii[lj]
    penetration_out[instance, lp] = bound_i + bound_j + wp.float32(margin) - wp.length(center_i - center_j)


@wp.kernel
def _compute_capsule_self_collision_residual_kernel(
    selected_link_pair: wp.array2d(dtype=wp.int32),
    link_pair_indices: wp.array2d(dtype=wp.int32),
    a_world: wp.array2d(dtype=wp.vec3),
    b_world: wp.array2d(dtype=wp.vec3),
    capsule_radii: wp.array1d(dtype=wp.float32),
    margin: float,
    weight: float,
    use_sqrt: wp.bool,
    eps: float,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    instance, k = wp.tid()  # pyright: ignore[reportAssignmentType]
    lp = selected_link_pair[instance, k]
    if lp < 0:
        out_residual[instance, row_offset + k] = wp.float32(0.0)
        return
    li = link_pair_indices[lp, 0]
    lj = link_pair_indices[lp, 1]
    cp = closest_segment_to_segment_func(
        a_world[instance, li], b_world[instance, li], a_world[instance, lj], b_world[instance, lj]
    )
    dist = wp.length(cp.c1 - cp.c2)
    surface_dist = dist - capsule_radii[li] - capsule_radii[lj]
    penalty = _compute_self_collision_penalty_func(wp.float32(margin) - surface_dist, use_sqrt, eps)
    out_residual[instance, row_offset + k] = weight * penalty[0]


@wp.kernel
def _compute_capsule_self_collision_jacobian_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    selected_link_pair: wp.array2d(dtype=wp.int32),
    link_pair_indices: wp.array2d(dtype=wp.int32),
    a_world: wp.array2d(dtype=wp.vec3),
    b_world: wp.array2d(dtype=wp.vec3),
    capsule_radii: wp.array1d(dtype=wp.float32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joint_actuated_col: wp.array1d(dtype=wp.int32),
    joint_actuated_weight: wp.array1d(dtype=wp.float32),
    has_floating_base: wp.bool,
    margin: float,
    weight: float,
    use_sqrt: wp.bool,
    eps: float,
    row_offset: int,
    out_jacobian: wp.array3d(dtype=wp.float32),
):
    instance, k = wp.tid()  # pyright: ignore[reportAssignmentType]
    lp = selected_link_pair[instance, k]
    if lp < 0:
        return  # buffer is zero-init; nothing to write
    link_i = link_pair_indices[lp, 0]
    link_j = link_pair_indices[lp, 1]

    # the envelope theorem gives the point-distance gradient at fixed witness points
    cp = closest_segment_to_segment_func(
        a_world[instance, link_i], b_world[instance, link_i], a_world[instance, link_j], b_world[instance, link_j]
    )
    center_i = cp.c1
    center_j = cp.c2
    diff = center_i - center_j
    dist = wp.length(diff)
    surface_dist = dist - capsule_radii[link_i] - capsule_radii[link_j]
    penetration = wp.float32(margin) - surface_dist

    if penetration <= wp.float32(0.0):
        return  # buffer is zero-init

    scale = _compute_self_collision_penalty_func(penetration, use_sqrt, eps)[1]

    denom = wp.max(wp.float32(1e-8), dist)
    normal = diff / denom
    weighted_scale = weight * scale

    _accumulate_pair_jacobian_func(
        instance,
        row_offset + k,
        center_i,
        center_j,
        link_i,
        link_j,
        normal,
        weighted_scale,
        S_world,
        T_world_base,
        link_ancestor_joints_mask,
        joint_actuated_col,
        joint_actuated_weight,
        has_floating_base,
        out_jacobian,
    )


@wp.kernel
def _compute_link_sphere_cost_and_gradient_kernel(
    S_world: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array1d(dtype=wp_vec7),
    world_centers: wp.array2d(dtype=wp.vec3),
    radii: wp.array1d(dtype=wp.float32),
    pair_indices: wp.array2d(dtype=wp.int32),
    link_ancestor_joints_mask: wp.array2d(dtype=wp.bool),
    joint_actuated_col: wp.array1d(dtype=wp.int32),
    joint_actuated_weight: wp.array1d(dtype=wp.float32),
    has_floating_base: wp.bool,
    margin: float,
    weight: float,
    use_sqrt: wp.bool,
    eps: float,
    col_offset: int,
    out_cost: wp.array1d(dtype=wp.float32),
    out_gradient: wp.array2d(dtype=wp.float32),
):
    instance, pair_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    link_i = pair_indices[pair_idx, 0]
    link_j = pair_indices[pair_idx, 1]
    point_i = world_centers[instance, link_i]
    point_j = world_centers[instance, link_j]
    diff = point_i - point_j
    dist = wp.length(diff)
    penetration = wp.float32(margin) - (dist - radii[link_i] - radii[link_j])

    if penetration <= wp.float32(0.0):
        return

    penalty = _compute_self_collision_penalty_func(penetration, use_sqrt, eps)
    r = weight * penalty[0]
    scale = penalty[1]

    wp.atomic_add(out_cost, instance, wp.float32(0.5) * r * r)

    denom = wp.max(wp.float32(1e-8), dist)
    normal = diff / denom
    contribution_factor = weight * scale * r

    base_dofs = wp.int32(0)
    if has_floating_base:
        base_dofs = wp.int32(6)
        adj_mat = se3_adjoint_func(T_world_base[instance])
        for col_idx in range(6):
            unit_vec = wp_vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            unit_vec[col_idx] = wp.float32(1.0)
            st = adj_mat * unit_vec
            ang = wp.vec3(st[3], st[4], st[5])
            rel_vel = wp.cross(ang, diff)
            wp.atomic_add(out_gradient, instance, col_offset + col_idx, contribution_factor * wp.dot(normal, rel_vel))

    num_joints = S_world.shape[2]
    for joint_idx in range(num_joints):
        actuated_idx = joint_actuated_col[joint_idx]
        if actuated_idx < 0:
            continue
        is_anc_i = link_ancestor_joints_mask[link_i, joint_idx]
        is_anc_j = link_ancestor_joints_mask[link_j, joint_idx]
        if (not is_anc_i) and (not is_anc_j):
            continue

        linear = wp.vec3(
            S_world[instance, 0, joint_idx], S_world[instance, 1, joint_idx], S_world[instance, 2, joint_idx]
        )
        angular = wp.vec3(
            S_world[instance, 3, joint_idx], S_world[instance, 4, joint_idx], S_world[instance, 5, joint_idx]
        )
        column_i = wp.vec3(0.0, 0.0, 0.0)
        column_j = wp.vec3(0.0, 0.0, 0.0)
        if is_anc_i:
            column_i = linear + wp.cross(angular, point_i)
        if is_anc_j:
            column_j = linear + wp.cross(angular, point_j)
        ddistance_dq = wp.dot(normal, column_i - column_j)
        col = base_dofs + actuated_idx
        wp.atomic_add(
            out_gradient,
            instance,
            col_offset + col,
            contribution_factor * joint_actuated_weight[joint_idx] * ddistance_dq,
        )


# --- task -------------------------------------------------------------------
class SelfCollisionTask(RobotTask, ResidualTask, GradientTask):
    """Penalize the most active robot self-collision pairs."""

    SUPPORTS_AUTODIFF = False  # the pair selection and the capsule kernel give the tape nothing usable

    def __init__(
        self,
        robot: Optional["Robot"] = None,
        weight: float = 1.0,
        margin: float = 0.02,
        representation: Literal["sphere", "capsule", "link_sphere"] = "sphere",
        # link sphere uses one synthesized sphere per link
        link_sphere_radius: Union[float, Literal["auto"]] = "auto",
        link_sphere_center: Literal["bounding", "origin"] = "bounding",
        residual_mode: Literal["abs", "sqrt_abs"] = "abs",
        residual_eps: float = 1e-6,
        filter_adjacent_links: bool = True,
        max_active_pairs: int = 50,
        name: str = "",
    ):
        self.name = name
        self.robot = robot
        self.weight = weight
        self.margin = margin
        self.representation = representation
        self.link_sphere_radius = link_sphere_radius
        self.link_sphere_center = link_sphere_center
        self.residual_mode = residual_mode
        self.residual_eps = residual_eps
        self.filter_adjacent_links = filter_adjacent_links
        self.max_active_pairs = max_active_pairs

        self._device: Optional[wp_device_type] = None
        self._pair_indices_np: Optional[np.ndarray] = None
        self._num_pairs = 0
        self._link_pair_indices_np: Optional[np.ndarray] = None
        self._link_pair_indices_wp: Optional[wp.array] = None
        self._num_link_pairs = 0
        self._link_pair_offsets_np: Optional[np.ndarray] = None
        self._link_pair_offsets_wp: Optional[wp.array] = None
        self._sorted_pair_indices_np: Optional[np.ndarray] = None
        self._sorted_pair_indices_wp: Optional[wp.array] = None
        self._link_pair_penetration_wp: Optional[wp.array] = None
        self._selected_link_pair_wp: Optional[wp.array] = None
        self._selected_count_wp: Optional[wp.array] = None
        self._selected_pair_keys_wp: Optional[wp.array] = None
        self._sorted_link_pair_wp: Optional[wp.array] = None
        self._chunk_surface_wp: Optional[wp.array] = None
        self._chunk_pair_wp: Optional[wp.array] = None
        self._active_sphere_pairs_wp: Optional[wp.array] = None
        self._joint_actuated_col_wp: Optional[wp.array] = None
        self._joint_actuated_weight_wp: Optional[wp.array] = None
        self._batch_buffers: Dict[int, Tuple[wp.array, ...]] = {}
        self._pair_indices_wp: Optional[wp.array] = None
        self._link_sphere_centers_local_np: Optional[np.ndarray] = None
        self._link_sphere_centers_local_wp: Optional[wp.array] = None
        self._link_sphere_radii_np: Optional[np.ndarray] = None
        self._link_sphere_radii_wp: Optional[wp.array] = None
        self._link_sphere_centers_world_wp: Optional[wp.array] = None
        self._world_batch = 0
        self._cost_grad_scratch_wp: Optional[wp.array] = None
        # reuse active pairs between residual and Jacobian calls at the same configuration
        self._selection_valid = False

        if self.robot is None:
            return  # config-time stub
        self.precompute_collision_geometry = self.representation if self.representation != "link_sphere" else ""
        if self.max_active_pairs < 1:
            raise ValueError("max_active_pairs must be >= 1.")
        spec = self.robot.spec

        if self.representation == "sphere":
            if not spec.has_collision_spheres:
                raise ValueError("Robot has no collision spheres.")
            if len(spec.local_collision_sphere_centers) < 2:
                raise ValueError("Expected at least two collision spheres.")
            geom_link = spec.collision_spheres_link_indices
            pairs = self._geom_pairs(spec, None)
        else:
            if self.representation == "capsule" and not spec.has_link_capsules:
                raise ValueError("Robot has no link capsules; load it with load_meshes=True.")
            if self.representation == "link_sphere" and not spec.has_collision_spheres:
                raise ValueError("Robot has no collision spheres (needed for link_sphere bounding geometry).")
            geom_link = np.arange(spec.num_links)
            pairs = self._geom_pairs(spec, geom_link)
            if self.representation == "capsule":
                # drop links without fitted capsules
                radii = spec.link_capsule_radii
                pairs = pairs[(radii[pairs[:, 0]] > 0.0) & (radii[pairs[:, 1]] > 0.0)]

        self._pair_indices_np = pairs.astype(np.int32)
        self._num_pairs = len(self._pair_indices_np)
        self._build_link_pair_csr(self._pair_indices_np, geom_link, spec.num_links)

        if self.representation == "link_sphere":
            if self.link_sphere_center == "bounding":
                centers = spec.local_link_bounding_sphere_centers.astype(np.float32)
            else:
                centers = np.zeros((spec.num_links, 3), dtype=np.float32)
            if self.link_sphere_radius == "auto":
                radii = spec.link_bounding_sphere_radii.astype(np.float32)
            else:
                radii = np.full(spec.num_links, float(self.link_sphere_radius), dtype=np.float32)
            self._link_sphere_centers_local_np = centers
            self._link_sphere_radii_np = radii

    def _geom_pairs(self, spec: "RobotSpec", geom_link_indices: Optional[np.ndarray]) -> np.ndarray:
        """Return candidate geometry pairs: adjacency filter (if on) composed with the spec ignore pairs."""
        # spec pairs only apply while filtering is on: filter_adjacent_links=False still means all pairs
        geom_link = spec.collision_spheres_link_indices if geom_link_indices is None else geom_link_indices
        if self.filter_adjacent_links:
            pairs = compute_active_collision_pairs(spec, geom_link_indices=geom_link_indices)
            ignored = {(int(a), int(b)) for a, b in spec.self_collision_ignored_pairs}
        else:
            i, j = np.triu_indices(len(geom_link), k=1)
            pairs = np.stack([i, j], axis=1).astype(np.int32)
            ignored = set()
        if ignored and len(pairs) > 0:
            ignored_both = ignored | {(j, i) for i, j in ignored}
            link_pairs = [(int(geom_link[i]), int(geom_link[j])) for i, j in pairs]
            keep = [li != lj and (li, lj) not in ignored_both for li, lj in link_pairs]
            pairs = pairs[np.asarray(keep, dtype=bool)]
        return pairs

    def _build_link_pair_csr(self, pairs: np.ndarray, geom_link: np.ndarray, num_links: int):
        """Group geometry pairs by link pair for narrow-phase selection."""
        if len(pairs) == 0:
            self._link_pair_indices_np = np.zeros((0, 2), dtype=np.int32)
            self._num_link_pairs = 0
            self._sorted_pair_indices_np = np.zeros((0, 2), dtype=np.int32)
            self._link_pair_offsets_np = np.zeros((1,), dtype=np.int32)
            return
        li = geom_link[pairs[:, 0]].astype(np.int64)
        lj = geom_link[pairs[:, 1]].astype(np.int64)
        key = np.minimum(li, lj) * num_links + np.maximum(li, lj)
        uniq_key, inverse = np.unique(key, return_inverse=True)
        link_pair_indices = np.empty((len(uniq_key), 2), dtype=np.int32)
        link_pair_indices[:, 0] = (uniq_key // num_links).astype(np.int32)
        link_pair_indices[:, 1] = (uniq_key % num_links).astype(np.int32)
        self._link_pair_indices_np = link_pair_indices
        self._num_link_pairs = len(link_pair_indices)
        order = np.argsort(inverse, kind="stable")
        self._sorted_pair_indices_np = pairs[order].astype(np.int32)
        counts = np.bincount(inverse, minlength=self._num_link_pairs)
        self._link_pair_offsets_np = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)

    def set_robot(self, robot: "Robot"):
        """Build a config-time stub's collision pairs with the configured settings."""
        SelfCollisionTask.__init__(
            self,
            robot,
            self.weight,
            self.margin,
            self.representation,
            self.link_sphere_radius,
            self.link_sphere_center,
            self.residual_mode,
            self.residual_eps,
            self.filter_adjacent_links,
            self.max_active_pairs,
        )

    def _init_buffers(self, device: wp_device_type):
        self._device = device
        self._link_pair_indices_wp = wp.from_numpy(self._link_pair_indices_np, dtype=wp.int32, device=device)
        self._link_pair_offsets_wp = wp.from_numpy(self._link_pair_offsets_np, dtype=wp.int32, device=device)
        self._sorted_pair_indices_wp = wp.from_numpy(self._sorted_pair_indices_np, dtype=wp.int32, device=device)

        if self._link_sphere_centers_local_np is not None:
            self._pair_indices_wp = wp.from_numpy(self._pair_indices_np, dtype=wp.int32, device=device)
            self._link_sphere_centers_local_wp = wp.from_numpy(
                self._link_sphere_centers_local_np, dtype=wp.vec3, device=device
            )
            self._link_sphere_radii_wp = wp.from_numpy(self._link_sphere_radii_np, dtype=wp.float32, device=device)

        # compact each joint's actuated column and mimic weight
        mapping = self.robot.spec.joints_to_actuated_mapping
        nonzero_mask = mapping != 0
        has_nz = nonzero_mask.any(axis=1)
        first_nz_col = np.argmax(nonzero_mask, axis=1)
        actuated_col = np.where(has_nz, first_nz_col, -1).astype(np.int32)
        actuated_weight = mapping[np.arange(mapping.shape[0]), first_nz_col].astype(np.float32)
        actuated_weight[~has_nz] = 0.0
        self._joint_actuated_col_wp = wp.from_numpy(actuated_col, dtype=wp.int32, device=device)
        self._joint_actuated_weight_wp = wp.from_numpy(actuated_weight, dtype=wp.float32, device=device)

    def _ensure_batch_buffers(self, batch_size: int, device: wp_device_type):
        # Cache each batch width; reallocating during capture frees buffers recorded by line search.
        if batch_size not in self._batch_buffers:
            L, K = max(1, self._num_link_pairs), self.max_active_pairs
            self._batch_buffers[batch_size] = (
                wp.empty((batch_size, L), dtype=wp.float32, device=device),
                wp.empty((batch_size, K), dtype=wp.int32, device=device),
                wp.empty((batch_size,), dtype=wp.int32, device=device),
                wp.empty((batch_size, K, 2), dtype=wp.int32, device=device),
                wp.empty((batch_size, K), dtype=wp.float32, device=device),
                wp.empty((batch_size, K), dtype=wp.int32, device=device),
                wp.empty((batch_size, K, _PAIR_CHUNKS), dtype=wp.float32, device=device),
                wp.empty((batch_size, K, _PAIR_CHUNKS), dtype=wp.int32, device=device),
            )
        (
            self._link_pair_penetration_wp,
            self._selected_link_pair_wp,
            self._selected_count_wp,
            self._active_sphere_pairs_wp,
            self._selected_pair_keys_wp,
            self._sorted_link_pair_wp,
            self._chunk_surface_wp,
            self._chunk_pair_wp,
        ) = self._batch_buffers[batch_size]

    def _select_link_pairs(self, batch_size: int, kernel_device: wp_device_type):
        """Select and sort the active link pairs."""
        wp.launch(
            kernel=_reset_selection_kernel,
            dim=(batch_size, self.max_active_pairs),
            inputs=[self.max_active_pairs],
            outputs=[self._selected_link_pair_wp, self._selected_count_wp],
            device=kernel_device,
        )
        wp.launch(
            kernel=_select_active_link_pairs_kernel,
            dim=(batch_size, self._num_link_pairs),
            inputs=[self._link_pair_penetration_wp, self.max_active_pairs],
            outputs=[self._selected_link_pair_wp, self._selected_count_wp],
            device=kernel_device,
        )
        wp.launch(
            kernel=_compute_selected_pair_keys_kernel,
            dim=(batch_size, self.max_active_pairs),
            inputs=[self._link_pair_penetration_wp, self._selected_link_pair_wp],
            outputs=[self._selected_pair_keys_wp],
            device=kernel_device,
        )
        wp.launch(
            kernel=_sort_selected_link_pairs_kernel,
            dim=(batch_size, self.max_active_pairs),
            inputs=[
                self._selected_link_pair_wp,
                self._selected_pair_keys_wp,
                self._selected_count_wp,
                self.max_active_pairs,
            ],
            outputs=[self._sorted_link_pair_wp],
            device=kernel_device,
        )
        wp.copy(self._selected_link_pair_wp, self._sorted_link_pair_wp)

    def _transform_link_spheres(self, var: "RobotState", kernel_device: wp_device_type):
        """Transform synthesized link spheres to the world frame."""
        batch_size = var.batch_size
        num_links = self.robot.spec.num_links
        if self._link_sphere_centers_world_wp is None or self._world_batch != batch_size:
            self._link_sphere_centers_world_wp = wp.empty((batch_size, num_links), dtype=wp.vec3, device=kernel_device)
            self._world_batch = batch_size
        wp.launch(
            kernel=transform_link_points_kernel,
            dim=(batch_size, num_links),
            inputs=[
                var.T_world_link.reshape((batch_size, num_links)),
                self._link_sphere_centers_local_wp,
                var.spec_tensors.link_identity_indices,
            ],
            outputs=[self._link_sphere_centers_world_wp],
            device=kernel_device,
        )

    def _prepare_sphere_geometry(
        self, var: "RobotState", kernel_device: wp_device_type
    ) -> Tuple[wp.array, wp.array, wp.array, wp.array, wp.array]:
        """Return narrow- and broad-phase sphere geometry."""
        spec_tensors = var.spec_tensors
        if self.representation == "link_sphere":
            self._transform_link_spheres(var, kernel_device)
            c, r = self._link_sphere_centers_world_wp, self._link_sphere_radii_wp
            assert c is not None and r is not None
            return c, r, spec_tensors.link_identity_indices, c, r
        if not var.is_collision_spheres_computed:
            self.robot.transform_collision_spheres(var)
        num_links = self.robot.spec.num_links
        return (
            var.collision_sphere_centers_world,
            spec_tensors.collision_sphere_radii,
            spec_tensors.collision_spheres_link_indices,
            var.link_bounding_sphere_centers_world.reshape((var.batch_size, num_links)),
            spec_tensors.link_bounding_sphere_radii,
        )

    def _compute_active_sphere_pairs(
        self,
        var: "RobotState",
        kernel_device: wp_device_type,
        reuse: bool,
        narrow_centers: wp.array,
        narrow_radii: wp.array,
        bound_centers: wp.array,
        bound_radii: wp.array,
    ) -> wp.array:
        """Return the selected sphere pairs, optionally reusing the prior selection."""
        if reuse and self._selection_valid:
            self._selection_valid = False
            assert self._active_sphere_pairs_wp is not None
            return self._active_sphere_pairs_wp

        batch_size = var.batch_size
        self._ensure_batch_buffers(batch_size, kernel_device)
        active_sphere_pairs = self._active_sphere_pairs_wp
        assert active_sphere_pairs is not None

        wp.launch(
            kernel=_compute_link_pair_penetration_kernel,
            dim=(batch_size, self._num_link_pairs),
            inputs=[bound_centers, bound_radii, self._link_pair_indices_wp, self.margin],
            outputs=[self._link_pair_penetration_wp],
            device=kernel_device,
        )
        self._select_link_pairs(batch_size, kernel_device)
        wp.launch(
            kernel=_find_chunk_most_penetrating_pair_kernel,
            dim=(batch_size, self.max_active_pairs, _PAIR_CHUNKS),
            inputs=[
                self._selected_link_pair_wp,
                self._link_pair_offsets_wp,
                self._sorted_pair_indices_wp,
                narrow_centers,
                narrow_radii,
            ],
            outputs=[self._chunk_surface_wp, self._chunk_pair_wp],
            device=kernel_device,
        )
        wp.launch(
            kernel=_find_most_penetrating_sphere_pair_kernel,
            dim=(batch_size, self.max_active_pairs),
            inputs=[
                self._selected_link_pair_wp,
                self._link_pair_offsets_wp,
                self._sorted_pair_indices_wp,
                self._chunk_surface_wp,
                self._chunk_pair_wp,
            ],
            outputs=[active_sphere_pairs],
            device=kernel_device,
        )
        self._selection_valid = True
        return active_sphere_pairs

    def _compute_active_link_pairs(self, var: "RobotState", kernel_device: wp_device_type, reuse: bool) -> wp.array:
        """Return the selected capsule link pairs, optionally reusing the prior selection."""
        if reuse and self._selection_valid:
            self._selection_valid = False
            assert self._selected_link_pair_wp is not None
            return self._selected_link_pair_wp

        batch_size = var.batch_size
        num_links = self.robot.spec.num_links
        spec_tensors = var.spec_tensors
        self._ensure_batch_buffers(batch_size, kernel_device)

        wp.launch(
            kernel=_compute_capsule_link_pair_penetration_kernel,
            dim=(batch_size, self._num_link_pairs),
            inputs=[
                var.link_capsule_endpoint_a_world.reshape((batch_size, num_links)),
                var.link_capsule_endpoint_b_world.reshape((batch_size, num_links)),
                spec_tensors.link_capsule_radii,
                self._link_pair_indices_wp,
                self.margin,
            ],
            outputs=[self._link_pair_penetration_wp],
            device=kernel_device,
        )
        self._select_link_pairs(batch_size, kernel_device)
        self._selection_valid = True
        assert self._selected_link_pair_wp is not None
        return self._selected_link_pair_wp

    @property
    def residual_dim(self) -> int:
        return self.max_active_pairs

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        """Compute self-collision residuals.

        Lifecycle:
            1. Prepare world collision geometry.
            2. Select the most active broad-phase link pairs.
            3. Evaluate the closest sphere or capsule pair for each slot.
        """
        var = var_values.get(self.var_key)
        if self._device is None:
            self._init_buffers(var.q.device)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)

        batch_size = var.batch_size
        kernel_device = out_residual.device if out_residual is not None else self._device
        spec_tensors = var.spec_tensors

        if out_residual is None:
            out_residual = wp.empty((batch_size, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        if self._num_pairs == 0 or self._num_link_pairs == 0:
            return out_residual

        use_sqrt = self.residual_mode == "sqrt_abs"
        num_links = self.robot.spec.num_links

        if self.representation == "capsule":
            if not var.is_capsules_computed:
                self.robot.transform_collision_capsules(var)
            selected_link_pair = self._compute_active_link_pairs(var, kernel_device, reuse=False)
            wp.launch(
                kernel=_compute_capsule_self_collision_residual_kernel,
                dim=(batch_size, self.max_active_pairs),
                inputs=[
                    selected_link_pair,
                    self._link_pair_indices_wp,
                    var.link_capsule_endpoint_a_world.reshape((batch_size, num_links)),
                    var.link_capsule_endpoint_b_world.reshape((batch_size, num_links)),
                    spec_tensors.link_capsule_radii,
                    self.margin,
                    self.weight,
                    use_sqrt,
                    self.residual_eps,
                    row_offset,
                ],
                outputs=[out_residual],
                device=kernel_device,
            )
            return out_residual

        narrow_centers, narrow_radii, _, bound_centers, bound_radii = self._prepare_sphere_geometry(var, kernel_device)
        active_sphere_pairs = self._compute_active_sphere_pairs(
            var, kernel_device, False, narrow_centers, narrow_radii, bound_centers, bound_radii
        )
        wp.launch(
            kernel=_compute_self_collision_residual_kernel,
            dim=(batch_size, self.max_active_pairs),
            inputs=[
                active_sphere_pairs,
                narrow_centers,
                narrow_radii,
                self.margin,
                self.weight,
                use_sqrt,
                self.residual_eps,
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
        """Compute the analytic self-collision Jacobian.

        Lifecycle:
            1. Prepare kinematics and motion subspaces.
            2. Reuse the residual's active-pair selection when available.
            3. Differentiate the selected sphere or capsule distances.
        """
        var = var_values.get(self.var_key)
        assert var_values.tangent_offset(self.var_key) == 0
        if self._device is None:
            self._init_buffers(var.q.device)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)

        batch_size = var.batch_size
        total_dofs = var.tangent_dim
        kernel_device = out_jacobian.device if out_jacobian is not None else self._device
        spec_tensors = var.spec_tensors

        if out_jacobian is None:
            out_jacobian = wp.zeros(
                (batch_size, self.residual_dim, total_dofs),
                dtype=wp.float32,
                device=kernel_device,
            )
            row_offset = 0

        if self._num_pairs == 0 or self._num_link_pairs == 0:
            return out_jacobian

        use_sqrt = self.residual_mode == "sqrt_abs"
        num_links = self.robot.spec.num_links

        if self.representation == "capsule":
            if not var.is_capsules_computed:
                self.robot.transform_collision_capsules(var)
            selected_link_pair = self._compute_active_link_pairs(var, kernel_device, reuse=True)
            wp.launch(
                kernel=_compute_capsule_self_collision_jacobian_kernel,
                dim=(batch_size, self.max_active_pairs),
                inputs=[
                    var.S_world,
                    var.T_world_base.flatten(),
                    selected_link_pair,
                    self._link_pair_indices_wp,
                    var.link_capsule_endpoint_a_world.reshape((batch_size, num_links)),
                    var.link_capsule_endpoint_b_world.reshape((batch_size, num_links)),
                    spec_tensors.link_capsule_radii,
                    spec_tensors.link_ancestor_joints_mask,
                    self._joint_actuated_col_wp,
                    self._joint_actuated_weight_wp,
                    var.has_floating_base,
                    self.margin,
                    self.weight,
                    use_sqrt,
                    self.residual_eps,
                    row_offset,
                ],
                outputs=[out_jacobian],
                device=kernel_device,
            )
            return out_jacobian

        narrow_centers, narrow_radii, link_indices, bound_centers, bound_radii = self._prepare_sphere_geometry(
            var, kernel_device
        )
        active_sphere_pairs = self._compute_active_sphere_pairs(
            var, kernel_device, True, narrow_centers, narrow_radii, bound_centers, bound_radii
        )
        wp.launch(
            kernel=_compute_self_collision_jacobian_kernel,
            dim=(batch_size, self.max_active_pairs),
            inputs=[
                var.S_world,
                var.T_world_base.flatten(),
                narrow_centers,
                narrow_radii,
                link_indices,
                active_sphere_pairs,
                spec_tensors.link_ancestor_joints_mask,
                self._joint_actuated_col_wp,
                self._joint_actuated_weight_wp,
                var.has_floating_base,
                self.margin,
                self.weight,
                use_sqrt,
                self.residual_eps,
                row_offset,
            ],
            outputs=[out_jacobian],
            device=kernel_device,
        )
        return out_jacobian

    def compute_weighted_cost_and_gradient(
        self,
        var_values: VarValues,
        out_cost: wp.array,
        out_gradient: Optional[wp.array] = None,
    ):
        """Compute fused link-sphere cost and optional gradient.

        Lifecycle:
            1. Prepare link spheres and motion subspaces.
            2. Allocate a scratch gradient for cost-only evaluations.
            3. Accumulate every active link-pair contribution.
        """
        if self.representation != "link_sphere":
            raise NotImplementedError("compute_weighted_cost_and_gradient is only implemented for link_sphere.")
        var = var_values.get(self.var_key)
        col_offset = var_values.tangent_offset(self.var_key)
        if self._device is None:
            self._init_buffers(var.q.device)
        if not var.is_fk_computed:
            self.robot.forward_kinematics(var)
        if not var.is_motion_subspace_computed:
            self.robot.compute_motion_subspace(var)
        if self._num_pairs == 0:
            return
        self._transform_link_spheres(var, self._device)

        # use full-width scratch storage because the fused kernel always writes gradients
        if out_gradient is None:
            width = col_offset + var.tangent_dim
            if self._cost_grad_scratch_wp is None or self._cost_grad_scratch_wp.shape != (var.batch_size, width):
                self._cost_grad_scratch_wp = wp.zeros((var.batch_size, width), dtype=wp.float32, device=self._device)
            out_gradient = self._cost_grad_scratch_wp

        use_sqrt = self.residual_mode == "sqrt_abs"
        spec_tensors = var.spec_tensors
        wp.launch(
            kernel=_compute_link_sphere_cost_and_gradient_kernel,
            dim=(var.batch_size, self._num_pairs),
            inputs=[
                var.S_world,
                var.T_world_base.flatten(),
                self._link_sphere_centers_world_wp,
                self._link_sphere_radii_wp,
                self._pair_indices_wp,
                spec_tensors.link_ancestor_joints_mask,
                self._joint_actuated_col_wp,
                self._joint_actuated_weight_wp,
                var.has_floating_base,
                self.margin,
                self.weight,
                use_sqrt,
                self.residual_eps,
                col_offset,
            ],
            outputs=[out_cost, out_gradient],
            device=self._device,
        )


__all__ = ["SelfCollisionTask", "compute_active_collision_pairs"]
