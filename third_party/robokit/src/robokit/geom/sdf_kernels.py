# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
from typing import Tuple

import warp as wp

from robokit.xform.warp.transforms import inverse_tf_mat_func, tf_mat_to_rot_mat_func


# --- segment closest-point utilities ---------------------------------------
_SEG_EPS = wp.constant(1.0e-6)


@wp.struct
class SegmentClosest:
    """Witness points of the closest approach between two segments."""

    c1: wp.vec3
    c2: wp.vec3


@wp.func
def closest_point_on_segment_func(a: wp.vec3, b: wp.vec3, p: wp.vec3) -> wp.vec3:
    """Closest point on segment `[a, b]` to point `p`."""
    ab = b - a
    t = wp.clamp(wp.dot(p - a, ab) / (wp.dot(ab, ab) + _SEG_EPS), 0.0, 1.0)
    return a + ab * t


@wp.func
def closest_segment_to_segment_func(a1: wp.vec3, b1: wp.vec3, a2: wp.vec3, b2: wp.vec3) -> SegmentClosest:
    """Closest points between segments `[a1, b1]` and `[a2, b2]` (Gram solve, then clamp to `[0, 1]` and refine)."""
    d1 = b1 - a1
    d2 = b2 - a2
    r = a1 - a2

    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)
    c = wp.dot(d1, r)
    b = wp.dot(d1, d2)
    denom = a * e - b * b

    s = wp.float32(0.0)
    t = wp.float32(0.0)
    if denom < _SEG_EPS:  # parallel / degenerate segments
        s = -c / (a + _SEG_EPS)
        t = f / (e + _SEG_EPS)
    else:
        s = (b * f - c * e) / denom
        t = (a * f - b * c) / denom

    s_clamped = wp.clamp(s, 0.0, 1.0)
    t_clamped = wp.clamp(t, 0.0, 1.0)

    t_recomp = wp.dot(d2, (a1 + d1 * s_clamped) - a2) / (e + _SEG_EPS)
    t_final = t_clamped
    if wp.abs(s - s_clamped) > _SEG_EPS:
        t_final = wp.clamp(t_recomp, 0.0, 1.0)

    s_recomp = wp.dot(d1, (a2 + d2 * t_final) - a1) / (a + _SEG_EPS)
    s_final = s_clamped
    if wp.abs(t - t_final) > _SEG_EPS:
        s_final = wp.clamp(s_recomp, 0.0, 1.0)

    out = SegmentClosest()
    out.c1 = a1 + d1 * s_final
    out.c2 = a2 + d2 * t_final
    return out


# --- per-geometry closest-point functions ----------------------------------
PRIM_BOX = wp.constant(0)
PRIM_SPHERE = wp.constant(1)
PRIM_PLANE = wp.constant(2)
PRIM_CAPSULE = wp.constant(3)


@wp.struct
class SdfHit:
    """Best hit for one point: dist, normal, closest (world) + local closest and element index; `index < 0` = none."""

    dist: wp.float32
    normal: wp.vec3
    normal_jacobian: wp.mat33
    closest: wp.vec3
    closest_local: wp.vec3
    index: wp.int32
    observed: wp.bool


@wp.struct
class VolumeObservations:
    """References to fusion weights and the live unknown-space policy; empty for mesh grids."""

    values: wp.array3d(dtype=wp.vec2)
    unknown_is_occupied: wp.array(dtype=wp.bool)


@wp.func
def normalized_vector_jacobian(vector: wp.vec3, jacobian: wp.mat33, epsilon: float) -> wp.mat33:
    length = wp.length(vector)
    if length <= epsilon:
        return wp.mat33(0.0)
    normal = vector / length
    return (wp.identity(n=3, dtype=wp.float32) - wp.outer(normal, normal)) * jacobian / length


@wp.func
def volume_normal_jacobian(
    volume_id: wp.uint64, point: wp.vec3, lower: wp.vec3, upper: wp.vec3, gradient: wp.vec3
) -> wp.mat33:
    """Differentiate the sampled gradient within its voxel cell, or the AABB normal outside."""
    if wp.min(point - lower) < 0.0 or wp.max(point - upper) > 0.0:
        closest = wp.vec3(
            wp.clamp(point[0], lower[0], upper[0]),
            wp.clamp(point[1], lower[1], upper[1]),
            wp.clamp(point[2], lower[2], upper[2]),
        )
        active = wp.vec3(
            float(point[0] < lower[0] or point[0] > upper[0]),
            float(point[1] < lower[1] or point[1] > upper[1]),
            float(point[2] < lower[2] or point[2] > upper[2]),
        )
        return normalized_vector_jacobian(point - closest, wp.diag(active), 1.0e-8)
    index = wp.volume_world_to_index(volume_id, point)
    base = wp.vec3(wp.floor(index[0]), wp.floor(index[1]), wp.floor(index[2]))
    fraction = index - base
    mixed = wp.vec3(0.0, 0.0, 0.0)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                value = wp.volume_lookup_f(volume_id, int(base[0]) + i, int(base[1]) + j, int(base[2]) + k)
                dx, dy, dz = float(2 * i - 1), float(2 * j - 1), float(2 * k - 1)
                wx = wp.where(i == 0, 1.0 - fraction[0], fraction[0])
                wy = wp.where(j == 0, 1.0 - fraction[1], fraction[1])
                wz = wp.where(k == 0, 1.0 - fraction[2], fraction[2])
                mixed += value * wp.vec3(dx * dy * wz, dx * wy * dz, wx * dy * dz)
    hessian = wp.mat33(0.0, mixed[0], mixed[1], mixed[0], 0.0, mixed[2], mixed[1], mixed[2], 0.0)
    x = wp.volume_world_to_index_dir(volume_id, wp.vec3(1.0, 0.0, 0.0))
    y = wp.volume_world_to_index_dir(volume_id, wp.vec3(0.0, 1.0, 0.0))
    z = wp.volume_world_to_index_dir(volume_id, wp.vec3(0.0, 0.0, 1.0))
    # Rows from scalars: warp >= 1.12 rejects matrices built from vectors.
    index_jacobian = wp.mat33(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2])
    return normalized_vector_jacobian(gradient, hessian * index_jacobian, 1.0e-8)


@wp.func
def closest_on_meshes(
    q_pt: wp.vec3,
    start_dist: wp.float32,
    mesh_ids: wp.array(dtype=wp.uint64),
    aabb_min: wp.array(dtype=wp.vec3),
    aabb_max: wp.array(dtype=wp.vec3),
    inv_poses: wp.array(dtype=wp.mat44),
    enable_inv_poses: bool,
    scales: wp.array(dtype=wp.float32),
    enable_scales: bool,
    begin: wp.int32,
    end: wp.int32,
    max_dist: wp.float32,
    compute_normal: bool,
) -> SdfHit:
    """Closest of meshes `[begin, end)`, starting from `start_dist`; AABBs cull the BVH walk (result unchanged)."""
    hit = SdfHit()
    hit.dist = start_dist
    hit.index = wp.int32(-1)
    for m_idx in range(begin, end):
        if enable_inv_poses:
            q_local = wp.transform_point(inv_poses[m_idx], q_pt)
        else:
            q_local = q_pt
        if enable_scales:
            q_local = q_local / scales[m_idx]

        aabb_lo = aabb_min[m_idx]
        aabb_hi = aabb_max[m_idx]
        aabb_dist = wp.length(
            wp.vec3(
                wp.max(wp.max(aabb_lo[0] - q_local[0], q_local[0] - aabb_hi[0]), 0.0),
                wp.max(wp.max(aabb_lo[1] - q_local[1], q_local[1] - aabb_hi[1]), 0.0),
                wp.max(wp.max(aabb_lo[2] - q_local[2], q_local[2] - aabb_hi[2]), 0.0),
            )
        )
        radius = max_dist
        if aabb_dist > 0.0:  # inside the AABB the signed distance is unbounded below - never cull
            scale = wp.where(enable_scales, scales[m_idx], wp.float32(1.0))
            if aabb_dist * scale >= hit.dist:
                continue
            radius = wp.min(hit.dist / scale, max_dist)

        query = wp.mesh_query_point(mesh_ids[m_idx], q_local, radius)
        if query.result:
            clst_local = wp.mesh_eval_position(mesh_ids[m_idx], query.face, query.u, query.v)
            unscaled = wp.length(clst_local - q_local) * query.sign
            if enable_scales:
                dist = unscaled * scales[m_idx]
            else:
                dist = unscaled
            if dist < hit.dist:
                hit.dist = dist
                hit.index = m_idx
                if compute_normal:
                    normal_local = wp.mesh_eval_face_normal(mesh_ids[m_idx], query.face)
                    if enable_scales:
                        clst = clst_local * scales[m_idx]
                    else:
                        clst = clst_local
                    if enable_inv_poses:
                        fwd = inverse_tf_mat_func(inv_poses[m_idx])
                        hit.closest = wp.transform_point(fwd, clst)
                        hit.normal = tf_mat_to_rot_mat_func(fwd) * normal_local
                    else:
                        hit.closest = clst
                        hit.normal = normal_local
                    hit.closest_local = clst_local
    return hit


@wp.func
def closest_on_primitives(
    q_pt: wp.vec3,
    start_dist: wp.float32,
    prim_type: wp.int32,
    params: wp.array(dtype=wp.vec4),
    inv_poses: wp.array(dtype=wp.mat44),
    enable_inv_poses: bool,
    scales: wp.array(dtype=wp.float32),
    enable_scales: bool,
    begin: wp.int32,
    end: wp.int32,
    compute_normal: bool,
    compute_normal_jacobian: bool = False,
) -> SdfHit:
    """Closest of primitives `[begin, end)`, starting from `start_dist` (shared with the trajectory kernel)."""
    hit = SdfHit()
    hit.dist = start_dist
    hit.index = wp.int32(-1)
    for p_idx in range(begin, end):
        if enable_inv_poses:
            q_local = wp.transform_point(inv_poses[p_idx], q_pt)
        else:
            q_local = q_pt
        if enable_scales:
            q_local = q_local / scales[p_idx]

        prm = params[p_idx]
        unscaled = wp.float32(0.0)
        normal_local = wp.vec3(0.0, 0.0, 1.0)
        clst_local = q_local
        normal_jacobian = wp.mat33(0.0)

        if prim_type == PRIM_BOX:
            half_extents = wp.vec3(prm[0], prm[1], prm[2])
            abs_p = wp.vec3(wp.abs(q_local[0]), wp.abs(q_local[1]), wp.abs(q_local[2]))
            q_vec = abs_p - half_extents
            q_max = wp.max(wp.max(q_vec[0], q_vec[1]), q_vec[2])
            q_pos = wp.vec3(wp.max(q_vec[0], 0.0), wp.max(q_vec[1], 0.0), wp.max(q_vec[2], 0.0))
            unscaled = wp.length(q_pos) + wp.min(q_max, 0.0)
            if q_max < 0.0:  # inside: normal of the nearest face
                if q_vec[0] >= q_vec[1] and q_vec[0] >= q_vec[2]:
                    normal_local = wp.vec3(wp.where(q_local[0] < 0.0, wp.float32(-1.0), wp.float32(1.0)), 0.0, 0.0)
                elif q_vec[1] >= q_vec[2]:
                    normal_local = wp.vec3(0.0, wp.where(q_local[1] < 0.0, wp.float32(-1.0), wp.float32(1.0)), 0.0)
                else:
                    normal_local = wp.vec3(0.0, 0.0, wp.where(q_local[2] < 0.0, wp.float32(-1.0), wp.float32(1.0)))
            else:
                clamped = wp.vec3(
                    wp.clamp(q_local[0], -half_extents[0], half_extents[0]),
                    wp.clamp(q_local[1], -half_extents[1], half_extents[1]),
                    wp.clamp(q_local[2], -half_extents[2], half_extents[2]),
                )
                diff = q_local - clamped
                dlen = wp.length(diff)
                if dlen > 1.0e-10:
                    normal_local = diff / dlen
                    if compute_normal_jacobian:
                        active = wp.vec3(float(q_vec[0] > 0.0), float(q_vec[1] > 0.0), float(q_vec[2] > 0.0))
                        normal_jacobian = normalized_vector_jacobian(diff, wp.diag(active), 1.0e-10)
            clst_local = q_local - unscaled * normal_local
        elif prim_type == PRIM_SPHERE:
            r = prm[0]
            qlen = wp.length(q_local)
            unscaled = qlen - r
            if qlen > 1.0e-10:
                normal_local = q_local / qlen
                if compute_normal_jacobian:
                    normal_jacobian = normalized_vector_jacobian(q_local, wp.identity(n=3, dtype=wp.float32), 1.0e-10)
            else:
                normal_local = wp.vec3(1.0, 0.0, 0.0)
            clst_local = normal_local * r
        elif prim_type == PRIM_PLANE:
            unscaled = q_local[2]
            normal_local = wp.vec3(0.0, 0.0, 1.0)
            clst_local = wp.vec3(q_local[0], q_local[1], 0.0)
        else:  # PRIM_CAPSULE
            r = prm[0]
            half_h = prm[1]
            t = wp.clamp(q_local[2], -half_h, half_h)
            axis_pt = wp.vec3(0.0, 0.0, t)
            diff = q_local - axis_pt
            dlen = wp.length(diff)
            unscaled = dlen - r
            if dlen > 1.0e-10:
                normal_local = diff / dlen
                if compute_normal_jacobian:
                    active = wp.vec3(1.0, 1.0, float(wp.abs(q_local[2]) > half_h))
                    normal_jacobian = normalized_vector_jacobian(diff, wp.diag(active), 1.0e-10)
            else:
                normal_local = wp.vec3(1.0, 0.0, 0.0)
            clst_local = axis_pt + normal_local * r

        if enable_scales:
            dist = unscaled * scales[p_idx]
        else:
            dist = unscaled
        if dist < hit.dist:
            hit.dist = dist
            hit.index = p_idx
            if compute_normal:
                if enable_scales:
                    clst = clst_local * scales[p_idx]
                else:
                    clst = clst_local
                if enable_inv_poses:
                    fwd = inverse_tf_mat_func(inv_poses[p_idx])
                    hit.closest = wp.transform_point(fwd, clst)
                    hit.normal = tf_mat_to_rot_mat_func(fwd) * normal_local
                else:
                    hit.closest = clst
                    hit.normal = normal_local
                hit.closest_local = clst_local
                if compute_normal_jacobian:
                    if enable_inv_poses:
                        rotation = tf_mat_to_rot_mat_func(fwd)
                        normal_jacobian = rotation * normal_jacobian * tf_mat_to_rot_mat_func(inv_poses[hit.index])
                    if enable_scales:
                        normal_jacobian = normal_jacobian / scales[hit.index]
                    hit.normal_jacobian = normal_jacobian
    return hit


@wp.func
def _sample_volume_sdf_func(
    point: wp.vec3,
    volume_id: wp.uint64,
    aabb_lo: wp.vec3,
    aabb_hi: wp.vec3,
    padding: float,
    observations: VolumeObservations,
) -> Tuple[wp.vec4, bool]:
    """Sample distance/gradient and observation status; unknown never bypasses obstacle checks."""
    grad_f = wp.vec3(0.0, 0.0, 0.0)
    observed = True
    has_observations = observations.values.shape[0] > 0
    inside_aabb = (
        point[0] >= aabb_lo[0]
        and point[0] <= aabb_hi[0]
        and point[1] >= aabb_lo[1]
        and point[1] <= aabb_hi[1]
        and point[2] >= aabb_lo[2]
        and point[2] <= aabb_hi[2]
    )
    if inside_aabb:
        uvw = wp.volume_world_to_index(volume_id, point)
        unscaled = wp.volume_sample_grad_f(volume_id, uvw, wp.Volume.LINEAR, grad_f)
        if has_observations:
            observed = observations.values[int(wp.rint(uvw[0])), int(wp.rint(uvw[1])), int(wp.rint(uvw[2]))][1] > 0.0
    else:
        closest_local = wp.vec3(
            wp.clamp(point[0], aabb_lo[0], aabb_hi[0]),
            wp.clamp(point[1], aabb_lo[1], aabb_hi[1]),
            wp.clamp(point[2], aabb_lo[2], aabb_hi[2]),
        )
        outside = point - closest_local
        aabb_dist = wp.length(outside)
        unscaled = aabb_dist + padding
        grad_f = outside / aabb_dist
        if has_observations:
            observed = False
            voxel = wp.length(wp.volume_index_to_world_dir(volume_id, wp.vec3(1.0, 0.0, 0.0)))
            # Keep checking known obstacles even when the query lies outside observation coverage.
            uvw = wp.volume_world_to_index(volume_id, closest_local)
            boundary_grad = wp.vec3(0.0, 0.0, 0.0)
            boundary = wp.volume_sample_grad_f(volume_id, uvw, wp.Volume.LINEAR, boundary_grad)
            center_bound = wp.max(boundary - 0.366025404 * voxel, 0.0)
            center_dist = wp.sqrt(aabb_dist * aabb_dist + center_bound * center_bound)
            unscaled = center_dist - 0.866025404 * voxel
            for axis in range(3):
                if outside[axis] != 0.0:
                    boundary_grad[axis] = 0.0
            grad_f = (outside + center_bound * boundary_grad) / center_dist
    block_unknown = False
    if has_observations:
        block_unknown = observations.unknown_is_occupied[0]
    if block_unknown:
        # Distance to unknown space outside the map also protects a sphere whose center is inside.
        clearance = float(1e20)
        inward = wp.vec3(0.0)
        for axis in range(3):
            if point[axis] - aabb_lo[axis] < clearance:
                clearance = point[axis] - aabb_lo[axis]
                inward = wp.vec3(0.0)
                inward[axis] = 1.0
            if aabb_hi[axis] - point[axis] < clearance:
                clearance = aabb_hi[axis] - point[axis]
                inward = wp.vec3(0.0)
                inward[axis] = -1.0
        if clearance < unscaled:
            unscaled, grad_f = clearance, inward
    return wp.vec4(unscaled, grad_f[0], grad_f[1], grad_f[2]), observed


@wp.func
def closest_on_volumes(
    q_pt: wp.vec3,
    start_dist: wp.float32,
    volume_ids: wp.array(dtype=wp.uint64),
    aabb_min: wp.array(dtype=wp.vec3),
    aabb_max: wp.array(dtype=wp.vec3),
    paddings: wp.array(dtype=wp.float32),
    observations: wp.array(dtype=VolumeObservations),
    inv_poses: wp.array(dtype=wp.mat44),
    enable_inv_poses: bool,
    scales: wp.array(dtype=wp.float32),
    enable_scales: bool,
    begin: wp.int32,
    end: wp.int32,
    compute_normal: bool,
    compute_normal_jacobian: bool = False,
) -> SdfHit:
    """Closest of SDF volumes `[begin, end)`, starting from `start_dist` (AABB-gated NanoVDB sample)."""
    hit = SdfHit()
    hit.dist = start_dist
    hit.index = wp.int32(-1)
    hit.observed = True
    for v_idx in range(begin, end):
        if enable_inv_poses:
            q_local = wp.transform_point(inv_poses[v_idx], q_pt)
        else:
            q_local = q_pt
        if enable_scales:
            q_local = q_local / scales[v_idx]

        aabb_lo = aabb_min[v_idx]
        aabb_hi = aabb_max[v_idx]
        sample, observed = _sample_volume_sdf_func(
            q_local,
            volume_ids[v_idx],
            aabb_lo,
            aabb_hi,
            paddings[v_idx],
            observations[v_idx],
        )
        hit.observed = hit.observed and observed
        unscaled = sample[0]
        grad_f = wp.vec3(sample[1], sample[2], sample[3])

        if enable_scales:
            dist = unscaled * scales[v_idx]
        else:
            dist = unscaled
        if dist < hit.dist:
            hit.dist = dist
            hit.index = v_idx
            normal_jacobian = wp.mat33(0.0)
            if compute_normal_jacobian:
                normal_jacobian = volume_normal_jacobian(volume_ids[v_idx], q_local, aabb_lo, aabb_hi, grad_f)
                if observations[v_idx].values.shape[0] > 0:
                    # Include changes of the active obstacle/unknown-boundary branch.
                    epsilon = wp.length(wp.volume_index_to_world_dir(volume_ids[v_idx], wp.vec3(1.0, 0.0, 0.0))) * 0.001
                    for axis in range(3):
                        delta = wp.vec3(0.0, 0.0, 0.0)
                        delta[axis] = epsilon
                        plus, _ = _sample_volume_sdf_func(
                            q_local + delta,
                            volume_ids[v_idx],
                            aabb_lo,
                            aabb_hi,
                            paddings[v_idx],
                            observations[v_idx],
                        )
                        minus, _ = _sample_volume_sdf_func(
                            q_local - delta,
                            volume_ids[v_idx],
                            aabb_lo,
                            aabb_hi,
                            paddings[v_idx],
                            observations[v_idx],
                        )
                        normal_plus = wp.normalize(wp.vec3(plus[1], plus[2], plus[3]))
                        normal_minus = wp.normalize(wp.vec3(minus[1], minus[2], minus[3]))
                        column = (normal_plus - normal_minus) / (2.0 * epsilon)
                        for row in range(3):
                            normal_jacobian[row, axis] = column[row]

            if compute_normal:
                grad_len = wp.length(grad_f)
                if grad_len > 1.0e-8:
                    normal_local = grad_f / grad_len
                else:
                    normal_local = wp.vec3(0.0, 0.0, 1.0)
                clst_local = q_local - unscaled * normal_local
                if enable_scales:
                    clst = clst_local * scales[v_idx]
                else:
                    clst = clst_local
                if enable_inv_poses:
                    fwd = inverse_tf_mat_func(inv_poses[v_idx])
                    hit.closest = wp.transform_point(fwd, clst)
                    hit.normal = tf_mat_to_rot_mat_func(fwd) * normal_local
                else:
                    hit.closest = clst
                    hit.normal = normal_local
                hit.closest_local = clst_local
                if compute_normal_jacobian:
                    if enable_inv_poses:
                        rotation = tf_mat_to_rot_mat_func(fwd)
                        normal_jacobian = rotation * normal_jacobian * tf_mat_to_rot_mat_func(inv_poses[hit.index])
                    if enable_scales:
                        normal_jacobian = normal_jacobian / scales[hit.index]
                    hit.normal_jacobian = normal_jacobian
    return hit


@wp.func
def closest_on_meshes_with_sdf(
    q_pt: wp.vec3,
    start_dist: wp.float32,
    mesh_ids: wp.array(dtype=wp.uint64),
    aabb_min: wp.array(dtype=wp.vec3),
    aabb_max: wp.array(dtype=wp.vec3),
    volume_ids: wp.array(dtype=wp.uint64),
    paddings: wp.array(dtype=wp.float32),
    refine_band: wp.float32,
    max_dist: wp.float32,
    inv_poses: wp.array(dtype=wp.mat44),
    enable_inv_poses: bool,
    scales: wp.array(dtype=wp.float32),
    enable_scales: bool,
    begin: wp.int32,
    end: wp.int32,
    compute_normal: bool,
    compute_normal_jacobian: bool = False,
) -> SdfHit:
    """LOD cascade for meshes with a baked SDF mid-tier (`volume_ids[i]` baked from `mesh_ids[i]`):

    outside the padded AABB       -> aabb_dist + padding
    inside                        -> O(1) volume sample
    within refine_band of surface -> exact mesh BVH
    """
    hit = SdfHit()
    hit.dist = start_dist
    hit.index = wp.int32(-1)
    for v_idx in range(begin, end):
        if enable_inv_poses:
            q_local = wp.transform_point(inv_poses[v_idx], q_pt)
        else:
            q_local = q_pt
        if enable_scales:
            q_local = q_local / scales[v_idx]

        aabb_lo = aabb_min[v_idx]
        aabb_hi = aabb_max[v_idx]
        grad_f = wp.vec3(0.0, 0.0, 0.0)
        inside_aabb = (
            q_local[0] >= aabb_lo[0]
            and q_local[0] <= aabb_hi[0]
            and q_local[1] >= aabb_lo[1]
            and q_local[1] <= aabb_hi[1]
            and q_local[2] >= aabb_lo[2]
            and q_local[2] <= aabb_hi[2]
        )
        if inside_aabb:
            uvw = wp.volume_world_to_index(volume_ids[v_idx], q_local)
            unscaled = wp.volume_sample_grad_f(volume_ids[v_idx], uvw, wp.Volume.LINEAR, grad_f)
        else:
            closest_local = wp.vec3(
                wp.clamp(q_local[0], aabb_lo[0], aabb_hi[0]),
                wp.clamp(q_local[1], aabb_lo[1], aabb_hi[1]),
                wp.clamp(q_local[2], aabb_lo[2], aabb_hi[2]),
            )
            outside = q_local - closest_local
            aabb_dist = wp.length(outside)
            unscaled = aabb_dist + paddings[v_idx]
            grad_f = outside / aabb_dist

        if enable_scales:
            vol_dist = unscaled * scales[v_idx]
        else:
            vol_dist = unscaled

        grad_len = wp.length(grad_f)
        if grad_len > 1.0e-8:
            normal_local = grad_f / grad_len
        else:
            normal_local = wp.vec3(0.0, 0.0, 1.0)
        clst_local = q_local - unscaled * normal_local
        use_dist = vol_dist
        used_mesh = False

        # gate the mesh BVH on this object's own volume reading (near-surface only)
        if vol_dist <= refine_band:
            query = wp.mesh_query_point(mesh_ids[v_idx], q_local, max_dist)
            if query.result:
                clst_mesh_local = wp.mesh_eval_position(mesh_ids[v_idx], query.face, query.u, query.v)
                unscaled_mesh = wp.length(clst_mesh_local - q_local) * query.sign
                if enable_scales:
                    mesh_dist = unscaled_mesh * scales[v_idx]
                else:
                    mesh_dist = unscaled_mesh
                use_dist = mesh_dist
                used_mesh = True
                clst_local = clst_mesh_local
                normal_local = wp.mesh_eval_face_normal(mesh_ids[v_idx], query.face)

        if use_dist < hit.dist:
            hit.dist = use_dist
            hit.index = v_idx
            normal_jacobian = wp.mat33(0.0)
            if compute_normal_jacobian and not used_mesh:
                normal_jacobian = volume_normal_jacobian(volume_ids[v_idx], q_local, aabb_lo, aabb_hi, grad_f)
            if compute_normal:
                if enable_scales:
                    clst = clst_local * scales[v_idx]
                else:
                    clst = clst_local
                if enable_inv_poses:
                    fwd = inverse_tf_mat_func(inv_poses[v_idx])
                    hit.closest = wp.transform_point(fwd, clst)
                    hit.normal = tf_mat_to_rot_mat_func(fwd) * normal_local
                else:
                    hit.closest = clst
                    hit.normal = normal_local
                hit.closest_local = clst_local
                if compute_normal_jacobian:
                    if enable_inv_poses:
                        rotation = tf_mat_to_rot_mat_func(fwd)
                        normal_jacobian = rotation * normal_jacobian * tf_mat_to_rot_mat_func(inv_poses[hit.index])
                    if enable_scales:
                        normal_jacobian = normal_jacobian / scales[hit.index]
                    hit.normal_jacobian = normal_jacobian
    return hit


# --- scene query kernels ---------------------------------------------------
@wp.kernel
def clear_sdf_query_kernel(
    distances: wp.array(dtype=float),
    observed: wp.array(dtype=wp.bool),
    query_mask: wp.array(dtype=int),
    max_dist: float,
):
    """Initialize both outputs in one capturable launch (bool fill is not capturable on older Warp)."""
    index = wp.tid()
    distances[index] = max_dist
    if observed.shape[0] > 0:
        observed[index] = True
        if query_mask.shape[0] > 0:
            observed[index] = query_mask[index] != 0


@wp.kernel
def build_scene_per_point_kernel(
    scene_offsets: wp.array(dtype=wp.int32),
    scene_per_point: wp.array(dtype=wp.int32),
):
    """For each point, binary-search CSR `scene_offsets` for its scene id (avoids a host sync)."""
    pt_idx = wp.tid()
    lo = wp.int32(0)
    hi = scene_offsets.shape[0] - wp.int32(2)
    while lo < hi:
        mid = (lo + hi + wp.int32(1)) // wp.int32(2)
        if scene_offsets[mid] <= pt_idx:
            lo = mid
        else:
            hi = mid - wp.int32(1)
    scene_per_point[pt_idx] = lo


@wp.kernel
def query_sdf_on_meshes_kernel(
    points: wp.array(dtype=wp.vec3),
    scene_per_point: wp.array(dtype=wp.int32),
    mesh_ids: wp.array(dtype=wp.uint64),
    mesh_aabb_min: wp.array(dtype=wp.vec3),
    mesh_aabb_max: wp.array(dtype=wp.vec3),
    scene_offsets: wp.array(dtype=wp.int32),
    inv_mesh_poses: wp.array(dtype=wp.mat44),
    enable_inv_mesh_poses: bool,
    mesh_scales: wp.array(dtype=wp.float32),
    enable_mesh_scales: bool,
    query_mask: wp.array(dtype=wp.int32),
    enable_query_mask: bool,
    enable_normals: bool,
    max_dist: float,
    signed_dists: wp.array(dtype=wp.float32),
    normals: wp.array(dtype=wp.vec3),
    closest_points: wp.array(dtype=wp.vec3),
    closest_points_in_mesh_coord: wp.array(dtype=wp.vec3),
    closest_mesh_indices: wp.array(dtype=wp.int32),
    enable_normal_jacobians: bool,
    normal_jacobians: wp.array(dtype=wp.mat33),
):
    pt_idx = wp.tid()
    if enable_query_mask:
        if query_mask[pt_idx] == 0:
            return
    batch_idx = scene_per_point[pt_idx]
    q_pt = points[pt_idx]

    m_begin_idx = scene_offsets[batch_idx]
    m_end_idx = scene_offsets[batch_idx + 1]

    # start from the current per-point min so geom kernels can run in any order
    hit = closest_on_meshes(
        q_pt,
        signed_dists[pt_idx],
        mesh_ids,
        mesh_aabb_min,
        mesh_aabb_max,
        inv_mesh_poses,
        enable_inv_mesh_poses,
        mesh_scales,
        enable_mesh_scales,
        m_begin_idx,
        m_end_idx,
        max_dist,
        enable_normals,
    )
    if hit.index >= 0:
        signed_dists[pt_idx] = hit.dist
        if enable_normals:
            normals[pt_idx] = hit.normal
            closest_points[pt_idx] = hit.closest
            if enable_inv_mesh_poses or enable_mesh_scales:
                closest_points_in_mesh_coord[pt_idx] = hit.closest_local
                closest_mesh_indices[pt_idx] = hit.index

        if enable_normal_jacobians:
            normal_jacobians[pt_idx] = hit.normal_jacobian


@wp.kernel
def query_sdf_on_volumes_kernel(
    points: wp.array(dtype=wp.vec3),
    scene_per_point: wp.array(dtype=wp.int32),
    volume_ids: wp.array(dtype=wp.uint64),
    volume_aabb_min: wp.array(dtype=wp.vec3),
    volume_aabb_max: wp.array(dtype=wp.vec3),
    volume_paddings: wp.array(dtype=wp.float32),
    observations: wp.array(dtype=VolumeObservations),
    scene_offsets: wp.array(dtype=wp.int32),
    inv_volume_poses: wp.array(dtype=wp.mat44),
    enable_inv_volume_poses: bool,
    volume_scales: wp.array(dtype=wp.float32),
    enable_volume_scales: bool,
    query_mask: wp.array(dtype=wp.int32),
    enable_query_mask: bool,
    enable_normals: bool,
    max_dist: float,
    signed_dists: wp.array(dtype=wp.float32),
    normals: wp.array(dtype=wp.vec3),
    closest_points: wp.array(dtype=wp.vec3),
    closest_points_in_volume_coord: wp.array(dtype=wp.vec3),
    closest_volume_indices: wp.array(dtype=wp.int32),
    enable_normal_jacobians: bool,
    normal_jacobians: wp.array(dtype=wp.mat33),
    observed: wp.array(dtype=wp.bool),
):
    pt_idx = wp.tid()
    if enable_query_mask:
        if query_mask[pt_idx] == 0:
            return
    scene_idx = scene_per_point[pt_idx]
    hit = closest_on_volumes(
        points[pt_idx],
        signed_dists[pt_idx],
        volume_ids,
        volume_aabb_min,
        volume_aabb_max,
        volume_paddings,
        observations,
        inv_volume_poses,
        enable_inv_volume_poses,
        volume_scales,
        enable_volume_scales,
        scene_offsets[scene_idx],
        scene_offsets[scene_idx + 1],
        enable_normals,
        enable_normal_jacobians,
    )
    if observed.shape[0] > 0:
        observed[pt_idx] = observed[pt_idx] and hit.observed
    if hit.index >= 0:
        signed_dists[pt_idx] = hit.dist
        if enable_normals:
            normals[pt_idx] = hit.normal
            closest_points[pt_idx] = hit.closest
            if enable_inv_volume_poses or enable_volume_scales:
                closest_points_in_volume_coord[pt_idx] = hit.closest_local
                closest_volume_indices[pt_idx] = hit.index
        if enable_normal_jacobians:
            normal_jacobians[pt_idx] = hit.normal_jacobian


@wp.kernel
def query_sdf_on_meshes_with_sdf_kernel(
    points: wp.array(dtype=wp.vec3),
    scene_per_point: wp.array(dtype=wp.int32),
    mesh_ids: wp.array(dtype=wp.uint64),
    mesh_aabb_min: wp.array(dtype=wp.vec3),
    mesh_aabb_max: wp.array(dtype=wp.vec3),
    volume_ids: wp.array(dtype=wp.uint64),
    volume_paddings: wp.array(dtype=wp.float32),
    refine_band: float,
    scene_offsets: wp.array(dtype=wp.int32),
    inv_mesh_poses: wp.array(dtype=wp.mat44),
    enable_inv_mesh_poses: bool,
    mesh_scales: wp.array(dtype=wp.float32),
    enable_mesh_scales: bool,
    query_mask: wp.array(dtype=wp.int32),
    enable_query_mask: bool,
    enable_normals: bool,
    max_dist: float,
    signed_dists: wp.array(dtype=wp.float32),
    normals: wp.array(dtype=wp.vec3),
    closest_points: wp.array(dtype=wp.vec3),
    closest_points_in_mesh_coord: wp.array(dtype=wp.vec3),
    closest_mesh_indices: wp.array(dtype=wp.int32),
    enable_normal_jacobians: bool,
    normal_jacobians: wp.array(dtype=wp.mat33),
):
    """Mesh geom with a baked SDF mid-tier; see `closest_on_meshes_with_sdf` for the cascade."""
    pt_idx = wp.tid()
    if enable_query_mask:
        if query_mask[pt_idx] == 0:
            return
    batch_idx = scene_per_point[pt_idx]
    q_pt = points[pt_idx]

    m_begin_idx = scene_offsets[batch_idx]
    m_end_idx = scene_offsets[batch_idx + 1]

    hit = closest_on_meshes_with_sdf(
        q_pt,
        signed_dists[pt_idx],
        mesh_ids,
        mesh_aabb_min,
        mesh_aabb_max,
        volume_ids,
        volume_paddings,
        refine_band,
        max_dist,
        inv_mesh_poses,
        enable_inv_mesh_poses,
        mesh_scales,
        enable_mesh_scales,
        m_begin_idx,
        m_end_idx,
        enable_normals,
        enable_normal_jacobians,
    )
    if hit.index >= 0:
        signed_dists[pt_idx] = hit.dist
        if enable_normals:
            normals[pt_idx] = hit.normal
            closest_points[pt_idx] = hit.closest
            if enable_inv_mesh_poses or enable_mesh_scales:
                closest_points_in_mesh_coord[pt_idx] = hit.closest_local
                closest_mesh_indices[pt_idx] = hit.index

        if enable_normal_jacobians:
            normal_jacobians[pt_idx] = hit.normal_jacobian


@wp.kernel
def query_sdf_on_primitives_kernel(
    points: wp.array(dtype=wp.vec3),
    scene_per_point: wp.array(dtype=wp.int32),
    prim_type: wp.int32,
    primitive_params: wp.array(dtype=wp.vec4),
    scene_offsets: wp.array(dtype=wp.int32),
    inv_primitive_poses: wp.array(dtype=wp.mat44),
    enable_inv_primitive_poses: bool,
    primitive_scales: wp.array(dtype=wp.float32),
    enable_primitive_scales: bool,
    query_mask: wp.array(dtype=wp.int32),
    enable_query_mask: bool,
    enable_normals: bool,
    max_dist: float,
    signed_dists: wp.array(dtype=wp.float32),
    normals: wp.array(dtype=wp.vec3),
    closest_points: wp.array(dtype=wp.vec3),
    closest_points_in_primitive_coord: wp.array(dtype=wp.vec3),
    closest_primitive_indices: wp.array(dtype=wp.int32),
    enable_normal_jacobians: bool,
    normal_jacobians: wp.array(dtype=wp.mat33),
):
    """Fused query over one primitive shape; params: BOX (hx,hy,hz) / SPHERE (r) / PLANE () / CAPSULE (r, half_h)."""
    pt_idx = wp.tid()
    if enable_query_mask:
        if query_mask[pt_idx] == 0:
            return
    batch_idx = scene_per_point[pt_idx]
    q_pt = points[pt_idx]

    p_begin_idx = scene_offsets[batch_idx]
    p_end_idx = scene_offsets[batch_idx + 1]

    hit = closest_on_primitives(
        q_pt,
        signed_dists[pt_idx],
        prim_type,
        primitive_params,
        inv_primitive_poses,
        enable_inv_primitive_poses,
        primitive_scales,
        enable_primitive_scales,
        p_begin_idx,
        p_end_idx,
        enable_normals,
        enable_normal_jacobians,
    )
    if hit.index >= 0:
        signed_dists[pt_idx] = hit.dist
        if enable_normals:
            normals[pt_idx] = hit.normal
            closest_points[pt_idx] = hit.closest
            if enable_inv_primitive_poses or enable_primitive_scales:
                closest_points_in_primitive_coord[pt_idx] = hit.closest_local
                closest_primitive_indices[pt_idx] = hit.index

        if enable_normal_jacobians:
            normal_jacobians[pt_idx] = hit.normal_jacobian
