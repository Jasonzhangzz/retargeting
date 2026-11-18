# pyright: reportArgumentType=false
# pyright: reportOperatorIssue=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
"""Shared sphere-collision penalty and residual kernel."""

import warp as wp


PENALTY_SMOOTH = wp.constant(wp.int32(0))
PENALTY_PLAIN = wp.constant(wp.int32(1))
PENALTY_SURFACE_DISTANCE = wp.constant(wp.int32(2))
# Use sqrt(PENALTY_SMOOTH) as a residual whose half-square gives the least-squares potential.
PENALTY_SMOOTH_ROOT = wp.constant(wp.int32(3))


# --- device code ------------------------------------------------------------
@wp.func
def _compute_collision_penalty_func(signed_dist: float, margin: float, penalty_mode: wp.int32) -> wp.vec2:
    """Return the collision residual and its derivative with respect to `signed_dist`."""
    if penalty_mode == PENALTY_SURFACE_DISTANCE:
        return wp.vec2(wp.abs(signed_dist), wp.sign(signed_dist))
    if penalty_mode == PENALTY_PLAIN:
        if signed_dist < margin:
            return wp.vec2(margin - signed_dist, wp.float32(-1.0))
        return wp.vec2(wp.float32(0.0), wp.float32(0.0))
    if penalty_mode == PENALTY_SMOOTH_ROOT:
        if signed_dist < wp.float32(0.0):
            root = wp.sqrt(wp.float32(0.5) * margin - signed_dist)
            return wp.vec2(root, wp.float32(-0.5) / root)
        if signed_dist <= margin:
            scale = wp.sqrt(wp.float32(0.5) / (margin + wp.float32(1e-6)))
            return wp.vec2(scale * (margin - signed_dist), -scale)
        return wp.vec2(wp.float32(0.0), wp.float32(0.0))
    if signed_dist < wp.float32(0.0):
        return wp.vec2(wp.float32(0.5) * margin - signed_dist, wp.float32(-1.0))
    if signed_dist <= margin:
        diff = signed_dist - margin
        inv_margin = wp.float32(1.0) / (margin + wp.float32(1e-6))
        return wp.vec2(wp.float32(0.5) * inv_margin * diff * diff, inv_margin * diff)
    return wp.vec2(wp.float32(0.0), wp.float32(0.0))


@wp.func
def sphere_to_scene_signed_dist_from_query(
    raw_signed_dists: wp.array2d(dtype=wp.float32),
    sphere_centers_world: wp.array2d(dtype=wp.vec3),
    closest_points: wp.array2d(dtype=wp.vec3),
    radii_global: wp.array1d(dtype=wp.float32),
    on_tape: wp.bool,
    instance: int,
    global_idx: int,
) -> float:
    """Sphere-to-scene signed distance: distance from center minus radius.

    Distance paths:
    - Normal: use the scene query's distance.
    - `on_tape`: rebuild `||center - closest_point||` so gradients reach centers despite the query having no adjoint.
    """
    signed_dist = raw_signed_dists[instance, global_idx]
    if on_tape:
        dist = wp.length(sphere_centers_world[instance, global_idx] - closest_points[instance, global_idx])
        signed_dist = wp.where(signed_dist < wp.float32(0.0), -dist, dist)
    return signed_dist - radii_global[global_idx]


@wp.kernel
def sphere_collision_residual_apply_kernel(
    raw_signed_dists: wp.array2d(dtype=wp.float32),
    sphere_centers_world: wp.array2d(dtype=wp.vec3),
    closest_points: wp.array2d(dtype=wp.vec3),
    sphere_indices: wp.array1d(dtype=wp.int32),
    radii_global: wp.array1d(dtype=wp.float32),
    on_tape: wp.bool,
    residual_weight: wp.array1d(dtype=wp.float32),
    activation_dist: float,
    use_sqrt: wp.bool,
    eps: float,
    penalty_mode: wp.int32,
    row_offset: int,
    out_residual: wp.array2d(dtype=wp.float32),
):
    instance, sphere_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    signed_dist = sphere_to_scene_signed_dist_from_query(
        raw_signed_dists,
        sphere_centers_world,
        closest_points,
        radii_global,
        on_tape,
        instance,
        sphere_indices[sphere_idx],
    )
    residual = _compute_collision_penalty_func(signed_dist, wp.float32(activation_dist), penalty_mode)[0]
    if use_sqrt:
        residual = wp.sqrt(residual + eps)
    out_residual[instance, row_offset + sphere_idx] = residual_weight[sphere_idx] * residual


__all__ = ["sphere_collision_residual_apply_kernel", "sphere_to_scene_signed_dist_from_query"]
