"""Full grasp objective and hard constraints, shared by CEM and MPPI."""

import numpy as np
import warp as wp
from retarget.task.dex import cost as fa
from .wrist_prior import WristPosePrior, wrist_pose_value


def force_priorities(forces, count):
    """Optional forces are magnitudes in N, in semantic-contact order."""
    if forces is None:
        return np.ones(count, np.float32)
    values = np.asarray(forces, dtype=float)
    if values.shape != (count,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Contact forces must be finite, nonnegative, one per contact")
    maximum = values.max(initial=0.0)
    return (
        (1.0 + 0.5 * values / maximum).astype(np.float32)
        if maximum > 0
        else np.ones(count, np.float32)
    )


@wp.struct
class SearchState:
    x: wp.array2d(dtype=float)
    qpos: wp.array2d(dtype=float)
    seed_qpos: wp.array(dtype=float)
    hand_ids: wp.array(dtype=int)
    seed: wp.array(dtype=float)
    low: wp.array(dtype=float)
    high: wp.array(dtype=float)
    initial_sigma: wp.array(dtype=float)
    scores: wp.array(dtype=float)
    selection_scores: wp.array(dtype=float)
    root_rotation: wp.mat33
    population: int


@wp.struct
class CostData:
    position: wp.array2d(dtype=wp.vec3)
    rotation: wp.array2d(dtype=wp.mat33)
    seed_position: wp.array(dtype=wp.vec3)
    seed_rotation: wp.array(dtype=wp.mat33)
    # All hand-surface samples, grouped by semantic link.
    sample_body: wp.array(dtype=int)
    sample_local: wp.array(dtype=wp.vec3)
    sample_normal: wp.array(dtype=wp.vec3)
    sample_pad: wp.array(dtype=int)
    group_body: wp.array(dtype=int)
    group_start: wp.array(dtype=int)
    group_end: wp.array(dtype=int)
    preference: wp.array(dtype=wp.vec3)
    previous_local: wp.array(dtype=wp.vec3)
    anchor: wp.array(dtype=wp.vec3)
    push: wp.array(dtype=wp.vec3)
    rho: wp.array(dtype=float)
    alpha: wp.array(dtype=float)
    center_weight: wp.array(dtype=float)
    selected: wp.array2d(dtype=int)
    parts: wp.array2d(dtype=float)
    pose_cost: wp.array(dtype=float)
    pose_enabled: int
    region_enabled: int
    object_axis_enabled: int
    object_axis_palm: wp.vec3
    object_axis_cosine: float
    wrist_prior: WristPosePrior
    wrist_prior_cost: wp.array(dtype=float)
    sample_forbidden: wp.array(dtype=int)
    forbidden_gap: wp.array(dtype=float)
    penetration: wp.array(dtype=float)
    palm_gap: wp.array(dtype=float)
    self_gap: wp.array(dtype=float)
    self_clearance: float
    object_mesh: wp.uint64
    object_body: int
    palm_mesh: wp.uint64
    palm_body: int
    style_body: wp.array(dtype=int)
    style_apex: wp.array(dtype=wp.vec3)
    style_normal: wp.array(dtype=wp.vec3)
    style_axis: wp.array(dtype=wp.vec3)
    capsule_body: wp.array(dtype=int)
    capsule_start: wp.array(dtype=wp.vec3)
    capsule_end: wp.array(dtype=wp.vec3)
    capsule_radius: wp.array(dtype=float)
    box_center: wp.array(dtype=wp.vec3)
    box_half: wp.array(dtype=wp.vec3)
    box_certificate: int
    pair_left: wp.array(dtype=int)
    pair_right: wp.array(dtype=int)


@wp.func
def world_point(d: CostData, w: int, b: int, local: wp.vec3):
    return d.position[w, b] + d.rotation[w, b] @ local


@wp.func
def seed_point(d: CostData, b: int, local: wp.vec3):
    return d.seed_position[b] + d.seed_rotation[b] @ local


@wp.kernel(enable_backward=False)
def clear_cost(d: CostData):
    w = wp.tid()
    for k in range(4):
        d.parts[w, k] = 0.0
    d.penetration[w] = 0.0
    d.palm_gap[w] = float(wp.inf)
    d.self_gap[w] = float(wp.inf)
    if d.region_enabled != 0:
        d.forbidden_gap[w] = float(wp.inf)


@wp.kernel(enable_backward=False)
def contact_cost(d: CostData):
    w, group = wp.tid()
    body = d.group_body[group]
    obj = d.object_body
    anchor = world_point(d, w, obj, d.anchor[group])
    push = d.rotation[w, obj] @ d.push[group]
    pref = world_point(d, w, body, d.preference[group])
    best = int(d.group_start[group])
    best_value = float(wp.inf)
    for i in range(d.group_start[group], d.group_end[group]):
        if d.sample_pad[i] != 0:
            point = world_point(d, w, body, d.sample_local[i])
            value = wp.length_sq(point - anchor) + d.alpha[group] * wp.length_sq(
                point - pref
            )
            if value < best_value:
                best_value = value
                best = i
    d.selected[w, group] = best
    point = world_point(d, w, body, d.sample_local[best])
    normal = d.rotation[w, body] @ d.sample_normal[best]
    # deltaT maps the previous selected pad point into the current link pose.
    previous = world_point(d, w, body, d.previous_local[group])
    reach = wp.length_sq((point - anchor) * 1000.0)
    cone = 2.0 * 100.0 * (1.0 - wp.clamp(wp.dot(normal, push), -1.0, 1.0))
    fix = wp.length_sq((point - previous) * 1000.0)
    value = d.rho[group] * d.rho[group] * (1600.0 * (reach + cone) + 6.25 * fix)
    value += (
        d.rho[group]
        * d.rho[group]
        * 1600.0
        * d.center_weight[group]
        * wp.length_sq((point - pref) * 1000.0)
    )
    wp.atomic_add(d.parts, w, 0, value)


@wp.kernel(enable_backward=False)
def style_cost(d: CostData):
    w, i = wp.tid()
    body = d.style_body[i]
    normal = d.style_normal[i]
    axis = d.style_axis[i]
    value = 400.0 * (
        1.0
        - wp.clamp(
            wp.dot(d.rotation[w, body] @ axis, d.seed_rotation[body] @ axis), -1.0, 1.0
        )
    )
    # Only distal regions carry apex and belly-normal style terms.
    if i % 3 == 2:
        delta = (
            world_point(d, w, body, d.style_apex[i])
            - seed_point(d, body, d.style_apex[i])
        ) * 1000.0
        value += 9.0 * wp.length_sq(delta)
        value += 3600.0 * (
            1.0
            - wp.clamp(
                wp.dot(d.rotation[w, body] @ normal, d.seed_rotation[body] @ normal),
                -1.0,
                1.0,
            )
        )
    wp.atomic_add(d.parts, w, 1, value)


@wp.kernel(enable_backward=False)
def smooth_cost(d: CostData, s: SearchState):
    w = wp.tid()
    value = float(0.0)
    for j in range(s.seed.shape[0]):
        delta = s.x[w, j] - s.seed[j]
        if j < 3:
            value += 125.0 * delta * delta * 1.0e6
        elif j < 6:
            value += 100.0 * delta * delta
        elif j >= 26:
            value += 0.01 * delta * delta
        else:
            value += 0.01 * delta * delta
    d.parts[w, 2] = value


@wp.kernel(enable_backward=False)
def wrist_reference_cost(d: CostData):
    w = wp.tid()
    d.wrist_prior_cost[w] = wrist_pose_value(
        d.wrist_prior,
        d.position[w, d.palm_body],
        d.rotation[w, d.palm_body],
        d.position[w, d.object_body],
        d.rotation[w, d.object_body],
    )


@wp.kernel(enable_backward=False)
def object_cost(d: CostData):
    w, i = wp.tid()
    point = world_point(d, w, d.sample_body[i], d.sample_local[i])
    local = wp.transpose(d.rotation[w, d.object_body]) @ (
        point - d.position[w, d.object_body]
    )
    hit = wp.mesh_query_point(d.object_mesh, local, 1.0e6)
    depth = float(wp.inf)
    if hit.result:
        nearest = wp.mesh_eval_position(d.object_mesh, hit.face, hit.u, hit.v)
        depth = wp.max(-hit.sign * wp.length(local - nearest) * 1000.0, 0.0)
        if d.region_enabled != 0 and d.sample_forbidden[i] != 0:
            gap = hit.sign * wp.length(local - nearest) * 1000.0
            wp.atomic_min(d.forbidden_gap, w, gap)
            wp.atomic_add(d.parts, w, 3, 2500.0 * wp.pow(wp.max(1.0 - gap, 0.0), 2.0))
    wp.atomic_max(d.penetration, w, depth)
    wp.atomic_add(d.parts, w, 3, 196.0 * wp.pow(wp.max(depth - 2.0, 0.0), 2.0))


@wp.func
def matrix_axis(r: wp.mat33, i: int):
    return wp.vec3(r[0, i], r[1, i], r[2, i])


@wp.func
def box_separation(d: CostData, w: int, a: int, b: int):
    """SAT projection gap: a positive value certifies full mesh separation."""
    ba, bb = d.capsule_body[a], d.capsule_body[b]
    ra, rb = d.rotation[w, ba], d.rotation[w, bb]
    delta = world_point(d, w, ba, d.box_center[a]) - world_point(
        d, w, bb, d.box_center[b]
    )
    gap = float(-wp.inf)
    for k in range(15):
        axis = wp.vec3(0.0)
        if k < 3:
            axis = matrix_axis(ra, k)
        elif k < 6:
            axis = matrix_axis(rb, k - 3)
        else:
            axis = wp.cross(matrix_axis(ra, (k - 6) // 3), matrix_axis(rb, (k - 6) % 3))
        length = wp.length(axis)
        if length > 1.0e-6:
            axis = axis / length
            radius = float(0.0)
            for j in range(3):
                radius += wp.abs(wp.dot(axis, matrix_axis(ra, j))) * d.box_half[a][j]
                radius += wp.abs(wp.dot(axis, matrix_axis(rb, j))) * d.box_half[b][j]
            gap = wp.max(gap, wp.abs(wp.dot(delta, axis)) - radius)
    return gap * 1000.0


@wp.kernel(enable_backward=False)
def capsule_cost(d: CostData):
    w, i = wp.tid()
    a, b = d.pair_left[i], d.pair_right[i]
    p0 = world_point(d, w, d.capsule_body[a], d.capsule_start[a])
    p1 = world_point(d, w, d.capsule_body[a], d.capsule_end[a])
    q0 = world_point(d, w, d.capsule_body[b], d.capsule_start[b])
    q1 = world_point(d, w, d.capsule_body[b], d.capsule_end[b])
    gap = (
        wp.sqrt(fa._segment_distance_squared(p0, p1, q0, q1) + 1.0e-16)
        - d.capsule_radius[a]
        - d.capsule_radius[b]
    ) * 1000.0
    hard_gap = gap
    if d.box_certificate != 0:
        hard_gap = wp.max(hard_gap, box_separation(d, w, a, b))
    wp.atomic_min(d.self_gap, w, hard_gap)
    wp.atomic_add(d.parts, w, 3, 2.25 * wp.pow(wp.max(6.0 - gap, 0.0), 2.0))


@wp.func
def object_axis_cosine(d: CostData, w: int):
    axis = d.rotation[w, d.object_body] @ wp.vec3(0.0, 0.0, 1.0)
    palm = d.rotation[w, d.palm_body] @ d.object_axis_palm
    return wp.abs(wp.dot(axis, palm))


@wp.kernel(enable_backward=False)
def finish_cost(d: CostData, s: SearchState):
    w = wp.tid()
    value = float(0.0)
    for k in range(4):
        value += d.parts[w, k]
    if d.pose_enabled != 0:
        value += d.pose_cost[w]
    if d.wrist_prior.enabled != 0:
        value += d.wrist_prior_cost[w]
    if (
        d.penetration[w] > 3.0
        or d.palm_gap[w] < 6.0
        or d.self_gap[w] < d.self_clearance
        or not wp.isfinite(value)
    ):
        value = float(wp.inf)
    if d.region_enabled != 0 and d.forbidden_gap[w] < 1.0:
        value = float(wp.inf)
    if d.object_axis_enabled != 0 and object_axis_cosine(d, w) < d.object_axis_cosine:
        value = float(wp.inf)
    s.scores[w] = value


@wp.kernel(enable_backward=False)
def prepare_ranking(d: CostData, s: SearchState):
    """Restore feasibility only when the whole population violates hard limits.

    Invalid poses can guide the distribution; only finite hard scores are exportable.
    """
    w = wp.tid()
    feasible = bool(False)
    for i in range(s.population):
        feasible = feasible or wp.isfinite(s.scores[i])
    value = s.scores[w]
    if not feasible:
        value = (
            wp.pow(wp.max(d.penetration[w] - 3.0, 0.0), 2.0)
            + wp.pow(wp.max(6.0 - d.palm_gap[w], 0.0), 2.0)
            + wp.pow(wp.max(d.self_clearance - d.self_gap[w], 0.0), 2.0)
        )
        if d.region_enabled != 0:
            value += wp.pow(wp.max(1.0 - d.forbidden_gap[w], 0.0), 2.0)
        if d.object_axis_enabled != 0:
            value += 1000.0 * wp.pow(
                wp.max(d.object_axis_cosine - object_axis_cosine(d, w), 0.0), 2.0
            )
        for k in range(4):
            if not wp.isfinite(d.parts[w, k]):
                value = float(wp.inf)
        if d.pose_enabled != 0 and not wp.isfinite(d.pose_cost[w]):
            value = float(wp.inf)
    s.selection_scores[w] = value


@wp.kernel(enable_backward=False)
def export_contacts(
    d: CostData,
    annotation_group: wp.array(dtype=int),
    local: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    sample_index = d.selected[0, annotation_group[i]]
    local[i] = d.sample_local[sample_index]
    normal[i] = d.sample_normal[sample_index]


@wp.kernel(enable_backward=False)
def palm_clearance_masked(d: CostData, indices: wp.array(dtype=int)):
    w, slot = wp.tid()
    i = indices[slot]
    body = d.capsule_body[i]
    for k in range(9):
        local = wp.lerp(d.capsule_start[i], d.capsule_end[i], float(k) / 8.0)
        point = world_point(d, w, body, local)
        p = wp.transpose(d.rotation[w, d.palm_body]) @ (
            point - d.position[w, d.palm_body]
        )
        hit = wp.mesh_query_point(d.palm_mesh, p, 1.0e6)
        gap = float(-wp.inf)
        if hit.result:
            nearest = wp.mesh_eval_position(d.palm_mesh, hit.face, hit.u, hit.v)
            gap = (hit.sign * wp.length(p - nearest) - d.capsule_radius[i]) * 1000.0
        wp.atomic_min(d.palm_gap, w, gap)


from retarget.dynamics.explicit_parameters import Parameters


@wp.kernel
def step_pose_cost(
    p: Parameters,
    qpos: wp.array2d(dtype=float),
    step: int,
    costs: wp.array2d(dtype=float),
):
    w = wp.tid()
    address = p.object_qpos
    error = wp.vec3(
        qpos[w, address] - p.seed[address],
        qpos[w, address + 1] - p.seed[address + 1],
        qpos[w, address + 2] - p.seed[address + 2],
    )
    q = wp.normalize(
        wp.vec4(
            qpos[w, address + 3],
            qpos[w, address + 4],
            qpos[w, address + 5],
            qpos[w, address + 6],
        )
    )
    goal = wp.normalize(
        wp.vec4(
            p.seed[address + 3],
            p.seed[address + 4],
            p.seed[address + 5],
            p.seed[address + 6],
        )
    )
    dot = wp.clamp(wp.dot(q, goal), -1.0, 1.0)
    costs[w, step] = p.weight * (
        5000.0 * wp.length_sq(error) + 50.0 * wp.max(1.0 - dot * dot, 0.0)
    )


@wp.kernel
def average_pose_cost(costs: wp.array2d(dtype=float), out: wp.array(dtype=float)):
    w = wp.tid()
    value = float(0.0)
    for step in range(costs.shape[1]):
        value += costs[w, step] / float(costs.shape[1])
    out[w] = value
