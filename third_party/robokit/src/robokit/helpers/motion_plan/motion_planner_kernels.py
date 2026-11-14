# pyright: reportArgumentType=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportIndexIssue=false, reportOperatorIssue=false
"""Warp kernels of the motion planner: seeds, waypoint IK picks, frame splits, retries and timing."""

import warp as wp

from robokit.utils.warp_utils import warp_max, wp_vec7


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------
@wp.func
def _interpolate_pose(a: wp_vec7, z: wp_vec7, alpha: float) -> wp_vec7:
    """A pose lerped and slerped from `a` to `z`."""
    t = wp.lerp(wp.vec3(a[0], a[1], a[2]), wp.vec3(z[0], z[1], z[2]), alpha)
    r = wp.quat_slerp(wp.quat(a[4], a[5], a[6], a[3]), wp.quat(z[4], z[5], z[6], z[3]), alpha)
    return wp_vec7(t[0], t[1], t[2], r[3], r[0], r[1], r[2])


@wp.func
def _detour_func(alpha: float, peak: float) -> float:
    """A smooth bump from 0 at both ends of the path to 1 at `peak`."""
    t = wp.where(alpha <= peak, 0.5 * alpha / peak, 0.5 + 0.5 * (alpha - peak) / (1.0 - peak))
    return 16.0 * t * t * (1.0 - t) * (1.0 - t)


# ---------------------------------------------------------------------------
# goals and waypoint IK
# ---------------------------------------------------------------------------
@wp.kernel
def _pin_inactive_joints_kernel(
    start_q: wp.array2d(dtype=wp.float32),
    active_joint_mask: wp.array1d(dtype=wp.float32),
    num_candidates: int,
    target_q: wp.array2d(dtype=wp.float32),
):
    batch_idx, dof = wp.tid()  # pyright: ignore[reportAssignmentType]
    if active_joint_mask[dof] == 0.0:
        target_q[batch_idx, dof] = start_q[batch_idx // num_candidates, dof]


@wp.kernel
def _expand_ik_goal_candidates_kernel(
    ik_goal_q: wp.array2d(dtype=wp.float32),
    goal_offset: int,
    num_goal_candidates: int,
    total_goal_candidates: int,
    num_trajectory_candidates: int,
    best_detour_goals: int,  # the detour half (second) takes the first goals again, with new detours: a retry round
    valid_count: wp.array1d(dtype=wp.int32),
    ik_goal_base: wp.array1d(dtype=wp_vec7),  # each candidate's goal base pose comes along with its joints
    target_q_per_candidate: wp.array2d(dtype=wp.float32),
    target_base_per_candidate: wp.array1d(dtype=wp_vec7),
):
    batch_idx, dof = wp.tid()  # pyright: ignore[reportAssignmentType]
    instance = batch_idx // num_trajectory_candidates
    trajectory_candidate = batch_idx % num_trajectory_candidates
    repeats = num_trajectory_candidates // num_goal_candidates
    goal = goal_offset + trajectory_candidate // repeats
    if best_detour_goals != 0 and trajectory_candidate >= num_trajectory_candidates // 2:
        goal = (trajectory_candidate - num_trajectory_candidates // 2) // repeats
    if valid_count[instance] > 0:  # only valid IK goals seed trajectories while there are any
        goal = goal % valid_count[instance]
    target_q_per_candidate[batch_idx, dof] = ik_goal_q[instance * total_goal_candidates + goal, dof]
    if dof == 0:
        target_base_per_candidate[batch_idx] = ik_goal_base[instance * total_goal_candidates + goal]


@wp.kernel
def _blend_kernel(
    a: wp.array2d(dtype=wp.float32), b: wp.array2d(dtype=wp.float32), alpha: float, out: wp.array2d(dtype=wp.float32)
):
    row, dof = wp.tid()  # pyright: ignore[reportAssignmentType]
    out[row, dof] = (1.0 - alpha) * a[row, dof] + alpha * b[row, dof]


@wp.kernel
def _waypoint_rows_kernel(
    waypoint_q: wp.array3d(dtype=wp.float32),  # (batch, >= waypoints - 1, dofs): the waypoints before the goal
    target_q: wp.array2d(dtype=wp.float32),  # (rows, dofs): each guess's goal
    start_base: wp.array1d(dtype=wp_vec7),
    target_base: wp.array1d(dtype=wp_vec7),
    num_candidates: int,
    rows_q: wp.array3d(dtype=wp.float32),  # (rows, waypoints, dofs)
    rows_base: wp.array2d(dtype=wp_vec7),
):
    """Each guess's waypoints: its query's, then its own goal."""
    row, waypoint, dof = wp.tid()  # pyright: ignore[reportAssignmentType]
    last = rows_q.shape[1] - 1
    if waypoint == last:
        rows_q[row, waypoint, dof] = target_q[row, dof]
    else:
        rows_q[row, waypoint, dof] = waypoint_q[row // num_candidates, waypoint, dof]
    if dof == 0 and waypoint == 0:
        rows_base[row, 0] = start_base[row // num_candidates]
    if dof == 0 and waypoint == last:
        rows_base[row, last] = target_base[row]


@wp.kernel
def _pick_ik_kernel(
    pool_q: wp.array2d(dtype=wp.float32),  # (batch * answers, dofs)
    pool_base: wp.array1d(dtype=wp_vec7),
    valid_count: wp.array1d(dtype=wp.int32),
    waypoint: int,
    after: int,  # the next waypoint with joints
    rows_q: wp.array3d(dtype=wp.float32),
    rows_base: wp.array2d(dtype=wp_vec7),
):
    """Each guess's IK answer nearest the blend of its neighbors; only valid answers while there are any."""
    row = wp.tid()
    query = row // (rows_q.shape[0] // valid_count.shape[0])
    answers = pool_q.shape[0] // valid_count.shape[0]
    first = query * answers
    count = wp.where(valid_count[query] > 0, valid_count[query], answers)
    alpha = 1.0 / float(after - waypoint + 1)  # by place in the list
    best = int(0)
    best_distance = float(1.0e30)
    for answer in range(count):
        distance = float(0.0)
        for dof in range(rows_q.shape[2]):
            blend = (1.0 - alpha) * rows_q[row, waypoint - 1, dof] + alpha * rows_q[row, after, dof]
            distance += (pool_q[first + answer, dof] - blend) * (pool_q[first + answer, dof] - blend)
        if distance < best_distance:
            best_distance = distance
            best = answer
    for dof in range(rows_q.shape[2]):
        rows_q[row, waypoint, dof] = pool_q[first + best, dof]
    rows_base[row, waypoint] = pool_base[first + best]


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------
@wp.kernel
def _seed_kernel(
    waypoint_q: wp.array3d(dtype=wp.float32),  # (guesses, waypoints, dofs); out rows repeat each guess's
    frames: wp.array1d(dtype=wp.int32),  # each waypoint's frame
    locked: wp.array1d(dtype=wp.int32),  # each segment: 1 keeps the straight line
    rng_seed: int,
    active_joint_mask: wp.array1d(dtype=wp.float32),
    joint_limits: wp.array2d(dtype=wp.float32),
    sample_range: float,
    num_candidates: int,
    num_direct_candidates: int,
    random_peaks: int,  # 0: every detour peaks mid-segment; 1: random peaks, every other candidate bending twice
    out: wp.array3d(dtype=wp.float32),
):
    """Straight lines through each guess's waypoints; candidates past the direct ones detour on unlocked segments."""
    row, frame, dof = wp.tid()  # pyright: ignore[reportAssignmentType]
    repeats = out.shape[0] // waypoint_q.shape[0]  # detour samples per guess, each a size
    segment = int(0)
    while segment < locked.shape[0] - 1 and frames[segment + 1] <= frame:
        segment += 1
    start = waypoint_q[row // repeats, segment, dof]
    if active_joint_mask[dof] == 0.0:
        out[row, frame, dof] = start
        return
    target = waypoint_q[row // repeats, segment + 1, dof]
    alpha = wp.float32(frame - frames[segment]) / wp.float32(frames[segment + 1] - frames[segment])
    direct = start + alpha * (target - start)
    candidate = row % (num_candidates * repeats)
    if candidate < num_direct_candidates or locked[segment] != 0:
        out[row, frame, dof] = direct
        return
    stream = row * locked.shape[0] + segment  # one per row and segment
    random_state = wp.rand_init(wp.int32(rng_seed), wp.int32(stream * out.shape[2] + dof))
    joint_range = joint_limits[dof, 1] - joint_limits[dof, 0]
    size = float(candidate % repeats + 1) / float(repeats)  # consecutive candidates cycle through detour sizes
    offset = (wp.randf(random_state) - 0.5) * sample_range * size * joint_range
    # Seed detours:
    # - Mid-segment: early frames barely move.
    # - Random: joints share a peak; alternate candidates bend early and late to explore obstacles.
    twice = random_peaks != 0 and candidate % 2 == 1
    r = wp.randf(wp.rand_init(wp.int32(rng_seed), wp.int32(stream)))
    peak = wp.where(random_peaks != 0, wp.where(twice, 0.1 + 0.3 * r, 0.25 + 0.5 * r), 0.5)
    q = direct + _detour_func(alpha, peak) * offset
    if twice:  # the second bend draws from another stream
        late = wp.randf(wp.rand_init(wp.int32(rng_seed + 7919), wp.int32(stream)))
        late_offset = wp.randf(wp.rand_init(wp.int32(rng_seed + 7919), wp.int32(stream * out.shape[2] + dof)))
        q += _detour_func(alpha, 0.6 + 0.3 * late) * (late_offset - 0.5) * sample_range * size * joint_range
    out[row, frame, dof] = wp.clamp(q, joint_limits[dof, 0], joint_limits[dof, 1])


@wp.kernel
def _seed_base_kernel(
    waypoint_base: wp.array2d(dtype=wp_vec7),  # (guesses, waypoints)
    frames: wp.array1d(dtype=wp.int32),
    known: wp.array1d(dtype=wp.int32),  # waypoints with a base pose; the base between them is interpolated
    out: wp.array2d(dtype=wp_vec7),
):
    """Each guess's floating-base path, lerped and slerped between the waypoints with a base pose."""
    row, frame = wp.tid()  # pyright: ignore[reportAssignmentType]
    left = int(0)
    for waypoint in range(frames.shape[0] - 1):
        if known[waypoint] != 0 and frames[waypoint] <= frame:
            left = waypoint
    right = left + 1
    while known[right] == 0:
        right += 1
    alpha = wp.float32(frame - frames[left]) / wp.float32(frames[right] - frames[left])
    out[row, frame] = _interpolate_pose(waypoint_base[row, left], waypoint_base[row, right], alpha)


@wp.kernel
def _min_distance_kernel(
    sphere_signed_dists: wp.array1d(dtype=wp.float32),
    eff_radii: wp.array1d(dtype=wp.float32),
    points_per_seed: int,
    seed_distance: wp.array1d(dtype=wp.float32),
):
    pt = wp.tid()
    wp.atomic_min(seed_distance, pt // points_per_seed, sphere_signed_dists[pt] - eff_radii[pt % eff_radii.shape[0]])


@wp.kernel
def _set_detour_seed_kernel(
    seeds: wp.array3d(dtype=wp.float32),
    seed_distance: wp.array1d(dtype=wp.float32),
    seeds_per_candidate: int,
    num_candidates: int,
    num_direct_candidates: int,
    target_q_per_candidate: wp.array2d(dtype=wp.float32),
    lanes: int,  # threads per candidate: a CUDA warp's 32, or 1 on CPU
    traj_q_init: wp.array3d(dtype=wp.float32),
):
    tid = wp.tid()
    row, lane = tid // lanes, tid % lanes
    candidate = row % num_candidates
    if candidate < num_direct_candidates:  # a direct path, unless an earlier candidate already goes to its goal
        repeated = int(0)
        for other in range(row - candidate, row):
            gap = float(0.0)
            for dof in range(target_q_per_candidate.shape[1]):
                gap = wp.max(gap, wp.abs(target_q_per_candidate[other, dof] - target_q_per_candidate[row, dof]))
            if gap < 1.0e-3:  # rad
                repeated = 1
        if repeated == 0:
            return
    # Shortest free seed, else farthest from obstacles; lanes scan strided seeds and agree on the first winner.
    first = row * seeds_per_candidate
    shortest = float(1.0e30)
    shortest_k = float(1.0e9)
    farthest = float(-1.0e30)
    farthest_k = float(1.0e9)
    for k in range(first + lane, first + seeds_per_candidate, lanes):
        if seed_distance[k] > farthest:
            farthest = seed_distance[k]
            farthest_k = float(k)
        if seed_distance[k] >= 0.0:
            length = float(0.0)
            for frame in range(seeds.shape[1] - 1):
                step = float(0.0)
                for dof in range(seeds.shape[2]):
                    step += (seeds[k, frame + 1, dof] - seeds[k, frame, dof]) * (
                        seeds[k, frame + 1, dof] - seeds[k, frame, dof]
                    )
                length += wp.sqrt(step)
            if length < shortest:
                shortest = length
                shortest_k = float(k)
    shortest_all = -warp_max(-shortest)
    best = -warp_max(-wp.where(shortest == shortest_all, shortest_k, 1.0e9))
    if shortest_all >= 1.0e30:
        farthest_all = warp_max(farthest)
        best = -warp_max(-wp.where(farthest == farthest_all, farthest_k, 1.0e9))
    k = int(best)
    for element in range(lane, seeds.shape[1] * seeds.shape[2], lanes):
        frame, dof = element // seeds.shape[2], element % seeds.shape[2]
        traj_q_init[row, frame, dof] = seeds[k, frame, dof]


# ---------------------------------------------------------------------------
# frame splits
# ---------------------------------------------------------------------------
@wp.kernel
def _hold_kernel(
    path: wp.array3d(dtype=wp.float32),
    frames: wp.array1d(dtype=wp.int32),
    fixed: wp.array1d(dtype=wp.int32),
    out: wp.array3d(dtype=wp.float32),
):
    """Hold each fixed segment at its start, and shift the frames after it to meet it."""
    row, frame, dof = wp.tid()  # pyright: ignore[reportAssignmentType]
    shift = float(0.0)
    for segment in range(fixed.shape[0]):
        left = frames[segment]
        right = frames[segment + 1]
        if fixed[segment] != 0:
            if frame >= left and frame <= right:
                out[row, frame, dof] = path[row, left, dof] - shift
                return
            if frame > right:
                shift += path[row, right, dof] - path[row, left, dof]
    out[row, frame, dof] = path[row, frame, dof] - shift


@wp.kernel
def _frame_need_kernel(
    path: wp.array3d(dtype=wp.float32),
    velocity_limits: wp.array1d(dtype=wp.float32),
    acceleration_limit: float,
    jerk_limit: float,
    need: wp.array1d(dtype=wp.float32),
):
    """Each frame's step under the velocity, acceleration and jerk limits (the retimer's stencils), over the batch."""
    row, frame, dof = wp.tid()  # pyright: ignore[reportAssignmentType]
    last = path.shape[1] - 1
    m2 = path[row, wp.max(frame - 2, 0), dof]
    m1 = path[row, wp.max(frame - 1, 0), dof]
    p1 = path[row, wp.min(frame + 1, last), dof]
    p2 = path[row, wp.min(frame + 2, last), dof]
    q0 = path[row, frame, dof]
    velocity = wp.abs(m2 - 8.0 * m1 + 8.0 * p1 - p2) / 12.0 / wp.max(velocity_limits[dof], 1.0e-8)
    acceleration = wp.sqrt(wp.abs(-m2 + 16.0 * m1 - 30.0 * q0 + 16.0 * p1 - p2) / 12.0 / acceleration_limit)
    jerk = wp.pow(wp.abs(-0.5 * m2 + m1 - p1 + 0.5 * p2) / jerk_limit, 1.0 / 3.0)
    wp.atomic_max(need, frame, wp.max(wp.max(velocity, acceleration), jerk))


@wp.kernel
def _resplit_kernel(
    need: wp.array1d(dtype=wp.float32),  # each frame's step under the limits
    frames: wp.array1d(dtype=wp.int32),  # each waypoint's frame
    fixed: wp.array1d(dtype=wp.int32),  # each segment: 1 keeps its frames
    elapsed: wp.array1d(dtype=wp.float32),  # scratch: the time needed up to each frame
    new_frames: wp.array1d(dtype=wp.int32),
    at: wp.array1d(dtype=wp.float32),  # each new frame's place on the old frames
):
    """Waypoint frames spread by the time each segment needs, two at least; one thread."""
    steps = need.shape[0] - 1
    segments = fixed.shape[0]
    elapsed[0] = 0.0
    for f in range(steps):
        elapsed[f + 1] = elapsed[f] + need[f] + need[f + 1] + 2.0e-9
    total = float(0.0)
    spare = steps  # frames beyond two per moving segment
    for s in range(segments):
        if fixed[s] != 0:
            spare -= frames[s + 1] - frames[s]
        else:
            total += elapsed[frames[s + 1]] - elapsed[frames[s]]
            spare -= 2
    left = steps
    for s in range(segments):  # each segment's frame count, for now
        count = frames[s + 1] - frames[s]
        if fixed[s] == 0:
            count = 2 + int(wp.floor((elapsed[frames[s + 1]] - elapsed[frames[s]]) / total * float(spare)))
        new_frames[s + 1] = count
        left -= count
    for _ in range(left):  # the largest remainders take the frames left
        best = int(-1)
        best_remainder = float(-1.0)
        for s in range(segments):
            share = (elapsed[frames[s + 1]] - elapsed[frames[s]]) / total * float(spare)
            remainder = share - wp.floor(share)
            if fixed[s] == 0 and new_frames[s + 1] == 2 + int(wp.floor(share)) and remainder > best_remainder:
                best = s
                best_remainder = remainder
        new_frames[best + 1] += 1
    new_frames[0] = 0
    k = int(0)
    for s in range(segments):
        new_frames[s + 1] += new_frames[s]
        count = new_frames[s + 1] - new_frames[s]
        for j in range(count):
            if fixed[s] != 0:
                at[new_frames[s] + j] = float(frames[s] + j)
            else:  # even steps of needed time, mapped back to frames
                a = elapsed[frames[s]]
                e = a + (elapsed[frames[s + 1]] - a) * float(j) / float(count)
                while k < steps - 1 and elapsed[k + 1] <= e:
                    k += 1
                at[new_frames[s] + j] = float(k) + (e - elapsed[k]) / (elapsed[k + 1] - elapsed[k])
    at[steps] = float(steps)


@wp.kernel
def _resample_kernel(
    path: wp.array3d(dtype=wp.float32), at: wp.array1d(dtype=wp.float32), out: wp.array3d(dtype=wp.float32)
):
    """Linear interpolation at fractional frames."""
    row, frame, dof = wp.tid()  # pyright: ignore[reportAssignmentType]
    i = wp.min(int(at[frame]), path.shape[1] - 2)
    w = at[frame] - float(i)
    out[row, frame, dof] = (1.0 - w) * path[row, i, dof] + w * path[row, i + 1, dof]


@wp.kernel
def _resample_base_kernel(
    path: wp.array2d(dtype=wp_vec7), at: wp.array1d(dtype=wp.float32), out: wp.array2d(dtype=wp_vec7)
):
    """Base poses lerped and slerped at fractional frames."""
    row, frame = wp.tid()  # pyright: ignore[reportAssignmentType]
    i = wp.min(int(at[frame]), path.shape[1] - 2)
    out[row, frame] = _interpolate_pose(path[row, i], path[row, i + 1], at[frame] - float(i))


@wp.kernel
def _frame_flags_kernel(
    frames: wp.array1d(dtype=wp.int32),
    locked: wp.array1d(dtype=wp.int32),  # each segment: 1 holds its frames
    check: wp.array1d(dtype=wp.int32),  # each segment: 1 checks world collision
    lock_goal: int,
    spheres: wp.array2d(dtype=wp.int32),  # each segment: 1 checks each sphere
    pairs: wp.array2d(dtype=wp.int32),  # each segment: each link pair's reference waypoint + 1, 0 off
    named: wp.array1d(dtype=wp.int32),  # each waypoint's target links as bits, -1 all
    held: wp.array1d(dtype=wp.int32),  # each frame's held target links as bits
    checked: wp.array1d(dtype=wp.int32),
    sphere_active: wp.array2d(dtype=wp.int32),
    reference_frames: wp.array2d(dtype=wp.int32),
    frame_segment: wp.array1d(dtype=wp.int32),  # or empty: the segment of each frame's interval to the next
):
    """Each frame of a split: its held links (a waypoint's named, all at a held goal, inside a locked segment those
    both its waypoints name), collision-checked, its checked spheres (at an inner waypoint, those both segments
    check) and its pairs' reference frame."""
    frame = wp.tid()
    last = frames.shape[0] - 1
    segment = int(0)
    while segment < last - 1 and frames[segment + 1] < frame:
        segment += 1
    if frame_segment.shape[0] > 0:
        frame_segment[frame] = wp.where(frame < frames[segment + 1] or segment == last - 1, segment, segment + 1)
    hold = int(0)
    for k in range(last):
        if frames[k] == frame:
            hold = named[k]
    if locked[segment] != 0 and frame > frames[segment] and frame < frames[segment + 1]:
        hold = named[segment] & named[segment + 1]
    held[frame] = wp.where(frame == frames[last] and lock_goal != 0, -1, hold)
    checked[frame] = wp.where(frame == 0, 1, check[segment])
    inner = frame == frames[segment + 1] and segment < last - 1
    for k in range(spheres.shape[1]):
        sphere_active[frame, k] = spheres[segment, k] * wp.where(inner, spheres[wp.min(segment + 1, last - 1), k], 1)
    for k in range(pairs.shape[1]):
        reference_frames[frame, k] = wp.where(pairs[segment, k] != 0, frames[wp.max(pairs[segment, k] - 1, 0)], -1)


@wp.kernel
def _held_radii_kernel(
    radii: wp.array1d(dtype=wp.float32),
    spheres: wp.array2d(dtype=wp.int32),  # each segment: 1 checks each sphere
    segment: int,
    out: wp.array1d(dtype=wp.float32),
):
    """The radii of the spheres `segment` checks, -1e9 (never colliding) for the others."""
    k = wp.tid()
    out[k] = wp.where(spheres[segment, k] != 0, radii[k], -1.0e9)


@wp.kernel
def _collision_mask_kernel(
    base: wp.array3d(dtype=wp.bool),
    checked: wp.array1d(dtype=wp.int32),
    sphere_active: wp.array2d(dtype=wp.int32),
    sphere_indices: wp.array1d(dtype=wp.int32),
    out: wp.array3d(dtype=wp.bool),
):
    row, frame, sphere = wp.tid()  # pyright: ignore[reportAssignmentType]
    active = sphere_active[frame, sphere_indices[sphere]] != 0
    out[row, frame, sphere] = base[row, frame, sphere] and checked[frame] != 0 and active


# ---------------------------------------------------------------------------
# stages and retries
# ---------------------------------------------------------------------------
@wp.kernel
def _score_condition_kernel(
    best_scores: wp.array1d(dtype=wp.float32), feasible: int, condition: wp.array1d(dtype=wp.int32)
):
    """condition = 1 when some query's best score is feasible (`feasible` 1) or infeasible (`feasible` 0)."""
    batch_idx = wp.tid()
    if (best_scores[batch_idx] < 1.0e6) == (feasible != 0):
        condition[0] = 1


@wp.kernel
def _set_best_retry_result_kernel(
    scores: wp.array1d(dtype=wp.float32),
    q: wp.array3d(dtype=wp.float32),
    base: wp.array2d(dtype=wp_vec7),
    start_distance: wp.array1d(dtype=wp.float32),
    best_score: wp.array1d(dtype=wp.float32),
    condition: wp.array1d(dtype=wp.int32),
    steps: wp.array2d(dtype=wp.float32),  # or None: each query's segment log steps
    final_q: wp.array3d(dtype=wp.float32),
    final_base: wp.array2d(dtype=wp_vec7),
    final_steps: wp.array2d(dtype=wp.float32),
):
    batch = wp.tid()
    if scores[batch] < best_score[batch]:
        best_score[batch] = scores[batch]
        for frame in range(q.shape[1]):
            final_base[batch, frame] = base[batch, frame]
            for dof in range(q.shape[2]):
                final_q[batch, frame, dof] = q[batch, frame, dof]
        for segment in range(final_steps.shape[1]):
            final_steps[batch, segment] = steps[batch, segment]
    # Retry infeasible plans (score ≥1e6), except a colliding start that retries cannot fix.
    if best_score[batch] >= 1.0e6 and start_distance[batch] >= 0.0:
        wp.atomic_max(condition, 0, 1)


# ---------------------------------------------------------------------------
# timing and online history
# ---------------------------------------------------------------------------
@wp.kernel
def _motion_time_kernel(
    step: wp.array1d(dtype=wp.float32), num_steps: int, offset: float, motion_time: wp.array1d(dtype=wp.float32)
):
    batch = wp.tid()
    motion_time[batch] = wp.float32(num_steps) * step[batch] + offset


@wp.kernel
def _append_q_history_kernel(
    q_history: wp.array3d(dtype=wp.float32),
    start_q: wp.array2d(dtype=wp.float32),
):
    batch, dof = wp.tid()  # pyright: ignore[reportAssignmentType]
    q_history[batch, 0, dof] = q_history[batch, 1, dof]
    q_history[batch, 1, dof] = q_history[batch, 2, dof]
    q_history[batch, 2, dof] = start_q[batch, dof]
