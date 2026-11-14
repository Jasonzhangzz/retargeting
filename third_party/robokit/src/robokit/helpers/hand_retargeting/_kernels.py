# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false
"""Warp kernels shared by online and offline hand retargeting."""

import warp as wp

from robokit.lie.se3_kernels import se3_compose_func, se3_inverse_func
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.warp.rotation_conversions import quaternion_multiply_func


@wp.kernel
def center_points_kernel(points: wp.array3d(dtype=float), root: int, centered: wp.array3d(dtype=float)):
    batch, point, axis = wp.tid()  # pyright: ignore[reportAssignmentType]
    centered[batch, point, axis] = points[batch, point, axis] - points[batch, root, axis]


@wp.kernel
def center_base_kernel(
    T_world_base: wp.array(dtype=wp_vec7),
    points: wp.array3d(dtype=float),
    root: int,
    T_local_base: wp.array(dtype=wp_vec7),
):
    batch = wp.tid()
    T = T_world_base[batch]
    for axis in range(3):
        T[axis] -= points[batch, root, axis]
    T_local_base[batch] = T


@wp.kernel
def restore_translation_kernel(points: wp.array3d(dtype=float), root: int, output: wp.array2d(dtype=float)):
    batch, axis = wp.tid()  # pyright: ignore[reportAssignmentType]
    output[batch, axis] += points[batch, root, axis]


@wp.kernel
def compute_vector_targets_kernel(
    points: wp.array3d(dtype=wp.float32),
    origin_indices: wp.array1d(dtype=wp.int32),
    task_indices: wp.array1d(dtype=wp.int32),
    pinch_mask: wp.array1d(dtype=wp.bool),
    target_scale: wp.float32,
    snap_distance: wp.float32,
    release_distance: wp.float32,
    ema_alpha: wp.float32,
    history_valid: int,
    latched: wp.array2d(dtype=wp.bool),
    targets: wp.array3d(dtype=wp.float32),
):
    batch_index, vector_index = wp.tid()  # pyright: ignore[reportAssignmentType]
    origin_index = origin_indices[vector_index]
    task_index = task_indices[vector_index]
    target = wp.vec3(
        points[batch_index, task_index, 0] - points[batch_index, origin_index, 0],
        points[batch_index, task_index, 1] - points[batch_index, origin_index, 1],
        points[batch_index, task_index, 2] - points[batch_index, origin_index, 2],
    )
    is_latched = False
    if pinch_mask[vector_index]:
        distance = wp.length(target)
        is_latched = latched[batch_index, vector_index]
        if distance < snap_distance:
            is_latched = True
        elif distance > release_distance:
            is_latched = False
        latched[batch_index, vector_index] = is_latched
        if is_latched:  # a pinched pair aims at a 0.1 mm gap along its current direction
            target *= 1.0e-4 / (distance + 1.0e-6)
        else:
            target *= target_scale
    else:
        target *= target_scale

    if history_valid != 0 and not is_latched:
        previous = wp.vec3(
            targets[batch_index, vector_index, 0],
            targets[batch_index, vector_index, 1],
            targets[batch_index, vector_index, 2],
        )
        target = ema_alpha * target + (1.0 - ema_alpha) * previous

    for axis in range(3):
        targets[batch_index, vector_index, axis] = target[axis]


@wp.kernel
def compute_direction_targets_kernel(
    points: wp.array3d(dtype=wp.float32),
    origin_indices: wp.array1d(dtype=wp.int32),
    task_indices: wp.array1d(dtype=wp.int32),
    ema_alpha: wp.float32,
    history_valid: int,
    targets: wp.array3d(dtype=wp.float32),
):
    batch_index, direction_index = wp.tid()  # pyright: ignore[reportAssignmentType]
    origin_index = origin_indices[direction_index]
    task_index = task_indices[direction_index]
    target = wp.vec3(
        points[batch_index, task_index, 0] - points[batch_index, origin_index, 0],
        points[batch_index, task_index, 1] - points[batch_index, origin_index, 1],
        points[batch_index, task_index, 2] - points[batch_index, origin_index, 2],
    )
    if history_valid != 0:
        previous = wp.vec3(
            targets[batch_index, direction_index, 0],
            targets[batch_index, direction_index, 1],
            targets[batch_index, direction_index, 2],
        )
        target = ema_alpha * target + (1.0 - ema_alpha) * previous

    for axis in range(3):
        targets[batch_index, direction_index, axis] = target[axis]


@wp.kernel
def set_online_solution_kernel(
    solution_q: wp.array2d(dtype=wp.float32),
    solution_base: wp.array1d(dtype=wp_vec7),
    joint_limits: wp.array2d(dtype=wp.float32),
    ema_alpha: wp.float32,
    ema_valid: int,
    previous_q: wp.array2d(dtype=wp.float32),
    previous_base: wp.array1d(dtype=wp_vec7),
    output_q_history: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
):
    batch_index, value_index = wp.tid()  # pyright: ignore[reportAssignmentType]
    if value_index < 7:
        value = solution_base[batch_index][value_index]
        output[batch_index, value_index] = value
        if value_index == 0:
            previous_base[batch_index] = solution_base[batch_index]
        return

    joint_index = value_index - 7
    raw = wp.clamp(
        solution_q[batch_index, joint_index],
        joint_limits[joint_index, 0],
        joint_limits[joint_index, 1],
    )
    filtered = raw
    if ema_valid != 0:
        filtered = ema_alpha * raw + (1.0 - ema_alpha) * output_q_history[batch_index, joint_index]
    previous_q[batch_index, joint_index] = raw
    output_q_history[batch_index, joint_index] = filtered
    output[batch_index, value_index] = filtered


@wp.kernel
def convert_root_quaternions_kernel(
    root_quat_wxyz: wp.array3d(dtype=wp.float32),
    q_wxyz_target_root: wp.vec4,
    targets: wp.array3d(dtype=wp.float32),
):
    batch_index, frame_index = wp.tid()  # pyright: ignore[reportAssignmentType]
    q_wxyz_world_target = wp.vec4(
        root_quat_wxyz[batch_index, frame_index, 0],
        root_quat_wxyz[batch_index, frame_index, 1],
        root_quat_wxyz[batch_index, frame_index, 2],
        root_quat_wxyz[batch_index, frame_index, 3],
    )
    q_wxyz_world_root = quaternion_multiply_func(q_wxyz_world_target, q_wxyz_target_root)
    for axis in range(4):
        targets[batch_index, frame_index, axis] = q_wxyz_world_root[axis]


@wp.kernel
def set_default_trajectory_state_kernel(
    target_points: wp.array4d(dtype=wp.float32),
    root_target_index: int,
    q_wxyz_world_root: wp.array3d(dtype=wp.float32),
    has_root_quaternions: int,
    T_base_root_rest: wp_vec7,
    floating_base: int,
    q_init: wp.array1d(dtype=wp.float32),
    q: wp.array3d(dtype=wp.float32),
    T_world_base: wp.array2d(dtype=wp_vec7),
):
    batch_index, frame_index = wp.tid()  # pyright: ignore[reportAssignmentType]
    for joint_index in range(q.shape[2]):
        q[batch_index, frame_index, joint_index] = q_init[joint_index]

    T_world_base_i = wp_vec7(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    if floating_base != 0:
        q_w = wp.float32(1.0)
        q_x = wp.float32(0.0)
        q_y = wp.float32(0.0)
        q_z = wp.float32(0.0)
        if has_root_quaternions != 0:
            q_w = q_wxyz_world_root[batch_index, frame_index, 0]
            q_x = q_wxyz_world_root[batch_index, frame_index, 1]
            q_y = q_wxyz_world_root[batch_index, frame_index, 2]
            q_z = q_wxyz_world_root[batch_index, frame_index, 3]
        T_world_root = wp_vec7(
            target_points[batch_index, frame_index, root_target_index, 0],
            target_points[batch_index, frame_index, root_target_index, 1],
            target_points[batch_index, frame_index, root_target_index, 2],
            q_w,
            q_x,
            q_y,
            q_z,
        )
        T_world_base_i = se3_compose_func(T_world_root, se3_inverse_func(T_base_root_rest))
    T_world_base[batch_index, frame_index] = T_world_base_i


@wp.kernel
def set_trajectory_solution_kernel(
    solution_q: wp.array3d(dtype=wp.float32),
    solution_base: wp.array2d(dtype=wp_vec7),
    joint_limits: wp.array2d(dtype=wp.float32),
    output: wp.array3d(dtype=wp.float32),
):
    batch_index, frame_index, value_index = wp.tid()  # pyright: ignore[reportAssignmentType]
    if value_index < 7:
        output[batch_index, frame_index, value_index] = solution_base[batch_index, frame_index][value_index]
        return
    joint_index = value_index - 7
    output[batch_index, frame_index, value_index] = wp.clamp(
        solution_q[batch_index, frame_index, joint_index],
        joint_limits[joint_index, 0],
        joint_limits[joint_index, 1],
    )
