# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
"""Warp kernels for humanoid retargeting."""

import warp as wp

from robokit.utils.warp_utils import wp_vec7
from robokit.xform.warp.rotation_conversions import quaternion_apply_func, quaternion_multiply_func


# --- functions ---
@wp.func
def compute_target_func(
    root: wp_vec7,
    joint: wp_vec7,
    height_scale: wp.float32,
    target_scale: wp.float32,
    root_scale: wp.float32,
    rotation_offset: wp.vec4,
    position_offset: wp.vec3,
    ground_height: wp.float32,
) -> wp_vec7:
    root_position = wp.vec3(root[0], root[1], root[2])
    joint_position = wp.vec3(joint[0], joint[1], joint[2])
    position = root_position * root_scale * height_scale
    position += (joint_position - root_position) * target_scale * height_scale
    joint_rotation = wp.vec4(joint[3], joint[4], joint[5], joint[6])
    rotation = quaternion_multiply_func(joint_rotation, rotation_offset)
    position += quaternion_apply_func(rotation, position_offset)
    return wp_vec7(
        position[0],
        position[1],
        position[2] - ground_height,
        rotation[0],
        rotation[1],
        rotation[2],
        rotation[3],
    )


# --- kernels ---
@wp.kernel
def compute_targets_kernel(
    T_world_human: wp.array2d(dtype=wp_vec7),
    human_heights: wp.array1d(dtype=wp.float32),
    target_joint_indices: wp.array1d(dtype=wp.int32),
    target_scales: wp.array1d(dtype=wp.float32),
    rotation_offsets: wp.array1d(dtype=wp.vec4),
    position_offsets: wp.array1d(dtype=wp.vec3),
    origin_joint_indices: wp.array1d(dtype=wp.int32),
    task_joint_indices: wp.array1d(dtype=wp.int32),
    origin_scales: wp.array1d(dtype=wp.float32),
    task_scales: wp.array1d(dtype=wp.float32),
    T_world_target: wp.array2d(dtype=wp_vec7),
    T_world_base_target: wp.array1d(dtype=wp_vec7),
    target_vectors: wp.array3d(dtype=wp.float32),
    num_pose_targets: int,
    num_vector_targets: int,
    root_joint_index: int,
    root_target_index: int,
    root_scale: wp.float32,
    human_height_assumption: wp.float32,
    num_frames: int,
    ground_height: wp.float32,
):
    instance_index, target_index = wp.tid()  # pyright: ignore[reportAssignmentType]
    scale = human_height_assumption / human_heights[instance_index // num_frames]
    root = T_world_human[instance_index, root_joint_index]

    if target_index < num_pose_targets:
        target = compute_target_func(
            root,
            T_world_human[instance_index, target_joint_indices[target_index]],
            scale,
            target_scales[target_index],
            root_scale,
            rotation_offsets[target_index],
            position_offsets[target_index],
            ground_height,
        )
        T_world_target[instance_index, target_index] = target
        return

    vector_index = target_index - num_pose_targets
    if vector_index < num_vector_targets:
        origin = T_world_human[instance_index, origin_joint_indices[vector_index]]
        task = T_world_human[instance_index, task_joint_indices[vector_index]]
        target_vectors[instance_index, vector_index, 0] = (
            task[0] * task_scales[vector_index] - origin[0] * origin_scales[vector_index]
        ) * scale
        target_vectors[instance_index, vector_index, 1] = (
            task[1] * task_scales[vector_index] - origin[1] * origin_scales[vector_index]
        ) * scale
        target_vectors[instance_index, vector_index, 2] = (
            task[2] * task_scales[vector_index] - origin[2] * origin_scales[vector_index]
        ) * scale
        return

    if root_target_index >= 0:
        T_world_base_target[instance_index] = compute_target_func(
            root,
            T_world_human[instance_index, target_joint_indices[root_target_index]],
            scale,
            target_scales[root_target_index],
            root_scale,
            rotation_offsets[root_target_index],
            position_offsets[root_target_index],
            ground_height,
        )
    else:
        root_position = wp.vec3(root[0], root[1], root[2]) * root_scale * scale
        T_world_base_target[instance_index] = wp_vec7(
            root_position[0],
            root_position[1],
            root_position[2] - ground_height,
            root[3],
            root[4],
            root[5],
            root[6],
        )


@wp.kernel
def compute_init_states_kernel(
    previous_q: wp.array2d(dtype=wp.float32),
    midrange_q: wp.array1d(dtype=wp.float32),
    joint_limits: wp.array2d(dtype=wp.float32),
    candidate_tangent: wp.array2d(dtype=wp.float32),
    T_world_base_target: wp.array1d(dtype=wp_vec7),
    candidate_q: wp.array2d(dtype=wp.float32),
    candidate_base: wp.array1d(dtype=wp_vec7),
    num_local_candidates: int,
    num_candidates: int,
):
    """Initialize humanoid candidates with the base at the target.

    Initialization:
    1. Seed the first `num_local_candidates + 1` at previous_q and the rest at midrange.
    2. Add each row's joint block from the constant body-frame `candidate_tangent` table.
    3. Clamp joints.
    """
    batch_index, candidate_index, value_index = wp.tid()  # pyright: ignore[reportAssignmentType]
    candidate_row = batch_index * num_candidates + candidate_index
    if value_index == 0:
        candidate_base[candidate_row] = T_world_base_target[batch_index]
        return
    dof_index = value_index - 1
    if candidate_index <= num_local_candidates:
        q = previous_q[batch_index, dof_index]
    else:
        q = midrange_q[dof_index]
    q += candidate_tangent[candidate_row, candidate_tangent.shape[1] - candidate_q.shape[1] + dof_index]
    candidate_q[candidate_row, dof_index] = wp.clamp(q, joint_limits[dof_index, 0], joint_limits[dof_index, 1])


@wp.kernel
def set_solution_state_kernel(
    solution_q: wp.array2d(dtype=wp.float32),
    solution_base: wp.array1d(dtype=wp_vec7),
    joint_limits: wp.array2d(dtype=wp.float32),
    max_dq: wp.array1d(dtype=wp.float32),
    previous_q: wp.array2d(dtype=wp.float32),
    previous_base: wp.array1d(dtype=wp_vec7),
    output: wp.array2d(dtype=wp.float32),
    clamp_velocity: int,
):
    batch_index, value_index = wp.tid()  # pyright: ignore[reportAssignmentType]
    if value_index < 7:
        output[batch_index, value_index] = solution_base[batch_index][value_index]
        if value_index == 0:
            previous_base[batch_index] = solution_base[batch_index]
        return

    dof_index = value_index - 7
    q = solution_q[batch_index, dof_index]
    if clamp_velocity != 0:
        previous = previous_q[batch_index, dof_index]
        q = previous + wp.clamp(q - previous, -max_dq[dof_index], max_dq[dof_index])
    q = wp.clamp(q, joint_limits[dof_index, 0], joint_limits[dof_index, 1])
    output[batch_index, value_index] = q
    previous_q[batch_index, dof_index] = q


@wp.kernel
def set_trajectory_solution_kernel(
    solution_q: wp.array3d(dtype=wp.float32),
    solution_base: wp.array2d(dtype=wp_vec7),
    joint_limits: wp.array2d(dtype=wp.float32),
    max_dq: wp.array1d(dtype=wp.float32),
    output: wp.array3d(dtype=wp.float32),
    clamp_velocity: int,
):
    batch_index, value_index = wp.tid()  # pyright: ignore[reportAssignmentType]
    for frame_index in range(solution_q.shape[1]):
        if value_index < 7:
            output[batch_index, frame_index, value_index] = solution_base[batch_index, frame_index][value_index]
        else:
            dof_index = value_index - 7
            q = solution_q[batch_index, frame_index, dof_index]
            if clamp_velocity != 0 and frame_index > 0:
                previous = output[batch_index, frame_index - 1, value_index]
                q = previous + wp.clamp(q - previous, -max_dq[dof_index], max_dq[dof_index])
            output[batch_index, frame_index, value_index] = wp.clamp(
                q, joint_limits[dof_index, 0], joint_limits[dof_index, 1]
            )
