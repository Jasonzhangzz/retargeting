"""Warp body-model engine (`body_fk_warp`, `body_lbs_warp`), generic over any LBS variant; outputs `wp_vec7`."""

# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportReturnType=false
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Tuple

import numpy as np
import warp as wp

from robokit.lie.se3_kernels import se3_compose_func
from robokit.smplx.spec import BodyModelSpec, BodyModelState
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.warp.rotation_conversions import quaternion_to_matrix_func


@dataclass
class BodyModelSpecTensors:
    """Lazy device-side view of a `BodyModelSpec`."""

    spec: BodyModelSpec
    device: str

    @cached_property
    def parent_joint_indices(self) -> wp.array:
        """`[J]` int32, root = -1."""
        return wp.from_numpy(self.spec.parents.astype(np.int32), device=self.device, dtype=wp.int32)

    @cached_property
    def J_template(self) -> wp.array:
        """`J_regressor @ v_template` - `[J]` vec3 rest joints at zero shape."""
        J_np = self.spec.J_regressor.astype(np.float32) @ self.spec.v_template.astype(np.float32)
        return wp.from_numpy(J_np, device=self.device, dtype=wp.vec3)

    @cached_property
    def J_shapedirs(self) -> wp.array:
        """`J_regressor @ shapedirs` - `[J, num_betas]` vec3 rest-joint deltas per beta."""
        J_reg = self.spec.J_regressor.astype(np.float32)
        sd = self.spec.shapedirs.astype(np.float32)
        js = np.einsum("jv,vkl->jlk", J_reg, sd)
        return wp.from_numpy(np.ascontiguousarray(js), device=self.device, dtype=wp.vec3)

    @cached_property
    def v_template(self) -> wp.array:
        """`[V]` vec3 rest-pose vertices."""
        return wp.from_numpy(self.spec.v_template.astype(np.float32), device=self.device, dtype=wp.vec3)

    @cached_property
    def shapedirs_v3(self) -> wp.array:
        """`[V, num_betas]` vec3 shape-PCA basis (transposed from `(V, 3, num_betas)`)."""
        sd = self.spec.shapedirs.astype(np.float32)
        sd_v3 = np.ascontiguousarray(sd.transpose(0, 2, 1))
        return wp.from_numpy(sd_v3, device=self.device, dtype=wp.vec3)

    @cached_property
    def posedirs_v3(self) -> wp.array:
        """`[P, V]` vec3 pose-PCA basis (zero-copy view of `(P, V*3)`)."""
        P = self.spec.posedirs.shape[0]
        V = self.spec.v_template.shape[0]
        pd = np.ascontiguousarray(self.spec.posedirs.astype(np.float32).reshape(P, V, 3))
        return wp.from_numpy(pd, device=self.device, dtype=wp.vec3)

    @cached_property
    def lbs_weights(self) -> wp.array:
        """`[V, J]` float32 blend weights."""
        return wp.from_numpy(self.spec.lbs_weights.astype(np.float32), device=self.device, dtype=wp.float32)

    @cached_property
    def static_landmark_vertex_ids(self) -> wp.array:
        """`[L]` int32 landmark vertex indices."""
        assert self.spec.static_landmark_vertex_ids is not None, (
            f"Spec {self.spec.name!r} has no static_landmark_vertex_ids."
        )
        return wp.from_numpy(
            self.spec.static_landmark_vertex_ids.astype(np.int32),
            device=self.device,
            dtype=wp.int32,
        )

    # --- convenience scalars ---
    @property
    def num_joints(self) -> int:
        return int(self.spec.J_regressor.shape[0])

    @property
    def num_betas(self) -> int:
        return int(self.spec.num_betas)

    @property
    def num_vertices(self) -> int:
        return int(self.spec.v_template.shape[0])

    @property
    def num_pose_dirs(self) -> int:
        return int(self.spec.posedirs.shape[0])

    @property
    def num_static_landmarks(self) -> int:
        if self.spec.static_landmark_vertex_ids is None:
            return 0
        return int(self.spec.static_landmark_vertex_ids.shape[0])


# --- forward kinematics kernels --------------------------------------------
@wp.func
def body_aa_to_quat_func(aa: wp.vec3) -> wp.vec4:
    """Axis-angle → quaternion `wxyz`; `1e-8` nudge matches `smplx.lbs` epsilon."""
    eps = wp.float32(1.0e-8)
    nudged = wp.vec3(aa[0] + eps, aa[1] + eps, aa[2] + eps)
    angle = wp.length(nudged)
    rot_dir = aa / angle
    half = angle * 0.5
    c = wp.cos(half)
    s = wp.sin(half)
    return wp.vec4(c, rot_dir[0] * s, rot_dir[1] * s, rot_dir[2] * s)


@wp.kernel
def body_aa_to_quat_kernel(
    full_pose_aa: wp.array2d(dtype=wp.vec3),  # [B, J]
    quat_wxyz: wp.array2d(dtype=wp.vec4),  # [B, J]
):
    b, j = wp.tid()  # pyright: ignore[reportAssignmentType]
    quat_wxyz[b, j] = body_aa_to_quat_func(full_pose_aa[b, j])


@wp.kernel
def body_j_rest_kernel(
    betas: wp.array2d(dtype=wp.float32),  # [B, num_betas]
    J_template: wp.array1d(dtype=wp.vec3),  # [J]
    J_shapedirs: wp.array2d(dtype=wp.vec3),  # [J, num_betas]
    J_rest: wp.array2d(dtype=wp.vec3),  # [B, J]  (output)
):
    """Shape-dependent rest-pose joints: `J_rest = J_template + betas @ J_shapedirs`."""
    b, j = wp.tid()  # pyright: ignore[reportAssignmentType]
    result = J_template[j]
    num_betas = J_shapedirs.shape[1]
    for l in range(num_betas):
        result = result + betas[b, l] * J_shapedirs[j, l]
    J_rest[b, j] = result


@wp.kernel
def body_fk_kernel(
    quat_wxyz: wp.array2d(dtype=wp.vec4),  # [B, J]
    transl: wp.array1d(dtype=wp.vec3),  # [B]
    parents: wp.array1d(dtype=wp.int32),  # [J]
    J_rest: wp.array2d(dtype=wp.vec3),  # [B, J]
    T_world_joint: wp.array2d(dtype=wp_vec7),  # [B, J]  (output)
):
    """Sequential kinematic chain (one thread per body); requires `parents[j] < j`."""
    b = wp.tid()
    num_joints = parents.shape[0]

    for j in range(num_joints):
        q = quat_wxyz[b, j]
        parent = parents[j]
        if parent == -1:
            J_j = J_rest[b, j]
            T_world_joint[b, j] = wp_vec7(J_j[0], J_j[1], J_j[2], q[0], q[1], q[2], q[3])
        else:
            offset = J_rest[b, j] - J_rest[b, parent]
            T_local = wp_vec7(offset[0], offset[1], offset[2], q[0], q[1], q[2], q[3])
            T_world_joint[b, j] = se3_compose_func(T_world_joint[b, parent], T_local)

    t = transl[b]
    for j in range(num_joints):
        T = T_world_joint[b, j]
        T_world_joint[b, j] = wp_vec7(T[0] + t[0], T[1] + t[1], T[2] + t[2], T[3], T[4], T[5], T[6])


# --- linear blend skinning kernels -----------------------------------------
@wp.kernel
def body_v_shaped_kernel(
    betas: wp.array2d(dtype=wp.float32),  # [B, num_betas]
    v_template: wp.array1d(dtype=wp.vec3),  # [V]
    shapedirs_v3: wp.array2d(dtype=wp.vec3),  # [V, num_betas]
    v_shaped: wp.array2d(dtype=wp.vec3),  # [B, V]  (output)
):
    """Shape-deformed mesh: `v_shaped = v_template + shapedirs @ betas`."""
    b, v = wp.tid()  # pyright: ignore[reportAssignmentType]
    result = v_template[v]
    num_betas = shapedirs_v3.shape[1]
    for l in range(num_betas):
        result = result + betas[b, l] * shapedirs_v3[v, l]
    v_shaped[b, v] = result


@wp.kernel
def body_pose_feature_kernel(
    quat_wxyz: wp.array2d(dtype=wp.vec4),  # [B, J]
    pose_feature: wp.array2d(dtype=wp.float32),  # [B, (J-1)*9]  (output)
):
    """`pose_feature[b, p]` = element of `(R_j - I)` for the non-root joint `j = p // 9 + 1`."""
    b, p = wp.tid()  # pyright: ignore[reportAssignmentType]
    j = p // 9 + 1
    rc = p % 9
    row = rc // 3
    col = rc % 3
    R = quaternion_to_matrix_func(quat_wxyz[b, j])
    val = R[row, col]
    if row == col:
        val = val - 1.0
    pose_feature[b, p] = val


@wp.kernel
def body_pose_blend_kernel(
    pose_feature: wp.array2d(dtype=wp.float32),  # [B, P]
    posedirs_v3: wp.array2d(dtype=wp.vec3),  # [P, V]
    pose_offsets: wp.array2d(dtype=wp.vec3),  # [B, V]  (output)
):
    """`pose_offsets[b, v] = sum_p pose_feature[b, p] * posedirs[p, v]`."""
    b, v = wp.tid()  # pyright: ignore[reportAssignmentType]
    P = pose_feature.shape[1]
    offset = wp.vec3()
    for p in range(P):
        offset = offset + pose_feature[b, p] * posedirs_v3[p, v]
    pose_offsets[b, v] = offset


@wp.kernel
def body_relative_transforms_kernel(
    T_world_joint: wp.array2d(dtype=wp_vec7),  # [B, J]  (already translated)
    J_rest: wp.array2d(dtype=wp.vec3),  # [B, J]  (shape-dep, no transl)
    transl: wp.array1d(dtype=wp.vec3),  # [B]
    A: wp.array2d(dtype=wp.mat44),  # [B, J]  (output)
):
    """Per-joint LBS transform; cancels the rest-pose joint center and strips `transl` (re-added after skinning)."""
    b, j = wp.tid()  # pyright: ignore[reportAssignmentType]
    T = T_world_joint[b, j]
    tr = transl[b]
    t_local = wp.vec3(T[0] - tr[0], T[1] - tr[1], T[2] - tr[2])
    q = wp.vec4(T[3], T[4], T[5], T[6])
    R = quaternion_to_matrix_func(q)
    J_rb = J_rest[b, j]
    t_A = t_local - R * J_rb
    # fmt: off
    A[b, j] = wp.mat44(
        R[0, 0], R[0, 1], R[0, 2], t_A[0],
        R[1, 0], R[1, 1], R[1, 2], t_A[1],
        R[2, 0], R[2, 1], R[2, 2], t_A[2],
        0.0, 0.0, 0.0, 1.0,
    )
    # fmt: on


@wp.kernel
def body_lbs_kernel(
    v_shaped: wp.array2d(dtype=wp.vec3),  # [B, V]
    pose_offsets: wp.array2d(dtype=wp.vec3),  # [B, V]
    A: wp.array2d(dtype=wp.mat44),  # [B, J]
    lbs_weights: wp.array2d(dtype=wp.float32),  # [V, J]
    transl: wp.array1d(dtype=wp.vec3),  # [B]
    vertices_out: wp.array2d(dtype=wp.vec3),  # [B, V]  (output, world coords)
):
    """Per-vertex linear blend skinning; root `transl` added at the end."""
    b, v = wp.tid()  # pyright: ignore[reportAssignmentType]
    num_joints = A.shape[1]
    T_skin = wp.mat44()
    for j in range(num_joints):
        T_skin = T_skin + lbs_weights[v, j] * A[b, j]

    v_posed = v_shaped[b, v] + pose_offsets[b, v]
    v_h = wp.vec4(v_posed[0], v_posed[1], v_posed[2], 1.0)
    out = T_skin * v_h
    tr = transl[b]
    vertices_out[b, v] = wp.vec3(out[0] + tr[0], out[1] + tr[1], out[2] + tr[2])


@wp.kernel
def body_landmark_gather_kernel(
    vertices: wp.array2d(dtype=wp.vec3),  # [B, V]
    landmark_vertex_ids: wp.array1d(dtype=wp.int32),  # [L]
    landmarks_out: wp.array2d(dtype=wp.vec3),  # [B, L]  (output)
):
    """`landmarks[b, k] = vertices[b, landmark_vertex_ids[k]]`."""
    b, k = wp.tid()  # pyright: ignore[reportAssignmentType]
    landmarks_out[b, k] = vertices[b, landmark_vertex_ids[k]]


# --- Python launchers ------------------------------------------------------
def _run_fk_stages(
    spec_tensors: BodyModelSpecTensors,
    state: BodyModelState,
) -> Tuple[wp.array, wp.array, wp.array, wp.array, wp.array]:
    """Run the three FK kernels and return `(betas, transl, quat, J_rest, T_world_joint)`."""
    device = spec_tensors.device
    B = state.batch_size
    J = spec_tensors.num_joints

    full_pose_wp = wp.from_numpy(state.full_pose_aa, dtype=wp.vec3, device=device)
    betas_wp = wp.from_numpy(state.betas, dtype=wp.float32, device=device)
    transl_wp = wp.from_numpy(state.transl, dtype=wp.vec3, device=device)

    quat_wxyz = wp.empty((B, J), dtype=wp.vec4, device=device)
    J_rest = wp.empty((B, J), dtype=wp.vec3, device=device)
    T_world_joint = wp.empty((B, J), dtype=wp_vec7, device=device)

    wp.launch(
        body_aa_to_quat_kernel,
        dim=(B, J),
        inputs=[full_pose_wp],
        outputs=[quat_wxyz],
        device=device,
    )
    wp.launch(
        body_j_rest_kernel,
        dim=(B, J),
        inputs=[betas_wp, spec_tensors.J_template, spec_tensors.J_shapedirs],
        outputs=[J_rest],
        device=device,
    )
    wp.launch(
        body_fk_kernel,
        dim=B,
        inputs=[quat_wxyz, transl_wp, spec_tensors.parent_joint_indices, J_rest],
        outputs=[T_world_joint],
        device=device,
    )
    return betas_wp, transl_wp, quat_wxyz, J_rest, T_world_joint


def body_fk_warp(
    spec_tensors: BodyModelSpecTensors,
    state: BodyModelState,
) -> wp.array:
    """Skeletal forward kinematics; returns `wp.array2d[wp_vec7]` of shape `[B, J]`."""
    return _run_fk_stages(spec_tensors, state)[-1]


def body_lbs_warp(
    spec_tensors: BodyModelSpecTensors,
    state: BodyModelState,
    return_landmarks: bool = True,
) -> Dict[str, wp.array]:
    """FK + LBS; returns `{T_world_joint, vertices[, landmarks]}` (all `wp.array`)."""
    device = spec_tensors.device
    B = state.batch_size
    J = spec_tensors.num_joints
    V = spec_tensors.num_vertices
    P = spec_tensors.num_pose_dirs

    betas_wp, transl_wp, quat_wxyz, J_rest, T_world_joint = _run_fk_stages(spec_tensors, state)

    v_shaped = wp.empty((B, V), dtype=wp.vec3, device=device)
    pose_feature = wp.empty((B, P), dtype=wp.float32, device=device)
    pose_offsets = wp.empty((B, V), dtype=wp.vec3, device=device)
    A = wp.empty((B, J), dtype=wp.mat44, device=device)
    vertices = wp.empty((B, V), dtype=wp.vec3, device=device)

    wp.launch(
        body_v_shaped_kernel,
        dim=(B, V),
        inputs=[betas_wp, spec_tensors.v_template, spec_tensors.shapedirs_v3],
        outputs=[v_shaped],
        device=device,
    )
    wp.launch(
        body_pose_feature_kernel,
        dim=(B, P),
        inputs=[quat_wxyz],
        outputs=[pose_feature],
        device=device,
    )
    wp.launch(
        body_pose_blend_kernel,
        dim=(B, V),
        inputs=[pose_feature, spec_tensors.posedirs_v3],
        outputs=[pose_offsets],
        device=device,
    )
    wp.launch(
        body_relative_transforms_kernel,
        dim=(B, J),
        inputs=[T_world_joint, J_rest, transl_wp],
        outputs=[A],
        device=device,
    )
    wp.launch(
        body_lbs_kernel,
        dim=(B, V),
        inputs=[v_shaped, pose_offsets, A, spec_tensors.lbs_weights, transl_wp],
        outputs=[vertices],
        device=device,
    )

    out: Dict[str, wp.array] = {
        "T_world_joint": T_world_joint,
        "vertices": vertices,
    }
    if return_landmarks:
        L = spec_tensors.num_static_landmarks
        landmarks = wp.empty((B, L), dtype=wp.vec3, device=device)
        wp.launch(
            body_landmark_gather_kernel,
            dim=(B, L),
            inputs=[vertices, spec_tensors.static_landmark_vertex_ids],
            outputs=[landmarks],
            device=device,
        )
        out["landmarks"] = landmarks
    return out
