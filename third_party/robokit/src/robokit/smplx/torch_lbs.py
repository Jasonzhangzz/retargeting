"""Torch reference LBS + differentiable MANO forward; lazily exported so the warp path never imports `torch`."""

from typing import Dict, Tuple

import torch

from robokit.smplx.constants import MANO_NUM_JOINTS, MANOPTH_KEYPOINT_PERMUTATION
from robokit.smplx.spec import BodyModelSpec, BodyModelState
from robokit.xform.warp.torch_wrappers import matrix_to_quaternion


def body_lbs_torch(
    spec: BodyModelSpec,
    state: BodyModelState,
    return_landmarks: bool = True,
) -> Dict[str, torch.Tensor]:
    """FK + LBS on torch; returns `{vertices, joints, global_T_world_joint_xyz_wxyz[, landmarks]}` in world coords."""
    # as_tensor: state may carry numpy or torch; no copy, autograd preserved
    betas_t = torch.as_tensor(state.betas, dtype=torch.float32)
    full_pose_t = torch.as_tensor(state.full_pose_aa, dtype=torch.float32)
    transl_t = torch.as_tensor(state.transl, dtype=torch.float32)

    device = full_pose_t.device
    dtype = full_pose_t.dtype
    B = state.batch_size
    V = spec.v_template.shape[0]
    J = spec.J_regressor.shape[0]

    v_template = torch.from_numpy(spec.v_template).to(device=device, dtype=dtype)
    shapedirs = torch.from_numpy(spec.shapedirs).to(device=device, dtype=dtype)
    posedirs = torch.from_numpy(spec.posedirs).to(device=device, dtype=dtype)
    J_regressor = torch.from_numpy(spec.J_regressor).to(device=device, dtype=dtype)
    lbs_weights = torch.from_numpy(spec.lbs_weights).to(device=device, dtype=dtype)
    parents = spec.parents

    v_shaped = v_template + torch.einsum("bl,vkl->bvk", betas_t, shapedirs)
    J_rest = torch.einsum("bvk,jv->bjk", v_shaped, J_regressor)

    # Rodrigues; the 1e-8 nudge matches `smplx.lbs` epsilon handling
    pose_flat = full_pose_t.reshape(-1, 3)
    angle = torch.norm(pose_flat + 1e-8, dim=1, keepdim=True)
    rot_dir = pose_flat / angle
    cos = torch.cos(angle).unsqueeze(-1)
    sin = torch.sin(angle).unsqueeze(-1)
    rx, ry, rz = rot_dir.unbind(dim=1)
    zero = torch.zeros_like(rx)
    K = torch.stack(
        [
            torch.stack([zero, -rz, ry], dim=1),
            torch.stack([rz, zero, -rx], dim=1),
            torch.stack([-ry, rx, zero], dim=1),
        ],
        dim=1,
    )
    ident = torch.eye(3, device=device, dtype=dtype).expand_as(K)
    rot_mats = (ident + sin * K + (1.0 - cos) * torch.bmm(K, K)).reshape(B, J, 3, 3)

    ident3 = torch.eye(3, device=device, dtype=dtype)
    pose_feature = (rot_mats[:, 1:] - ident3).reshape(B, -1)
    pose_offsets = (pose_feature @ posedirs).reshape(B, V, 3)
    v_posed = v_shaped + pose_offsets

    rel_joints = J_rest.clone()
    rel_joints[:, 1:] = rel_joints[:, 1:] - J_rest[:, parents[1:]]

    T_local = torch.zeros(B, J, 4, 4, device=device, dtype=dtype)
    T_local[:, :, :3, :3] = rot_mats
    T_local[:, :, :3, 3] = rel_joints
    T_local[:, :, 3, 3] = 1.0

    T_world_joint_list = [T_local[:, 0]]
    for i in range(1, J):
        T_world_joint_list.append(T_world_joint_list[int(parents[i])] @ T_local[:, i])
    T_world_joint = torch.stack(T_world_joint_list, dim=1)

    J_posed = T_world_joint[:, :, :3, 3]

    # cancel out the rest-pose joint center so LBS rotations pivot correctly
    J_rest_hzero = torch.cat([J_rest, torch.zeros(B, J, 1, device=device, dtype=dtype)], dim=-1).unsqueeze(-1)
    offset = (T_world_joint @ J_rest_hzero).squeeze(-1)
    A = T_world_joint.clone()
    A[:, :, :, 3] = A[:, :, :, 3] - offset

    T_skin = torch.einsum("vj,bjnm->bvnm", lbs_weights, A)
    v_posed_h = torch.cat([v_posed, torch.ones(B, V, 1, device=device, dtype=dtype)], dim=-1)
    verts = (T_skin @ v_posed_h.unsqueeze(-1)).squeeze(-1)[:, :, :3]

    transl_b1x3 = transl_t.unsqueeze(1)
    verts_world = verts + transl_b1x3
    joints_world = J_posed + transl_b1x3

    quat_wxyz = matrix_to_quaternion(T_world_joint[:, :, :3, :3])
    global_T_world_joint = torch.cat([joints_world, quat_wxyz], dim=-1)

    out: Dict[str, torch.Tensor] = {
        "vertices": verts_world,
        "joints": joints_world,
        "global_T_world_joint_xyz_wxyz": global_T_world_joint,
    }
    if return_landmarks and spec.static_landmark_vertex_ids is not None:
        lmk_idx = torch.from_numpy(spec.static_landmark_vertex_ids).to(device=device)
        out["landmarks"] = verts_world.index_select(dim=1, index=lmk_idx)
    return out


def mano_forward_pca_mm(
    spec: BodyModelSpec,
    pose_48: torch.Tensor,
    betas_10: torch.Tensor,
    transl: torch.Tensor,
    flat_hand_mean: bool,
    use_pca: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Manopth-compatible MANO forward in mm; returns `(vertices_mm, joints_21_mm)` in OpenPose keypoint order."""
    B = pose_48.shape[0]
    global_orient = pose_48[:, :3]
    finger_input = pose_48[:, 3:48]
    if use_pca:
        hands_components = torch.from_numpy(spec.metadata["hands_components"]).to(pose_48)
        finger_input = finger_input @ hands_components
    if not flat_hand_mean:
        hands_mean = torch.from_numpy(spec.metadata["hands_mean"]).to(pose_48)
        finger_input = finger_input + hands_mean
    full_pose_aa = torch.cat([global_orient.unsqueeze(1), finger_input.reshape(B, MANO_NUM_JOINTS - 1, 3)], dim=1)

    state = BodyModelState(betas=betas_10, full_pose_aa=full_pose_aa, transl=transl)
    out = body_lbs_torch(spec, state, return_landmarks=True)
    joints_native = torch.cat([out["joints"], out["landmarks"]], dim=1)
    perm = torch.as_tensor(MANOPTH_KEYPOINT_PERMUTATION, device=joints_native.device, dtype=torch.long)
    joints_21 = joints_native.index_select(dim=1, index=perm)
    return out["vertices"] * 1000.0, joints_21 * 1000.0
