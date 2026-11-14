"""Differentiable rotation conversion and rigid-frame invariant pose features."""
import math
import torch
import torch.nn.functional as F


def axis_angle_to_matrix(aa):
    angle = torch.linalg.vector_norm(aa, dim=-1)
    x, y, z = aa.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*aa.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device)
    return eye + torch.sinc(angle / math.pi)[..., None, None] * skew + (
        .5 * torch.sinc(angle / (2 * math.pi)).square()
    )[..., None, None] * (skew @ skew)


def matrix_to_rotation6d(matrix):
    return matrix[..., :, :2].transpose(-1, -2).flatten(-2)


def rotation6d_to_matrix(value):
    a, b = value[..., :3], value[..., 3:]
    x = F.normalize(a, dim=-1)
    y = F.normalize(b - (x * b).sum(-1, keepdim=True) * x, dim=-1)
    return torch.stack((x, y, torch.linalg.cross(x, y, dim=-1)), -1)


def pose_features(pose51):
    """Physical wrist xyz/rotvec + 15 absolute local joint rotvecs -> 99D.

    All wrist transforms must be expressed in the same coordinate frame within
    a window. Shape, image, object identity, and world origin never enter here.
    """
    if pose51.ndim != 3 or pose51.shape[-1] != 51 or not 1 <= pose51.shape[1] <= 8:
        raise ValueError("Expected pose51 [B,T,51] with 1 <= T <= 8")
    if not torch.isfinite(pose51).all():
        raise ValueError("Nonfinite MANO pose")
    rotations = axis_angle_to_matrix(pose51[..., 3:6])
    anchor_inverse = rotations[:, :1].transpose(-1, -2)
    translation = (anchor_inverse @ (pose51[..., :3] - pose51[:, :1, :3])[..., None]).squeeze(-1)
    wrist = matrix_to_rotation6d(anchor_inverse @ rotations)
    joints = matrix_to_rotation6d(axis_angle_to_matrix(pose51[..., 6:].reshape(*pose51.shape[:2], 15, 3)))
    return torch.cat((translation / .1, wrist, joints.flatten(-2)), -1)


def decode_features(features, anchor_pose51):
    """Recover physical wrist and absolute local rotations; inference utility."""
    from scipy.spatial.transform import Rotation
    import numpy as np
    x = features.detach().cpu().numpy()
    anchor = anchor_pose51.detach().cpu().numpy()
    relative = rotation6d_to_matrix(features[..., 3:9]).detach().cpu().numpy()
    a = Rotation.from_rotvec(anchor[:, 3:6]).as_matrix()
    out = np.empty((*x.shape[:2], 51), np.float32)
    out[..., :3] = np.einsum('bij,btj->bti', a, .1*x[..., :3]) + anchor[:, None, :3]
    out[..., 3:6] = Rotation.from_matrix((a[:, None] @ relative).reshape(-1, 3, 3)).as_rotvec().reshape(*x.shape[:2], 3)
    j = rotation6d_to_matrix(features[..., 9:].reshape(*features.shape[:2], 15, 6))
    out[..., 6:] = Rotation.from_matrix(j.detach().cpu().numpy().reshape(-1, 3, 3)).as_rotvec().reshape(*x.shape[:2], 45)
    return out
