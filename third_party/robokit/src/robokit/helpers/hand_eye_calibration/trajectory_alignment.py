"""Source-frame selection and coupled image blocks ordered as [camera, q]."""

import numpy as np
import torch

from robokit.terms.sparse.trajectory_smoothness_task import _temporal_differences


def initialization_frames(visible: np.ndarray, times: np.ndarray, count: int, guided: np.ndarray) -> np.ndarray:
    """Search jointly visible interval boundaries, then spread guided frames; visibility `[T,V]` prefers two views, `[T]` retains supplied visibility, while absent or partial masks still constrain refinement."""
    if visible.ndim == 2:
        counts = visible.sum(-1)
        visible = counts >= max(1, min(2, int(counts.max())))
    transitions = np.flatnonzero(visible[1:] != visible[:-1]) + 1
    chosen = np.unique(np.r_[0, len(times) - 1, transitions - 1, transitions])
    if visible.any():
        chosen = chosen[visible[chosen]]
    if len(chosen) > count:
        return chosen[np.linspace(0, len(chosen) - 1, count).round().astype(int)]
    pool = np.flatnonzero(visible & guided)
    if not len(pool):
        pool = np.flatnonzero(visible)
    if not len(pool):
        pool = np.arange(len(times))
    while len(chosen) < count:
        remaining = np.setdiff1d(pool, chosen)
        if not len(remaining):
            break
        distance = np.abs(times[remaining, None] - times[chosen][None]).min(axis=1)
        chosen = np.sort(np.r_[chosen, remaining[distance.argmax()]])
    return chosen


def trajectory_normal_equations(
    q: torch.Tensor,
    normal: torch.Tensor,
    gradient: torch.Tensor,
    times: torch.Tensor,
    velocity_weight: float = 0.015,
    acceleration_weight: float = 0.02,
    free_cameras: bool = False,
) -> tuple:
    """Assemble one coupled system from [views, frames, 6+joints] image blocks; joint smoothness uses source timestamps, no recorded states or task-specific priors.

    >>> # xdoctest: +REQUIRES(module:nvdiffrast)
    >>> q = torch.zeros((3, 1))
    >>> h = torch.zeros((1, 3, 7, 7))
    >>> g = torch.zeros((1, 3, 7))
    >>> matrix, vector, cost = trajectory_normal_equations(q, h, g, torch.arange(3.))
    >>> matrix.shape, vector.shape, cost.item()
    (torch.Size([3, 3]), torch.Size([3]), 0.0)
    """
    # --- joint blocks ---
    views, frames, width = gradient.shape
    joints = width - 6
    camera_dim = 6 * views if free_cameras else 0
    dimension = frames * joints + camera_dim
    matrix = torch.zeros((dimension, dimension), device=q.device, dtype=q.dtype)
    vector = torch.zeros(dimension, device=q.device, dtype=q.dtype)
    joint_dim = frames * joints
    indices = torch.arange(joint_dim, device=q.device).reshape(frames, joints)
    matrix[indices[:, :, None], indices[:, None, :]] = normal[:, :, 6:, 6:].sum(0)
    vector[:joint_dim] = gradient[:, :, 6:].sum(0).flatten()
    # --- shared cameras and cross terms ---
    if free_cameras:
        camera_indices = torch.arange(camera_dim, device=q.device).reshape(views, 6) + joint_dim
        matrix[camera_indices[:, :, None], camera_indices[:, None, :]] = normal[:, :, :6, :6].sum(1)
        matrix[:joint_dim, joint_dim:] = normal[:, :, 6:, :6].permute(1, 2, 0, 3).reshape(joint_dim, camera_dim)
        matrix[joint_dim:, :joint_dim] = normal[:, :, :6, 6:].permute(0, 2, 1, 3).reshape(camera_dim, joint_dim)
        vector[joint_dim:] = gradient[:, :, :6].sum(1).flatten()
    # --- temporal coupling ---
    temporal_cost = torch.zeros((), device=q.device, dtype=q.dtype)
    for difference in _temporal_differences(times, velocity_weight, acceleration_weight):
        residual = difference @ q
        temporal_cost += 0.5 * residual.square().sum()
        coupling = torch.kron(difference.T @ difference, torch.eye(joints, device=q.device, dtype=q.dtype))
        matrix[: frames * joints, : frames * joints] += coupling
        vector[: frames * joints] += (difference.T @ residual).flatten()
    return matrix, vector, temporal_cost
