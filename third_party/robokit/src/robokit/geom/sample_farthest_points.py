# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportReturnType=false
# pyright: reportGeneralTypeIssues=false
from typing import TYPE_CHECKING

import warp as wp


if TYPE_CHECKING:
    import torch


@wp.func
def squared_distance(p1: wp.vec3, p2: wp.vec3) -> wp.float32:
    diff = p1 - p2
    return wp.dot(diff, diff)


@wp.kernel
def fps_fused_kernel(
    points: wp.array2d(dtype=wp.vec3),
    lengths: wp.array(dtype=wp.int32),
    sampled_indices: wp.array2d(dtype=wp.int32),
    min_dists: wp.array2d(dtype=wp.float32),
    max_samples: wp.int32,
    max_points: wp.int32,
):
    batch_idx = wp.tid()
    n = lengths[batch_idx]

    for i in range(max_points):
        if i < n:
            min_dists[batch_idx, i] = wp.float32(1e10)

    sampled_indices[batch_idx, 0] = wp.int32(0)

    max_dist = wp.float32(-1.0)
    farthest_idx = wp.int32(0)

    for sample_idx in range(1, max_samples):
        selected_idx = sampled_indices[batch_idx, sample_idx - 1]
        selected_point = points[batch_idx, selected_idx]

        max_dist = wp.float32(-1.0)
        farthest_idx = wp.int32(0)

        for i in range(max_points):
            if i < n:
                current_point = points[batch_idx, i]
                dist2 = squared_distance(current_point, selected_point)
                min_dists[batch_idx, i] = wp.min(min_dists[batch_idx, i], dist2)

                if min_dists[batch_idx, i] > max_dist:
                    max_dist = min_dists[batch_idx, i]
                    farthest_idx = wp.int32(i)

        sampled_indices[batch_idx, sample_idx] = farthest_idx


def sample_farthest_points(
    points: "torch.Tensor",
    points_offsets: "torch.Tensor",
    num_samples: "torch.Tensor",
) -> "torch.Tensor":
    """Sample farthest points from batched point clouds using FPS algorithm.

    Args:
        points: All points concatenated, shape [total_points, 3]
        points_offsets: CSR batch boundaries, shape [num_batches + 1] (last = total_points)
        num_samples: Number of points to sample per batch, shape [num_batches]

    Returns:
        Sampled point indices (local to each batch), shape [num_batches, max_samples]

    Example:
        >>> import torch
        >>> points = torch.tensor([[0.0, 0, 0], [1, 0, 0], [10, 0, 0], [0, 0, 0], [0, 5, 0]])  # clouds of 3 and 2
        >>> sample_farthest_points(points, torch.tensor([0, 3, 5]), torch.tensor([2, 2])).tolist()
        [[0, 2], [0, 1]]
    """
    import torch

    wp.init()
    device = points.device
    wp_device = wp.device_from_torch(device)

    num_batches = points_offsets.shape[0] - 1
    max_samples = int(num_samples.max().item())

    lengths = (points_offsets[1:] - points_offsets[:-1]).to(torch.int32)
    max_points = int(lengths.max().item())

    point_indices = torch.arange(max_points, device=device).unsqueeze(0).expand(num_batches, -1)
    global_indices = points_offsets[:-1].unsqueeze(1) + point_indices
    valid_mask = point_indices < lengths.unsqueeze(1)
    global_indices = global_indices.clamp(max=points.shape[0] - 1)

    points_2d = torch.zeros(num_batches, max_points, 3, device=device, dtype=points.dtype)
    points_2d[valid_mask] = points[global_indices[valid_mask]]

    min_dists = torch.empty(num_batches, max_points, device=device, dtype=torch.float32)
    sampled_indices = torch.zeros(num_batches, max_samples, device=device, dtype=torch.int32)

    points_wp = wp.from_torch(points_2d, dtype=wp.vec3)
    min_dists_wp = wp.from_torch(min_dists)
    lengths_wp = wp.from_torch(lengths)
    sampled_indices_wp = wp.from_torch(sampled_indices)

    wp.launch(
        fps_fused_kernel,
        dim=num_batches,
        inputs=[points_wp, lengths_wp, sampled_indices_wp, min_dists_wp, max_samples, max_points],
        device=wp_device,
    )

    return sampled_indices


__all__ = ["sample_farthest_points"]
