"""Spheres covering a box or a cylinder, the collision geometry of a held object."""

from typing import Literal, Optional, Sequence

import numpy as np


def _compute_covering_spheres(
    shape: Literal["box", "cylinder"], size: Sequence[float], T_frame_shape: Optional[np.ndarray] = None
) -> np.ndarray:
    """Spheres `(n, 4)` `(x, y, z, radius)` covering a box (half extents) or a z-axis cylinder (radius, half height),
    posed by 4x4 `T_frame_shape`: each encloses a cell of a grid no wider than the larger of the smallest half extent
    and a quarter of the largest, so it bulges past each face by up to the radius less half the cell (e.g. 0.37 h for
    a cube of half side h)."""
    half = np.array([*size] if shape == "box" else [size[0], size[0], size[1]], dtype=np.float64)
    counts = np.ceil(2 * half / max(half.min(), half.max() / 4) - 1e-9).astype(int)
    cell = 2 * half / counts
    centers = (np.indices(counts).reshape(3, -1).T + 0.5) * cell - half
    if shape == "cylinder":  # the cells that reach it
        centers = centers[np.linalg.norm(np.maximum(np.abs(centers[:, :2]) - cell[:2] / 2, 0), axis=1) < size[0]]
    T = np.eye(4) if T_frame_shape is None else np.asarray(T_frame_shape)
    points = centers @ T[:3, :3].T + T[:3, 3]
    return np.c_[points, np.full(len(points), 0.5 * np.linalg.norm(cell))]
