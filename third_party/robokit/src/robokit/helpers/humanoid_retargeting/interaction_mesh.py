"""Build interaction-mesh targets for humanoid retargeting."""

from typing import Tuple

import numpy as np

from robokit.xform.numpy import quaternion_to_matrix


def _build_delaunay_graph(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a uniformly weighted CSR graph from a 3D Delaunay tetrahedralization."""
    from scipy.spatial import Delaunay

    tetrahedra = Delaunay(np.asarray(points, dtype=np.float64))
    neighbors = [set() for _ in points]
    for simplex in tetrahedra.simplices:
        for a in simplex:
            for b in simplex:
                if a != b:
                    neighbors[int(a)].add(int(b))

    offsets = np.zeros(len(points) + 1, dtype=np.int32)
    indices = []
    weights = []
    for i, vertex_neighbors in enumerate(neighbors):
        vertex_neighbors = sorted(vertex_neighbors)
        offsets[i + 1] = offsets[i] + len(vertex_neighbors)
        indices.extend(vertex_neighbors)
        if vertex_neighbors:
            weights.extend([1.0 / len(vertex_neighbors)] * len(vertex_neighbors))
    return offsets, np.asarray(indices, dtype=np.int32), np.asarray(weights, dtype=np.float32)


def _compute_laplacian_coordinates(
    points: np.ndarray, offsets: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Compute `p_i - sum_j w_ij p_j` for every vertex."""
    points = np.asarray(points, dtype=np.float32)
    coordinates = points.copy()
    rows = np.repeat(np.arange(len(points)), np.diff(offsets))
    np.add.at(coordinates, rows, -weights[:, None] * points[indices])
    return coordinates


def build_interaction_mesh_frame(
    human_key_positions_world: np.ndarray,
    object_points_local: np.ndarray,
    T_world_object: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the graph and target Laplacian consumed by `InteractionMeshTask.set_frame()`.

    Example:
        >>> human = np.array([[0, 0, 0]], dtype=np.float32)
        >>> object_points = np.eye(3, dtype=np.float32)
        >>> object_points = np.vstack([object_points, np.ones(3, dtype=np.float32)])
        >>> T_world_object = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        >>> frame = build_interaction_mesh_frame(human, object_points, T_world_object)
        >>> frame[3].shape, frame[4].shape
        ((5, 3), (5, 3))
    """
    object_positions_world = object_points_local @ quaternion_to_matrix(T_world_object[3:7]).T + T_world_object[:3]
    reference_vertex_positions = np.vstack([human_key_positions_world, object_positions_world]).astype(np.float32)
    neighbor_offsets, neighbor_indices, neighbor_weights = _build_delaunay_graph(reference_vertex_positions)
    target_laplacian = _compute_laplacian_coordinates(
        reference_vertex_positions, neighbor_offsets, neighbor_indices, neighbor_weights
    )
    return (
        neighbor_offsets,
        neighbor_indices,
        neighbor_weights,
        reference_vertex_positions,
        target_laplacian,
    )


__all__ = ["build_interaction_mesh_frame"]
