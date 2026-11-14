"""Source graph preparation excerpt; provenance and hashes in SOURCE.json."""
import math
import numpy as np
from ._numeric import ordered_matmul

CHAINS = tuple((0, *range(1 + 4*i, 5 + 4*i)) for i in range(5))


def prepare_targets(S, thumb_free=True, eps=1e-8, distance_scale=50.,
                    thumb_rel_skip_proximal=1):
    """Build source directed normalized Laplacian and direction targets.

    Keys: S, L, D, edges, directions, edge_lengths, neighbor_weights,
    row_weights, thumb_triplets. L @ S == D. Edges preserve chain order.
    """
    S = np.asarray(S, dtype=np.float64).reshape(21, 3).copy()
    if not np.isfinite(S).all():
        raise ValueError('source landmarks must be finite')
    neighbor_weights = np.ones(21)
    row_weights = np.ones(21)
    if thumb_free:
        neighbor_weights[1:4] = 0.
        row_weights[1:4] = 0.
    delta = S[:, None, :] - S[None, :, :]
    distances = np.sqrt((delta[..., 0]*delta[..., 0] + delta[..., 1]*delta[..., 1])
                        + delta[..., 2]*delta[..., 2])
    # The native graph calls scalar libm exp and sums each row left to right.
    # NumPy's vector exp/reduction are mathematically equivalent, but their
    # rounding can change an SLSQP stopping decision on later frames.
    A = np.array([[math.exp(-float(distance_scale)*float(d)) for d in row]
                  for row in distances]) * neighbor_weights[None, :]
    np.fill_diagonal(A, 0.)
    sums = np.zeros(21)
    for column in range(21):
        sums += A[:, column]
    L = np.eye(21)
    for i in range(21):
        if sums[i] > eps:
            L[i] -= A[i] / sums[i]
        else:
            L[i] = -1. / 20.
            L[i, i] = 1.
    all_edges = [(a, b) for chain in CHAINS[int(bool(thumb_free)):]
                 for a, b in zip(chain[:-1], chain[1:])]
    edges = np.asarray(all_edges, dtype=np.int64)
    vectors = S[edges[:, 1]] - S[edges[:, 0]]
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > eps
    edges, vectors, lengths = edges[valid], vectors[valid], lengths[valid]
    if not len(edges):
        raise ValueError('no nondegenerate source edges')
    skip = min(max(int(thumb_rel_skip_proximal), 0), 3)
    triplets = [(0, 1, 2), (1, 2, 3), (2, 3, 4)][skip:] if thumb_free else []
    return dict(S=S, L=L, D=ordered_matmul(L,S), edges=edges, directions=vectors/lengths[:, None],
                edge_lengths=lengths, neighbor_weights=neighbor_weights,
                row_weights=row_weights, thumb_triplets=np.asarray(triplets, dtype=np.int64).reshape(-1, 3))
