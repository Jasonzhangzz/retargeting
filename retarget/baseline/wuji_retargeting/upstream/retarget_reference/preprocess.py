"""Independent NumPy reconstruction of SDK 2026.8.31 input/target preparation.

Recovered from pinned wuji_sdk.so core VA 0x10bc16d and embedded hand YAML.
No SDK calls. The palm frame uses the native Gram-eigen/signed-sort/QR path,
including rank-deficient basis selection; exception classes differ from SDK.
"""
import math
import numpy as np
try:
    from .eigen import palm_normal
    from ._numeric import ordered_matmul
except ImportError:  # Also permit standalone analysis-module loading.
    from eigen import palm_normal
    from _numeric import ordered_matmul

CHAINS = tuple((0, *range(1 + 4*i, 5 + 4*i)) for i in range(5))
OPERATOR2MANO_RIGHT = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
OPERATOR2MANO_LEFT = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])


def _dot3(a, b):
    return ((float(a[0])*float(b[0]) + float(a[1])*float(b[1]))
            + float(a[2])*float(b[2]))


def estimate_frame(points):
    """Return columns x, palm normal, x-cross-normal from landmarks 0/5/9."""
    p = np.asarray(points, dtype=np.float64)[[0, 5, 9]]
    v = p[0] - p[2]
    c = p - ((p[0] + p[1]) + p[2]) / 3.
    normal = palm_normal(c)
    x = v - _dot3(v, normal) * normal
    length = math.sqrt(_dot3(x, x))
    if not np.isfinite(length) or length == 0.:
        raise ValueError('degenerate wrist/palm frame')
    x /= length
    z = np.cross(x, normal)
    if _dot3(z, p[1] - p[2]) < 0.:
        normal *= -1.
        z *= -1.
    return np.column_stack((x, normal, z))


def apply_pinch(points, threshold_m=.009, sigmoid_scale_m=.0015):
    """Blend all four close fingertip pairs toward one weighted midpoint."""
    result = np.asarray(points, dtype=np.float64).copy()
    t, s = float(threshold_m), float(sigmoid_scale_m)
    if not (np.isfinite(t) and np.isfinite(s) and t > 0. and s > 0.):
        return result
    tips = np.array([8, 12, 16, 20])
    thumb, fingers = result[4].copy(), result[tips].copy()
    if not (np.isfinite(thumb).all() and np.isfinite(fingers).all()):
        return result
    distances = np.linalg.norm(fingers - thumb, axis=1)
    # Arguments are nonnegative on the active branch, so no overflow occurs.
    denominator = 1. / (1. + np.exp(-t / s)) - .5
    if denominator <= np.finfo(float).eps:
        return result
    weights = np.zeros(4)
    active = distances < t
    weights[active] = np.clip((1. / (1. + np.exp(-(t-distances[active])/s))-.5)
                              / denominator, 0., 1.)
    total = weights.sum()
    if total <= 0.:
        return result
    center = (weights[:, None] * fingers).sum(axis=0) / total
    midpoint = (thumb + center) * .5
    thumb_weight = 1. - np.prod(1. - weights)
    result[4] = thumb + thumb_weight * (midpoint - thumb)
    result[tips] = fingers + weights[:, None] * (midpoint - fingers)
    return result


def preprocess(points, model='WujiHand', side='Right', *, pinch=True,
               rotation_degrees=None, wrist_offset_m=None, thumb_offset_m=None):
    """Flatten/C-order, cast f32, center, palm-align, rotate/offset, then pinch.

    Extra keyword overrides expose recovered configuration semantics; built-in
    sessions have zero offsets and Z rotations 0 / -90 degrees for Hand1 / Hand2.
    """
    flat = np.asarray(points).reshape(63)
    # The native Python binding extracts numeric scalars; unlike astype it does
    # not parse text or silently discard the imaginary component of complex.
    if flat.dtype.kind in 'US':
        raise TypeError('must be real number, not str')
    if flat.dtype.kind == 'c':
        raise TypeError('must be real number, not complex')
    if flat.dtype.kind == 'O':
        for value in flat:
            if isinstance(value, (str, bytes, complex)):
                raise TypeError('must be real number, not ' + type(value).__name__)
    p = flat.astype(np.float32).astype(np.float64).reshape(21, 3)
    if not np.isfinite(p).all():
        raise ValueError('input landmarks must be finite')
    side_key = str(getattr(side, 'name', side)).lower().split('.')[-1]
    if side_key not in ('left', 'right'):
        raise ValueError('side must be Left or Right')
    model_key = str(getattr(model, 'name', model)).lower().split('.')[-1]
    if model_key in ('wujihand', 'hand1', 'wujihand1'):
        default_rotation = (0., 0., 0.)
    elif model_key in ('wujihand2', 'hand2'):
        default_rotation = (0., 0., -90.)
    else:
        raise ValueError('model must be WujiHand or WujiHand2')
    p -= p[0].copy()
    operator = OPERATOR2MANO_RIGHT if side_key == 'right' else OPERATOR2MANO_LEFT
    p = ordered_matmul(ordered_matmul(p, estimate_frame(p)), operator)
    angles = np.deg2rad(default_rotation if rotation_degrees is None else rotation_degrees)
    if np.any(angles):
        x, y, z = angles
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)
        rx = np.array([[1.,0.,0.],[0.,cx,-sx],[0.,sx,cx]])
        ry = np.array([[cy,0.,sy],[0.,1.,0.],[-sy,0.,cy]])
        rz = np.array([[cz,-sz,0.],[sz,cz,0.],[0.,0.,1.]])
        p = ordered_matmul(p, ordered_matmul(ordered_matmul(rz, ry), rx).T)
    if wrist_offset_m is not None:
        p[5:] += np.asarray(wrist_offset_m, dtype=float)
    if thumb_offset_m is not None:
        p[1:5] += np.asarray(thumb_offset_m, dtype=float)
    result = apply_pinch(p) if pinch else p
    if np.max(np.abs(result)) <= 1e-8:
        raise ValueError('invalid keypoints: all zeros')
    return result


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
