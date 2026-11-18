"""Float64 evaluation order recovered from SDK 2026.8.31 scalar helpers.

The optimization callbacks are sensitive to rounding. These operations keep
the multiplication/addition order of ELF helpers 0x10a96f1, 0x10a9175 and
0x10aa362, without a BLAS reduction or fused multiply-add.
"""
import math
import numpy as np


def dot3(a, b):
    return (float(a[0]) * float(b[0]) + float(a[1]) * float(b[1])) + float(a[2]) * float(b[2])


def norm3(a):
    return math.sqrt(dot3(a, a))


def cross3(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]], dtype=np.float64)


def jacobian_transpose_vector(jacobian, vector):
    result = jacobian[0] * vector[0]
    result = jacobian[1] * vector[1] + result
    return jacobian[2] * vector[2] + result


def ordered_matmul(left, right):
    """Scalar-column GEMV path used for the SDK's NxN times Nx3 arrays."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError('incompatible matrix shapes')
    if left.shape[1] == 0:
        return np.zeros((left.shape[0], right.shape[1]), dtype=np.float64)
    result = left[:, 0, None] * right[None, 0, :]
    for k in range(1, left.shape[1]):
        result = left[:, k, None] * right[None, k, :] + result
    return result
