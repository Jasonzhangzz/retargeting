# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
from typing import Any

import warp as wp


@wp.func
def axis_angle_to_quaternion_func(axis_angle_vec: wp.vec3) -> wp.vec4:
    """Convert an axis-angle vector to a quaternion (wxyz)."""
    angle = wp.length(axis_angle_vec)
    eps = 1e-6

    if angle < eps:
        # small angle approximation q ≈ [1, axis_angle/2], via the Taylor series sin(x/2) ≈ x/2 - x³/48 for small x
        angle_sq = angle * angle
        scale = 0.5 - angle_sq / 48.0
        return wp.vec4(
            1.0 - angle_sq / 8.0,  # cos(angle/2) ≈ 1 - x²/8
            axis_angle_vec[0] * scale,
            axis_angle_vec[1] * scale,
            axis_angle_vec[2] * scale,
        )
    # standard conversion
    axis = axis_angle_vec / angle
    half_angle = angle * 0.5
    sin_half = wp.sin(half_angle)
    cos_half = wp.cos(half_angle)
    return wp.vec4(cos_half, axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half)


@wp.kernel
def axis_angle_to_quaternion_kernel(axis_angle: wp.array(dtype=wp.vec3), quat_wxyz: wp.array(dtype=wp.vec4)):
    """Warp kernel to convert axis-angle to quaternion."""
    tid = wp.tid()
    quat_wxyz[tid] = axis_angle_to_quaternion_func(axis_angle[tid])


@wp.func
def euler_angle_to_matrix_func(angle: wp.float32, axis: wp.int32) -> wp.mat33:
    c, s = wp.cos(angle), wp.sin(angle)
    if axis == 0:
        return wp.mat33(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)
    elif axis == 1:
        return wp.mat33(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)
    else:
        return wp.mat33(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)


@wp.kernel
def euler_angles_to_matrix_kernel(
    euler_angles: wp.array(dtype=wp.vec3), axes: wp.vec4i, rot_mat: wp.array(dtype=wp.mat33)
):
    tid = wp.tid()
    euler_angles_elem = euler_angles[tid]

    if axes[0] == 0:  # static/extrinsic rotation
        rot_mat[tid] = wp.mul(
            wp.mul(
                euler_angle_to_matrix_func(euler_angles_elem[2], axes[3]),
                euler_angle_to_matrix_func(euler_angles_elem[1], axes[2]),
            ),
            euler_angle_to_matrix_func(euler_angles_elem[0], axes[1]),
        )
    else:  # rotating/intrinsic rotation
        rot_mat[tid] = wp.mul(
            wp.mul(
                euler_angle_to_matrix_func(euler_angles_elem[0], axes[1]),
                euler_angle_to_matrix_func(euler_angles_elem[1], axes[2]),
            ),
            euler_angle_to_matrix_func(euler_angles_elem[2], axes[3]),
        )


@wp.kernel
def matrix_to_euler_angles_kernel(
    rot_mat: wp.array(dtype=wp.mat33), axes: wp.vec4i, euler_angles: wp.array(dtype=wp.vec3)
):
    """Same algorithm and angle ranges as the NumPy `matrix_to_euler_angles`."""
    tid = wp.tid()
    m = rot_mat[tid]

    # a rotating sequence is the reversed static sequence with the angles reversed
    i, last = axes[1], axes[3]
    if axes[0] == 1:
        i, last = axes[3], axes[1]
    odd = axes[2] != (i + 1) % 3
    j, k = (i + 1) % 3, (i + 2) % 3
    if odd:
        j, k = (i + 2) % 3, (i + 1) % 3

    s, a, b = float(0.0), float(0.0), float(0.0)
    if i == last:  # proper Euler
        s = wp.sqrt(m[i, j] * m[i, j] + m[i, k] * m[i, k])
        a = wp.atan2(m[i, j], m[i, k])
        b = wp.atan2(s, m[i, i])
    else:  # Tait-Bryan
        s = wp.sqrt(m[i, i] * m[i, i] + m[j, i] * m[j, i])
        a = wp.atan2(m[k, j], m[k, k])
        b = wp.atan2(-m[k, i], s)
    # gimbal lock: only the sum of the outer angles is defined, so put it all in the first one
    if s < 1.0e-6:
        a = wp.atan2(-m[j, k], m[j, j])
    if odd:
        a, b = -a, -b
        if i == last:  # keep the middle angle of proper Euler axes in [0, pi]
            b = -b
            if s >= 1.0e-6:
                a = a + wp.pi
    if axes[0] == 1:
        if s < 1.0e-6:
            a = 0.0
    if a > wp.pi:
        a = a - 2.0 * wp.pi

    # the last angle from what remains, so the angles always reproduce the matrix
    rest = wp.mul(m, wp.transpose(wp.mul(euler_angle_to_matrix_func(b, axes[2]), euler_angle_to_matrix_func(a, i))))
    p, q = (last + 1) % 3, (last + 2) % 3
    c = wp.atan2(rest[q, p], rest[p, p])

    if axes[0] == 0:
        euler_angles[tid] = wp.vec3(a, b, c)
    else:
        euler_angles[tid] = wp.vec3(c, b, a)


@wp.func
def matrix_to_quaternion_func(rot_mat: wp.mat33) -> wp.vec4:
    m00, m01, m02 = rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2]
    m10, m11, m12 = rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2]
    m20, m21, m22 = rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2]

    t0 = 1.0 + m00 + m11 + m22
    t1 = 1.0 + m00 - m11 - m22
    t2 = 1.0 - m00 + m11 - m22
    t3 = 1.0 - m00 - m11 + m22

    q0 = wp.sqrt(t0) if t0 > 0.0 else 0.0
    q1 = wp.sqrt(t1) if t1 > 0.0 else 0.0
    q2 = wp.sqrt(t2) if t2 > 0.0 else 0.0
    q3 = wp.sqrt(t3) if t3 > 0.0 else 0.0

    c0 = wp.vec4(q0 * q0, m21 - m12, m02 - m20, m10 - m01)
    c1 = wp.vec4(m21 - m12, q1 * q1, m10 + m01, m02 + m20)
    c2 = wp.vec4(m02 - m20, m10 + m01, q2 * q2, m12 + m21)
    c3 = wp.vec4(m10 - m01, m20 + m02, m21 + m12, q3 * q3)

    qmax, qb = q0, c0
    if q1 > qmax:
        qmax, qb = q1, c1
    if q2 > qmax:
        qmax, qb = q2, c2
    if q3 > qmax:
        qmax, qb = q3, c3

    denom = 2.0 * (qmax if qmax > 0.1 else 0.1)
    q = qb / denom
    return -q if q[0] < 0.0 else q


@wp.kernel
def matrix_to_quaternion_kernel(rot_mat: wp.array(dtype=wp.mat33), quat_wxyz: wp.array(dtype=wp.vec4)):
    tid = wp.tid()
    quat_wxyz[tid] = matrix_to_quaternion_func(rot_mat[tid])  # wxyz format


@wp.func
def quaternion_to_matrix_func(quat_wxyz: wp.vec4) -> wp.mat33:
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    two_s = 2.0 / (w * w + x * x + y * y + z * z)
    return wp.mat33(
        1.0 - two_s * (y * y + z * z),
        two_s * (x * y - z * w),
        two_s * (x * z + y * w),
        two_s * (x * y + z * w),
        1.0 - two_s * (x * x + z * z),
        two_s * (y * z - x * w),
        two_s * (x * z - y * w),
        two_s * (y * z + x * w),
        1.0 - two_s * (x * x + y * y),
    )


@wp.kernel
def quaternion_to_matrix_kernel(quat_wxyz: wp.array(dtype=wp.vec4), rot_mat: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    rot_mat[tid] = quaternion_to_matrix_func(quat_wxyz[tid])


@wp.func
def quaternion_apply_func(quat_wxyz: wp.vec4, vector: wp.vec3):
    # apply quaternion rotation: v' = v + 2 * w * (qv × v) + 2 * qv × (qv × v)
    w = quat_wxyz[0]
    qv = wp.vec3(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3])
    uv = wp.cross(qv, vector)
    uuv = wp.cross(qv, uv)

    return vector + 2.0 * (w * uv + uuv)


@wp.kernel
def quaternion_apply_kernel(
    quat_wxyz: wp.array(dtype=wp.vec4),
    vectors: wp.array(dtype=wp.vec3),
    out_vectors: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    out_vectors[tid] = quaternion_apply_func(quat_wxyz[tid], vectors[tid])


@wp.func
def quaternion_invert_func(quat_wxyz: Any):
    """Conjugate a unit quaternion (wxyz), float32 or float64."""
    return wp.vector(quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3])


@wp.kernel
def quaternion_invert_kernel(
    quat_wxyz: wp.array(dtype=wp.vec4),
    quat_wxyz_inv: wp.array(dtype=wp.vec4),
):
    tid = wp.tid()
    quat_wxyz_inv[tid] = quaternion_invert_func(quat_wxyz[tid])


@wp.func
def quaternion_to_axis_angle_func(quat_wxyz: Any):
    """Convert a quaternion (wxyz), float32 or float64, to an axis-angle vector."""
    w = quat_wxyz[0]
    xyz = wp.vector(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3])
    norm = wp.length(xyz)

    half_angle = wp.atan2(norm, w)
    angle = type(w)(2.0) * half_angle
    eps = type(w)(1e-6)
    small_angle = wp.abs(angle) < eps

    if small_angle:
        # for small angles, sin(half_angle) ≈ half_angle - half_angle^3/6, so sin(half_angle)/angle ≈ 0.5 - angle^2/48
        sin_half_angle_over_angle = type(w)(0.5) - (angle * angle) / type(w)(48.0)
    else:
        sin_half_angle = wp.sin(half_angle)
        sin_half_angle_over_angle = sin_half_angle / angle

    return xyz / sin_half_angle_over_angle


@wp.kernel
def quaternion_to_axis_angle_kernel(
    quat_wxyz: wp.array(dtype=wp.vec4),
    axis_angle: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    axis_angle[tid] = quaternion_to_axis_angle_func(quat_wxyz[tid])


@wp.func
def standardize_quaternion_func(quat_wxyz: Any):
    """Standardize quaternion to have non-negative real part."""
    # assumes wxyz format
    if quat_wxyz[0] < type(quat_wxyz[0])(0.0):
        return -quat_wxyz
    return quat_wxyz


@wp.func
def quaternion_multiply_func(quat_wxyz1: Any, quat_wxyz2: Any):
    """Multiply two quaternions (wxyz), float32 or float64, and standardize the result."""
    w1, x1, y1, z1 = quat_wxyz1[0], quat_wxyz1[1], quat_wxyz1[2], quat_wxyz1[3]
    w2, x2, y2, z2 = quat_wxyz2[0], quat_wxyz2[1], quat_wxyz2[2], quat_wxyz2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return standardize_quaternion_func(wp.vector(w, x, y, z))


@wp.kernel
def quaternion_multiply_kernel(
    quat_wxyz1: wp.array(dtype=wp.vec4), quat_wxyz2: wp.array(dtype=wp.vec4), output: wp.array(dtype=wp.vec4)
):
    tid = wp.tid()
    output[tid] = quaternion_multiply_func(quat_wxyz1[tid], quat_wxyz2[tid])
