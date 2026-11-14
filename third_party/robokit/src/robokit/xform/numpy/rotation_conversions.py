import numpy as np
from jaxtyping import Float


def matrix_to_quaternion(rot_mat: Float[np.ndarray, "... 3 3"]) -> Float[np.ndarray, "... 4"]:
    """
    Convert rotation matrices to quaternions using Shepperds's method.

    Args:
        rot_mat: rotation matrices, in shape of (..., 3, 3).

    Returns:
        quaternions with real part first (wxyz), in shape of (..., 4).

    Example:
        >>> q = matrix_to_quaternion(np.eye(3))
        >>> np.allclose(q, np.array([1., 0., 0., 0.]))
        True
        >>> q = matrix_to_quaternion(np.diag([1, -1, -1]))
        >>> np.allclose(q, np.array([0., 1., 0., 0.]))
        True
        >>> rot_mat = np.array([[-0.2533, -0.6075,  0.7529],
        ...                     [ 0.8445, -0.5185, -0.1343],
        ...                     [ 0.4720,  0.6017,  0.6443]])
        >>> q = matrix_to_quaternion(rot_mat)
        >>> np.allclose(q, np.array([0.4671, 0.3940, 0.1503, 0.7772]), atol=1e-4)
        True
    """
    m00, m01, m02 = rot_mat[..., 0, 0], rot_mat[..., 0, 1], rot_mat[..., 0, 2]
    m10, m11, m12 = rot_mat[..., 1, 0], rot_mat[..., 1, 1], rot_mat[..., 1, 2]
    m20, m21, m22 = rot_mat[..., 2, 0], rot_mat[..., 2, 1], rot_mat[..., 2, 2]

    q_abs = np.stack(
        (1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22), axis=-1
    )
    q_abs = np.sqrt(np.maximum(q_abs, 0.0))

    quat_by_rijk = np.stack(
        (
            np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
            np.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ),
        axis=-2,
    )
    flr = 0.1
    quat_candidates = quat_by_rijk / (2.0 * np.maximum(q_abs[..., None], flr))

    idx = np.argmax(q_abs, axis=-1)[..., None, None]
    quat = np.take_along_axis(quat_candidates, idx, axis=-2).squeeze(-2)

    return standardize_quaternion(quat)


def normalize_quaternion(q_wxyz: Float[np.ndarray, "... 4"]) -> Float[np.ndarray, "... 4"]:
    """
    Normalize quaternions to unit length.

    Args:
        q_wxyz: a numpy array of shape (..., 4), where the last dimension represents the quaternion components [w, x, y, z].

    Returns:
        A numpy array of shape (..., 4) with unit-norm quaternions.

    Example:
        >>> q = np.array([2., 0., 0., 0.])
        >>> np.allclose(normalize_quaternion(q), np.array([1., 0., 0., 0.]))
        True
        >>> q = np.array([1., 1., 1., 1.])
        >>> np.allclose(normalize_quaternion(q), np.array([0.5, 0.5, 0.5, 0.5]))
        True
    """
    return q_wxyz / np.linalg.norm(q_wxyz, axis=-1, keepdims=True)


def standardize_quaternion(q_wxyz: Float[np.ndarray, "... 4"]) -> Float[np.ndarray, "... 4"]:
    """
    Convert a unit quaternion to a standard form: one in which the real part is non-negative.

    Args:
        q_wxyz: a numpy array of shape (..., 4), where the last dimension represents the quaternion components [w, x, y, z].

    Returns:
        A numpy array with shape of (..., 4) representing the quaternion components [w, x, y, z].

    Example:
        >>> q_wxyz = np.array([[0.0, -0.7071, 0.0, 0.7071], [-0.7071, 0.0, 0.7071, 0.0]])
        >>> expected = np.array([[0.0, -0.7071, 0.0, 0.7071], [0.7071, -0.0, -0.7071, -0.0]])
        >>> np.allclose(standardize_quaternion(q_wxyz), expected)
        True
    """
    return np.where(q_wxyz[..., 0:1] < 0, -q_wxyz, q_wxyz)


def matrix_to_axis_angle(rot_mat: Float[np.ndarray, "... 3 3"]) -> Float[np.ndarray, "... 3"]:
    """
    Convert rotation matrices to axis-angle representation.

    Args:
        rot_mat: Rotation matrices with shape (..., 3, 3).

    Returns:
        Axis-angle representation with shape (..., 3) where the magnitude represents the rotation angle and the direction represents the rotation axis.

    Example:
        >>> rot_mat = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        >>> expected = np.array([np.pi/2, 0., 0.])
        >>> np.allclose(matrix_to_axis_angle(rot_mat), expected, atol=1e-4)
        True
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(rot_mat))


def matrix_to_rotation_angle(rot_mat: Float[np.ndarray, "... 3 3"]) -> Float[np.ndarray, "..."]:
    """
    Compute the rotation angle of rotation matrices; pass `R_a.T @ R_b` for the angle between two rotations.

    Args:
        rot_mat: Rotation matrices with shape (..., 3, 3).

    Returns:
        Rotation angles in radians, in [0, pi], with shape (...).

    Example:
        >>> rot_mat = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        >>> bool(np.isclose(matrix_to_rotation_angle(rot_mat), np.pi / 2))
        True
    """
    return np.linalg.norm(matrix_to_axis_angle(rot_mat), axis=-1)


def axis_angle_to_quaternion(axis_angle: Float[np.ndarray, "... 3"]) -> Float[np.ndarray, "... 4"]:
    """
    Convert axis-angle representation to quaternion.

    Args:
        axis_angle: Axis-angle representation [..., 3] where the magnitude is the rotation angle

    Returns:
        Quaternion representation [..., 4] in wxyz format (w, x, y, z)

    Example:
        >>> axis_angle = np.array([1.5708, 0.0, 0.0])
        >>> expected = np.array([0.7071, 0.7071, 0.0000, 0.0000])
        >>> np.allclose(axis_angle_to_quaternion(axis_angle), expected, atol=1e-4)
        True
    """
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = np.abs(angles) < eps

    sin_half_angles_over_angles = np.empty_like(angles)

    sin_half_angles_over_angles[~small_angles] = np.sin(half_angles[~small_angles]) / angles[~small_angles]

    # for small angles, Taylor series: sin(x/2) ≈ x/2 - (x/2)³/6, so sin(x/2)/x ≈ 1/2 - x²/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48

    w = np.cos(half_angles)
    xyz = axis_angle * sin_half_angles_over_angles

    return np.concatenate([w, xyz], axis=-1)


def axis_angle_to_matrix(axis_angle: Float[np.ndarray, "... 3"]) -> Float[np.ndarray, "... 3 3"]:
    """
    Convert an axis-angle representation to a 3x3 rotation matrix.

    Args:
        axis_angle: A numpy array of shape (..., 3) representing a vector in axis-angle format where the direction indicates the axis of rotation and the magnitude represents the rotation angle in radians.

    Returns:
        A numpy array of shape (..., 3, 3) representing the rotation matrices corresponding to each axis-angle input.

    Example:
        >>> axis_angle = np.array([np.pi/2, 0., 0.])
        >>> expected = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        >>> np.allclose(axis_angle_to_matrix(axis_angle), expected, atol=1e-4)
        True
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def euler_angles_to_matrix(
    euler_angles: Float[np.ndarray, "... 3"], axes: str = "sxyz"
) -> Float[np.ndarray, "... 3 3"]:
    """
    Convert Euler angles to rotation matrices, matching the Torch `euler_angles_to_matrix`.

    Args:
        euler_angles: angles in radians, in shape of (..., 3).
        axes: "s" (static/extrinsic) or "r" (rotating/intrinsic) plus an axis sequence, e.g. "sxyz" or "rxyz"; three-character forms get a leading "s".

    Returns:
        rotation matrices, in shape of (..., 3, 3).

    Example:
        >>> m = euler_angles_to_matrix(np.array([1.0, 0.5, 2.0]), axes="rxyz")
        >>> expected = [[-0.3652, -0.7980, 0.4794], [0.3234, -0.5917, -0.7385], [0.8729, -0.1146, 0.4742]]
        >>> np.allclose(m, expected, atol=1e-4)
        True
    """
    axes = axes.lower() if len(axes) == 4 else f"s{axes.lower()}"
    if axes[0] not in "sr" or axes[1] == axes[2] or axes[2] == axes[3]:
        raise ValueError(f"Invalid axes: {axes}")
    c, s = np.cos(euler_angles), np.sin(euler_angles)
    mats = []
    for i, axis in enumerate(axes[1:]):
        k = "xyz".index(axis)
        j, l = (k + 1) % 3, (k + 2) % 3
        m = np.zeros(c.shape[:-1] + (3, 3), dtype=c.dtype)
        m[..., k, k] = 1.0
        m[..., j, j] = m[..., l, l] = c[..., i]
        m[..., l, j], m[..., j, l] = s[..., i], -s[..., i]
        mats.append(m)
    return mats[2] @ mats[1] @ mats[0] if axes[0] == "s" else mats[0] @ mats[1] @ mats[2]


def matrix_to_euler_angles(rot_mat: Float[np.ndarray, "... 3 3"], axes: str = "sxyz") -> Float[np.ndarray, "... 3"]:
    """
    Convert rotation matrices to Euler angles, the inverse of `euler_angles_to_matrix`.

    Args:
        rot_mat: rotation matrices, in shape of (..., 3, 3).
        axes: "s" (static/extrinsic) or "r" (rotating/intrinsic) plus an axis sequence, e.g. "sxyz" or "rxyz"; three-character forms get a leading "s".

    Returns:
        angles in radians, in shape of (..., 3). As in scipy, the middle angle lies in [-pi/2, pi/2] for Tait-Bryan axes (e.g. "sxyz") and in [0, pi] for proper Euler axes (e.g. "szxz"), the others in [-pi, pi], and the third angle is zero at gimbal lock.

    Example:
        >>> angles = np.array([1.0, 0.5, 2.0])
        >>> np.allclose(matrix_to_euler_angles(euler_angles_to_matrix(angles, "rxyz"), "rxyz"), angles)
        True
    """
    axes = axes.lower() if len(axes) == 4 else f"s{axes.lower()}"
    if axes[0] not in "sr" or axes[1] == axes[2] or axes[2] == axes[3]:
        raise ValueError(f"Invalid axes: {axes}")
    # a rotating sequence is the reversed static sequence with the angles reversed
    seq = axes[1:] if axes[0] == "s" else axes[:0:-1]
    i, last = "xyz".index(seq[0]), "xyz".index(seq[2])
    odd = seq[1] != "xyz"[(i + 1) % 3]
    j, k = ((i + 2) % 3, (i + 1) % 3) if odd else ((i + 1) % 3, (i + 2) % 3)
    m = rot_mat
    if i == last:  # proper Euler
        s = np.hypot(m[..., i, j], m[..., i, k])
        a, b = np.arctan2(m[..., i, j], m[..., i, k]), np.arctan2(s, m[..., i, i])
    else:  # Tait-Bryan
        s = np.hypot(m[..., i, i], m[..., j, i])
        a, b = np.arctan2(m[..., k, j], m[..., k, k]), np.arctan2(-m[..., k, i], s)
    # gimbal lock: only the sum of the outer angles is defined, so put it all in the first one
    lock = s < 1e-6
    a = np.where(lock, np.arctan2(-m[..., j, k], m[..., j, j]), a)
    if odd:
        a, b = -a, -b
        if i == last:  # keep the middle angle of proper Euler axes in [0, pi]
            a, b = np.where(lock, a, a + np.pi), -b
    if axes[0] == "r":
        a = np.where(lock, 0.0, a)
    a = np.where(a > np.pi, a - 2 * np.pi, a)
    # the last angle from what remains, so the angles always reproduce the matrix
    rest = m @ np.swapaxes(euler_angles_to_matrix(np.stack([a, b, np.zeros_like(a)], axis=-1), f"s{seq}"), -1, -2)
    p, q = (last + 1) % 3, (last + 2) % 3
    c = np.arctan2(rest[..., q, p], rest[..., p, p])
    return np.stack([a, b, c] if axes[0] == "s" else [c, b, a], axis=-1)


def random_quaternions(n: int) -> Float[np.ndarray, "... 4"]:
    """
    Generate random quaternions representing rotations, i.e. versors with non-negative real part.

    Args:
        n: Number of quaternions in a batch to return.

    Returns:
        Quaternions as a numpy array of shape (n, 4).

    Example:
        >>> q = random_quaternions(5)
        >>> bool(np.allclose(np.linalg.norm(q, axis=1), 1.0) and (q[:, 0] >= 0).all())
        True
    """
    o = np.random.randn(n, 4)
    s = np.sum(o * o, axis=1)
    o = o / np.copysign(np.sqrt(s), o[:, 0])[:, np.newaxis]
    return o


def quaternion_to_matrix(q_wxyz: Float[np.ndarray, "... 4"]) -> Float[np.ndarray, "... 3 3"]:
    """
    Convert quaternions to rotation matrices.

    Args:
        q_wxyz: Quaternions [..., 4] in wxyz format (w, x, y, z)

    Returns:
        Rotation matrices [..., 3, 3]

    Example:
        >>> q = np.array([1., 0., 0., 0.])
        >>> m = quaternion_to_matrix(q)
        >>> np.allclose(m, np.eye(3))
        True
        >>> q = np.array([0., 1., 0., 0.])
        >>> m = quaternion_to_matrix(q)
        >>> np.allclose(m, np.diag([1, -1, -1]))
        True
    """
    w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rotation_matrix = np.stack(
        [
            np.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            np.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    )

    return rotation_matrix


def quaternion_raw_multiply(
    q_wxyz_1: Float[np.ndarray, "... 4"], q_wxyz_2: Float[np.ndarray, "... 4"]
) -> Float[np.ndarray, "... 4"]:
    """
    Multiply two quaternions using raw multiplication.

    Args:
        q_wxyz_1: Quaternions [..., 4] in wxyz format (w, x, y, z)
        q_wxyz_2: Quaternions [..., 4] in wxyz format (w, x, y, z)

    Returns:
        The product of q_wxyz_1 and q_wxyz_2 [..., 4]

    Example:
        >>> q_z90 = np.array([0.7071068, 0.0, 0.0, 0.7071068])
        >>> np.round(quaternion_raw_multiply(q_z90, q_z90), 4)  # 90° + 90° about z = 180° about z
        array([0., 0., 0., 1.])
    """
    aw, ax, ay, az = q_wxyz_1[..., 0], q_wxyz_1[..., 1], q_wxyz_1[..., 2], q_wxyz_1[..., 3]
    bw, bx, by, bz = q_wxyz_2[..., 0], q_wxyz_2[..., 1], q_wxyz_2[..., 2], q_wxyz_2[..., 3]

    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw

    return np.stack([ow, ox, oy, oz], axis=-1)


def quaternion_multiply(
    q_wxyz_1: Float[np.ndarray, "... 4"], q_wxyz_2: Float[np.ndarray, "... 4"]
) -> Float[np.ndarray, "... 4"]:
    """
    Multiply two quaternions (Hamilton product) and standardize the result.

    Args:
        q_wxyz_1: a numpy array of shape (..., 4), where the last dimension represents the quaternion components [w, x, y, z].
        q_wxyz_2: a numpy array of shape (..., 4), where the last dimension represents the quaternion components [w, x, y, z].

    Returns:
        A numpy array with shape of (..., 4) representing the quaternion components [w, x, y, z].

    Example:
        >>> a = np.array([0.9238795, 0.0, 0.0, 0.3826834])
        >>> b = np.array([0.9238795, 0.0, 0.0, 0.3826834])
        >>> expected = np.array([0.7071068, 0.0, 0.0, 0.7071068])
        >>> np.allclose(quaternion_multiply(a, b), expected, atol=1e-6)
        True
    """
    return standardize_quaternion(quaternion_raw_multiply(q_wxyz_1, q_wxyz_2))


def quaternion_invert(q_wxyz: Float[np.ndarray, "... 4"]) -> Float[np.ndarray, "... 4"]:
    """
    Invert quaternions by negating the imaginary parts.

    Args:
        q_wxyz: Quaternions [..., 4] in wxyz format (w, x, y, z)

    Returns:
        Inverted quaternions [..., 4]

    Example:
        >>> q = np.array([1., 1., 1., 1.])
        >>> q_inv = quaternion_invert(q)
        >>> np.allclose(q_inv, np.array([1., -1., -1., -1.]))
        True
    """
    w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
    return np.stack([w, -x, -y, -z], axis=-1)


def quaternion_apply(
    q_wxyz: Float[np.ndarray, "... 4"], points: Float[np.ndarray, "... 3"]
) -> Float[np.ndarray, "... 3"]:
    """
    Apply quaternion rotations to 3D points using raw quaternion multiplication.

    Args:
        q_wxyz: Quaternions [..., 4] in wxyz format (w, x, y, z); normalized before use
        points: 3D points [..., 3] to rotate

    Returns:
        Rotated points [..., 3]

    Example:
        >>> quaternion = np.array([0.7071, 0.0, 0.0, 0.7071])
        >>> point = np.array([1.0, 0.0, 0.0])
        >>> expected = np.array([0., 1., 0.])
        >>> np.allclose(quaternion_apply(quaternion, point), expected, atol=1e-4)
        True
    """
    point_quaternions = np.concatenate([np.zeros(points.shape[:-1] + (1,)), points], axis=-1)

    q_wxyz = normalize_quaternion(q_wxyz)
    q_inv = quaternion_invert(q_wxyz)
    first = quaternion_raw_multiply(q_wxyz, point_quaternions)
    rotated_quaternions = quaternion_raw_multiply(first, q_inv)

    return rotated_quaternions[..., 1:]


def quaternion_to_axis_angle(q_wxyz: Float[np.ndarray, "... 4"]) -> Float[np.ndarray, "... 3"]:
    """
    Convert quaternions to axis-angle representation.

    Args:
        q_wxyz: Quaternions [..., 4] in wxyz format (w, x, y, z)

    Returns:
        Axis-angle representation [..., 3]

    Example:
        >>> q_wxyz = np.array([0.7071, 0.7071, 0.0, 0.0])
        >>> expected = np.array([1.5708, 0.0000, 0.0000])
        >>> np.allclose(quaternion_to_axis_angle(q_wxyz), expected, atol=1e-4)
        True
    """
    norms = np.linalg.norm(q_wxyz[..., 1:], axis=-1, keepdims=True)
    half_angles = np.arctan2(norms, q_wxyz[..., :1])
    angles = 2 * half_angles

    eps = 1e-6
    small_angles = np.abs(angles) < eps

    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = np.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6, so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48

    return q_wxyz[..., 1:] / sin_half_angles_over_angles


def rotation_between_vectors(
    src: Float[np.ndarray, "... 3"], dst: Float[np.ndarray, "... 3"]
) -> Float[np.ndarray, "... 3 3"]:
    """
    Compute the smallest rotation that turns the direction of `src` into the direction of `dst`.

    Args:
        src: Source vectors [..., 3]; only their direction is used.
        dst: Destination vectors [..., 3]; only their direction is used.

    Returns:
        Rotation matrices [..., 3, 3] with `rot_mat @ src` parallel to `dst`; a half turn for opposite vectors.

    Example:
        >>> rot_mat = rotation_between_vectors(np.array([0., 0., 1.]), np.array([1., 0., 0.]))
        >>> np.allclose(rot_mat @ np.array([0., 0., 1.]), [1., 0., 0.])
        True
    """
    u = src / np.linalg.norm(src, axis=-1, keepdims=True)
    v = dst / np.linalg.norm(dst, axis=-1, keepdims=True)
    w, xyz = 1.0 + np.sum(u * v, axis=-1, keepdims=True), np.cross(u, v)
    # (nearly) opposite vectors: turn half way around any axis perpendicular to `src`
    helper = np.where(np.abs(u[..., :1]) < 0.9, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    opposite = w + np.linalg.norm(xyz, axis=-1, keepdims=True) < 1e-6
    xyz = np.where(opposite, np.cross(u, helper), xyz)
    return quaternion_to_matrix(np.concatenate([np.where(opposite, 0.0, w), xyz], axis=-1))


def quaternion_slerp(
    q_wxyz_1: Float[np.ndarray, "... 4"], q_wxyz_2: Float[np.ndarray, "... 4"], t: Float[np.ndarray, "..."]
) -> Float[np.ndarray, "... 4"]:
    """
    Spherically interpolate between unit quaternions along the shorter arc.

    Args:
        q_wxyz_1: Unit quaternions [..., 4] in wxyz format, returned at `t = 0`.
        q_wxyz_2: Unit quaternions [..., 4] in wxyz format, reached at `t = 1` (up to sign).
        t: Interpolation fractions [...], broadcast against the quaternion batch dimensions.

    Returns:
        Interpolated quaternions [..., 4] in wxyz format with non-negative real part.

    Example:
        >>> q_z90 = np.array([0.7071068, 0.0, 0.0, 0.7071068])
        >>> np.allclose(quaternion_slerp(np.array([1., 0., 0., 0.]), q_z90, 0.5), [0.9238795, 0., 0., 0.3826834])
        True
    """
    delta = quaternion_to_axis_angle(quaternion_multiply(quaternion_invert(q_wxyz_1), q_wxyz_2))
    return quaternion_multiply(q_wxyz_1, axis_angle_to_quaternion(np.asarray(t)[..., None] * delta))


__all__ = [
    "matrix_to_quaternion",
    "matrix_to_axis_angle",
    "matrix_to_rotation_angle",
    "matrix_to_euler_angles",
    "euler_angles_to_matrix",
    "rotation_between_vectors",
    "quaternion_slerp",
    "normalize_quaternion",
    "standardize_quaternion",
    "axis_angle_to_quaternion",
    "axis_angle_to_matrix",
    "quaternion_to_matrix",
    "quaternion_apply",
    "quaternion_raw_multiply",
    "quaternion_multiply",
    "quaternion_invert",
    "quaternion_to_axis_angle",
    "random_quaternions",
]
