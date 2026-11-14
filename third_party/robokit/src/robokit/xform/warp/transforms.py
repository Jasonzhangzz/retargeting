import warp as wp


@wp.kernel
def transform_points_kernel(
    points: wp.array(dtype=wp.vec3),
    tf_mat: wp.array(dtype=wp.mat44),
    n_pts: wp.int32,
    out_pt: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    b_idx = tid / (n_pts)
    out_pt[tid] = wp.transform_point(tf_mat[b_idx], points[tid])


@wp.kernel
def rotate_points_kernel(
    points: wp.array(dtype=wp.vec3),
    rot_mat: wp.array(dtype=wp.mat33),
    n_pts: wp.int32,
    out_pt: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    b_idx = tid / (n_pts)
    out_pt[tid] = wp.mul(rot_mat[b_idx], points[tid])


@wp.func
def project_point_func(pt: wp.vec3, intrinsic: wp.mat33) -> wp.vec2:
    """Pinhole projection of a camera-frame point to pixels; depth is clamped at 1e-3 m."""
    z = wp.max(pt[2], 1e-3)  # pyright: ignore[reportCallIssue, reportArgumentType]
    h = intrinsic * wp.vec3(pt[0] / z, pt[1] / z, 1.0)  # pyright: ignore[reportArgumentType]
    return wp.vec2(h[0], h[1])


@wp.func
def tf_mat_to_rot_mat_func(tf_mat: wp.mat44) -> wp.mat33:
    """Extract the 3x3 rotation block from a 4x4 transform."""
    # fmt: off
    return wp.mat33(
        tf_mat[0, 0], tf_mat[0, 1], tf_mat[0, 2],
        tf_mat[1, 0], tf_mat[1, 1], tf_mat[1, 2],
        tf_mat[2, 0], tf_mat[2, 1], tf_mat[2, 2],
    )
    # fmt: on


@wp.func
def inverse_tf_mat_func(tf_mat: wp.mat44) -> wp.mat44:
    """Inverse of a rigid SE(3) transform, `[R^T | -R^T t]`."""
    # R^T (rotation transpose)
    r00, r01, r02 = tf_mat[0, 0], tf_mat[1, 0], tf_mat[2, 0]
    r10, r11, r12 = tf_mat[0, 1], tf_mat[1, 1], tf_mat[2, 1]
    r20, r21, r22 = tf_mat[0, 2], tf_mat[1, 2], tf_mat[2, 2]
    # -R^T @ t
    tx, ty, tz = tf_mat[0, 3], tf_mat[1, 3], tf_mat[2, 3]
    nx = -(r00 * tx + r01 * ty + r02 * tz)  # pyright: ignore[reportOperatorIssue]
    ny = -(r10 * tx + r11 * ty + r12 * tz)  # pyright: ignore[reportOperatorIssue]
    nz = -(r20 * tx + r21 * ty + r22 * tz)  # pyright: ignore[reportOperatorIssue]
    # fmt: off
    return wp.mat44(
        r00, r01, r02, nx,
        r10, r11, r12, ny,
        r20, r21, r22, nz,
        0.0, 0.0, 0.0, 1.0,  # type: ignore
    )
    # fmt: on


@wp.kernel
def inverse_tf_mat_kernel(
    tf_mat: wp.array(dtype=wp.mat44),
    tf_mat_inv: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()
    tf_mat_inv[tid] = inverse_tf_mat_func(tf_mat[tid])
