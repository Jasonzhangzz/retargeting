# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
import warp as wp

from robokit.lie.so3_kernels import hat_func, so3_jac_left_inv_func
from robokit.utils.warp_utils import wp_mat66, wp_vec6, wp_vec7
from robokit.xform.warp.rotation_conversions import (
    matrix_to_quaternion_func,
    quaternion_apply_func,
    quaternion_multiply_func,
    quaternion_to_axis_angle_func,
    quaternion_to_matrix_func,
)


@wp.func
def se3_from_matrix_func(mat: wp.mat44) -> wp_vec7:
    xyz = wp.vec3(mat[0, 3], mat[1, 3], mat[2, 3])
    rotation = wp.mat33(
        mat[0, 0], mat[0, 1], mat[0, 2], mat[1, 0], mat[1, 1], mat[1, 2], mat[2, 0], mat[2, 1], mat[2, 2]
    )
    wxyz = matrix_to_quaternion_func(rotation)
    return wp_vec7(xyz[0], xyz[1], xyz[2], wxyz[0], wxyz[1], wxyz[2], wxyz[3])


@wp.kernel
def se3_from_matrix_kernel(matrix: wp.array(dtype=wp.mat44), xyz_wxyz: wp.array(dtype=wp_vec7)):
    tid = wp.tid()
    xyz_wxyz[tid] = se3_from_matrix_func(matrix[tid])


@wp.kernel
def se3_to_matrix_kernel(xyz_wxyz: wp.array(dtype=wp_vec7), matrix: wp.array(dtype=wp.mat44)):
    """Fused kernel to convert SE(3) pose to 4x4 transformation matrix."""
    tid = wp.tid()
    trans = wp.vec3(xyz_wxyz[tid][0], xyz_wxyz[tid][1], xyz_wxyz[tid][2])
    quat_wxyz = wp.vec4(xyz_wxyz[tid][3], xyz_wxyz[tid][4], xyz_wxyz[tid][5], xyz_wxyz[tid][6])
    rot = quaternion_to_matrix_func(quat_wxyz)
    # fmt: off
    matrix[tid] = wp.mat44(
        rot[0, 0], rot[0, 1], rot[0, 2], trans[0],
        rot[1, 0], rot[1, 1], rot[1, 2], trans[1],
        rot[2, 0], rot[2, 1], rot[2, 2], trans[2],
        0.0, 0.0, 0.0, 1.0,
    )
    # fmt: on


@wp.func
def se3_apply_func(xyz_wxyz: wp_vec7, point: wp.vec3) -> wp.vec3:
    translation = wp.vec3(xyz_wxyz[0], xyz_wxyz[1], xyz_wxyz[2])
    quat_wxyz = wp.vec4(xyz_wxyz[3], xyz_wxyz[4], xyz_wxyz[5], xyz_wxyz[6])
    rotated = quaternion_apply_func(quat_wxyz, point)
    return rotated + translation


@wp.kernel
def se3_apply_kernel(
    xyz_wxyz: wp.array(dtype=wp_vec7),
    points: wp.array(dtype=wp.vec3),
    result: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    result[tid] = se3_apply_func(xyz_wxyz[tid], points[tid])


@wp.func
def se3_inverse_func(xyz_wxyz: wp_vec7) -> wp_vec7:
    xyz = wp.vec3(xyz_wxyz[0], xyz_wxyz[1], xyz_wxyz[2])
    q_wxyz = wp.vec4(xyz_wxyz[3], xyz_wxyz[4], xyz_wxyz[5], xyz_wxyz[6])
    q_inv_wxyz = wp.vec4(q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3])

    # invert translation: t_inv = -q_inv * t * q
    t_inv = -quaternion_apply_func(q_inv_wxyz, xyz)

    return wp_vec7(t_inv[0], t_inv[1], t_inv[2], q_inv_wxyz[0], q_inv_wxyz[1], q_inv_wxyz[2], q_inv_wxyz[3])


@wp.kernel
def se3_inverse_kernel(xyz_wxyz: wp.array(dtype=wp_vec7), result: wp.array(dtype=wp_vec7)):
    tid = wp.tid()
    result[tid] = se3_inverse_func(xyz_wxyz[tid])


@wp.func
def se3_compose_func(xyz_wxyz1: wp_vec7, xyz_wxyz2: wp_vec7) -> wp_vec7:
    xyz1 = wp.vec3(xyz_wxyz1[0], xyz_wxyz1[1], xyz_wxyz1[2])
    quat_wxyz1 = wp.vec4(xyz_wxyz1[3], xyz_wxyz1[4], xyz_wxyz1[5], xyz_wxyz1[6])
    xyz2 = wp.vec3(xyz_wxyz2[0], xyz_wxyz2[1], xyz_wxyz2[2])
    quat_wxyz2 = wp.vec4(xyz_wxyz2[3], xyz_wxyz2[4], xyz_wxyz2[5], xyz_wxyz2[6])

    # multiply quaternions to get the combined rotation
    quat_wxyz_result = quaternion_multiply_func(quat_wxyz1, quat_wxyz2)

    # apply first rotation to second translation, then add first translation: t_result = t1 + R1 * t2
    rotated_t2 = quaternion_apply_func(quat_wxyz1, xyz2)
    xyz_result = xyz1 + rotated_t2

    return wp_vec7(
        xyz_result[0],
        xyz_result[1],
        xyz_result[2],
        quat_wxyz_result[0],
        quat_wxyz_result[1],
        quat_wxyz_result[2],
        quat_wxyz_result[3],
    )


se3_multiply_func = se3_compose_func


@wp.kernel
def se3_compose_kernel(
    xyz_wxyz1: wp.array(dtype=wp_vec7),
    xyz_wxyz2: wp.array(dtype=wp_vec7),
    result: wp.array(dtype=wp_vec7),
):
    tid = wp.tid()
    result[tid] = se3_compose_func(xyz_wxyz1[tid], xyz_wxyz2[tid])


@wp.func
def se3_exp_map_func(log_transform: wp_vec6, eps: wp.float32) -> wp_vec7:
    log_translation = wp.vec3(log_transform[0], log_transform[1], log_transform[2])
    log_rotation = wp.vec3(log_transform[3], log_transform[4], log_transform[5])

    # regularized angle
    angle_sq = wp.dot(log_rotation, log_rotation)
    angle_sq_reg = angle_sq + eps * eps
    angle_reg = wp.sqrt(angle_sq_reg)

    # quaternion computation
    half_angle = angle_reg * 0.5
    sin_half_angle = wp.sin(half_angle)
    cos_half_angle = wp.cos(half_angle)

    # k = sin(theta/2) / theta -> limit 0.5 as theta->0
    k = sin_half_angle / angle_reg

    quat_wxyz = wp.vec4(cos_half_angle, k * log_rotation[0], k * log_rotation[1], k * log_rotation[2])
    quat_wxyz = wp.normalize(quat_wxyz)

    # V matrix coefficients
    sin_angle = wp.sin(angle_reg)
    cos_angle = wp.cos(angle_reg)

    # A = sin(theta)/theta
    A = sin_angle / angle_reg
    # coeff1 = (1-cos(theta))/theta^2
    coeff1 = (1.0 - cos_angle) / angle_sq_reg
    # coeff2 = (theta-sin(theta))/theta^3 = (1-A)/theta^2
    coeff2 = (1.0 - A) / angle_sq_reg

    K = hat_func(log_rotation)
    K_squared = K * K

    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    V = identity + coeff1 * K + coeff2 * K_squared

    translation = V * log_translation

    return wp_vec7(
        translation[0], translation[1], translation[2], quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    )


@wp.kernel
def se3_exp_map_kernel(
    log_transform: wp.array(dtype=wp_vec6),
    eps: wp.float32,
    xyz_wxyz_output: wp.array(dtype=wp_vec7),
):
    tid = wp.tid()
    xyz_wxyz_output[tid] = se3_exp_map_func(log_transform[tid], eps)


@wp.func
def se3_exp_map_to_matrix_func(log_transform: wp_vec6, eps: wp.float32) -> wp.mat44:
    """Fused SE(3) exp map to matrix; eps^2-padded denominators keep gradients defined as theta -> 0."""
    rho = wp.vec3(log_transform[0], log_transform[1], log_transform[2])
    omega = wp.vec3(log_transform[3], log_transform[4], log_transform[5])
    theta_sq = wp.dot(omega, omega)

    # regularize denominators to avoid division by zero
    theta_sq_reg = theta_sq + eps * eps
    theta_reg = wp.sqrt(theta_sq_reg)

    K = hat_func(omega)
    K2 = K * K

    sin_theta = wp.sin(theta_reg)
    cos_theta = wp.cos(theta_reg)

    # A = sin(theta)/theta -> 1 as theta->0
    A = sin_theta / theta_reg
    # B = (1-cos(theta))/theta^2 -> 0.5 as theta->0
    B = (1.0 - cos_theta) / theta_sq_reg
    # C = (1-A)/theta^2 -> 1/6 as theta->0
    C = (1.0 - A) / theta_sq_reg

    eye = wp.identity(n=3, dtype=wp.float32)
    R = eye + A * K + B * K2
    V = eye + B * K + C * K2

    t = V * rho

    # fmt: off
    return wp.mat44(
        R[0, 0], R[0, 1], R[0, 2], t[0],
        R[1, 0], R[1, 1], R[1, 2], t[1],
        R[2, 0], R[2, 1], R[2, 2], t[2],
        0.0, 0.0, 0.0, 1.0,
    )
    # fmt: on


@wp.kernel
def se3_exp_map_to_matrix_kernel(
    log_transform: wp.array(dtype=wp_vec6),
    eps: wp.float32,
    matrix_output: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()
    matrix_output[tid] = se3_exp_map_to_matrix_func(log_transform[tid], eps)


@wp.func
def se3_V_matrix_inverse_func(log_rotation: wp.vec3, eps: wp.float32) -> wp.mat33:
    """Inverse of the V matrix for SE(3) log: V^-1 = I - 0.5*[log_rotation]_x + c2*[log_rotation]_x^2."""
    angle = wp.length(log_rotation)

    # identity matrix
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    if angle < eps:
        # small angle approximation: V^-1 ~ I - 0.5 * [log_rotation]_x + (1/12) * [log_rotation]_x^2
        log_rotation_hat = wp.mat33(
            0.0,
            -log_rotation[2],
            log_rotation[1],
            log_rotation[2],
            0.0,
            -log_rotation[0],
            -log_rotation[1],
            log_rotation[0],
            0.0,
        )
        log_rotation_hat_sq = wp.mul(log_rotation_hat, log_rotation_hat)
        return identity - 0.5 * log_rotation_hat + (1.0 / 12.0) * log_rotation_hat_sq
    else:
        # full formula with numerical stability
        sin_angle = wp.sin(angle)
        cos_angle = wp.cos(angle)

        # skew symmetric matrix of log_rotation
        log_rotation_hat = wp.mat33(
            0.0,
            -log_rotation[2],
            log_rotation[1],
            log_rotation[2],
            0.0,
            -log_rotation[0],
            -log_rotation[1],
            log_rotation[0],
            0.0,
        )

        # log_rotation_hat^2
        log_rotation_hat_sq = wp.mul(log_rotation_hat, log_rotation_hat)

        # c2 coefficient (numerically stable): (1/angle^2) * (1 - angle*sin(angle)/(2*(1-cos(angle))))
        one_minus_cos = 1.0 - cos_angle
        if wp.abs(one_minus_cos) < eps:
            # near zero rotation: use series expansion
            c2 = 1.0 / 12.0
        else:
            c2 = (1.0 / (angle * angle)) * (1.0 - (angle * sin_angle) / (2.0 * one_minus_cos))

        return identity - 0.5 * log_rotation_hat + c2 * log_rotation_hat_sq


@wp.func
def se3_log_map_func(xyz_wxyz: wp_vec7, eps: wp.float32) -> wp_vec6:
    """SE(3) logarithm map function."""
    # extract translation and quaternion
    t = wp.vec3(xyz_wxyz[0], xyz_wxyz[1], xyz_wxyz[2])
    quat_wxyz = wp.vec4(xyz_wxyz[3], xyz_wxyz[4], xyz_wxyz[5], xyz_wxyz[6])

    # convert quaternion to axis-angle (log rotation)
    log_rotation = quaternion_to_axis_angle_func(quat_wxyz)

    # compute V^-1
    V_inv = se3_V_matrix_inverse_func(log_rotation, eps)

    # compute log translation: log_t = V^-1 * t
    log_translation = wp.mul(V_inv, t)

    # return combined result
    return wp_vec6(
        log_translation[0],
        log_translation[1],
        log_translation[2],
        log_rotation[0],
        log_rotation[1],
        log_rotation[2],
    )


@wp.kernel
def se3_log_map_kernel(
    xyz_wxyz: wp.array(dtype=wp_vec7),
    eps: wp.float32,
    log_transform: wp.array(dtype=wp_vec6),
):
    """SE3 logarithm map kernel."""
    tid = wp.tid()
    log_transform[tid] = se3_log_map_func(xyz_wxyz[tid], eps)


@wp.func
def se3_adjoint_func(xyz_wxyz: wp_vec7) -> wp_mat66:
    # extract translation and quaternion
    t = wp.vec3(xyz_wxyz[0], xyz_wxyz[1], xyz_wxyz[2])
    q_wxyz = wp.vec4(xyz_wxyz[3], xyz_wxyz[4], xyz_wxyz[5], xyz_wxyz[6])

    # get rotation matrix and translation
    R = quaternion_to_matrix_func(q_wxyz)
    t_skew = wp.mat33(0.0, -t[2], t[1], t[2], 0.0, -t[0], -t[1], t[0], 0.0)
    t_skew_R = t_skew * R

    # construct 6x6 adjoint matrix
    # [[R, t_skew_R],
    #  [0, R]]
    # fmt: off
    return wp_mat66(
        R[0, 0], R[0, 1], R[0, 2], t_skew_R[0, 0], t_skew_R[0, 1], t_skew_R[0, 2],
        R[1, 0], R[1, 1], R[1, 2], t_skew_R[1, 0], t_skew_R[1, 1], t_skew_R[1, 2],
        R[2, 0], R[2, 1], R[2, 2], t_skew_R[2, 0], t_skew_R[2, 1], t_skew_R[2, 2],
        0.0, 0.0, 0.0, R[0, 0], R[0, 1], R[0, 2],
        0.0, 0.0, 0.0, R[1, 0], R[1, 1], R[1, 2],
        0.0, 0.0, 0.0, R[2, 0], R[2, 1], R[2, 2],
    )
    # fmt: on


@wp.kernel
def se3_adjoint_kernel(
    xyz_wxyz: wp.array(dtype=wp_vec7),
    result: wp.array(dtype=wp_mat66),
):
    tid = wp.tid()
    result[tid] = se3_adjoint_func(xyz_wxyz[tid])


@wp.func
def se3_jlog_func(xyz_wxyz: wp_vec7, eps: wp.float32) -> wp_mat66:
    """SE(3) jlog function.

    Args:
        xyz_wxyz: SE(3) pose in [x, y, z, qw, qx, qy, qz] format
        eps: Epsilon for numerical stability

    Returns:
        6x6 jlog matrix
    """
    # extract translation and quaternion
    xyz = wp.vec3(xyz_wxyz[0], xyz_wxyz[1], xyz_wxyz[2])
    quat_wxyz = wp.vec4(xyz_wxyz[3], xyz_wxyz[4], xyz_wxyz[5], xyz_wxyz[6])

    w = quaternion_to_axis_angle_func(quat_wxyz)
    theta_squared = wp.dot(w, w)
    theta = wp.sqrt(theta_squared)
    use_taylor = theta_squared < eps

    # compute SO(3) jlog
    jlog_so3 = so3_jac_left_inv_func(w, eps)
    jlog_so3_T = wp.transpose(jlog_so3)

    # safe versions for division
    theta_safe = wp.where(use_taylor, 1.0, theta)
    theta_inv = 1.0 / theta_safe
    theta_squared_inv = theta_inv * theta_inv

    # trigonometric functions
    st = wp.sin(theta)
    ct = wp.cos(theta)

    # compute intermediate terms
    one_minus_ct = 1.0 - ct
    inv_2_2ct = wp.where(use_taylor, 0.5, 1.0 / (2.0 * one_minus_ct))
    beta = theta_squared_inv - st * theta_inv * inv_2_2ct
    beta_dot_over_theta = (
        -2.0 * theta_squared_inv * theta_squared_inv + (1.0 + st * theta_inv) * theta_squared_inv * inv_2_2ct
    )

    # cross term w^T * p
    wTp = wp.dot(w, xyz)

    # compute v3_tmp term
    v3_tmp = beta_dot_over_theta * wTp * w - (theta_squared * beta_dot_over_theta + 2.0 * beta) * xyz

    # compute C matrix components
    C = wp.outer(v3_tmp, w) + beta * wp.outer(w, xyz) + (wTp * beta) * wp.identity(n=3, dtype=wp.float32)
    # add skew-symmetric part
    skew_t = hat_func(xyz)
    C = C + 0.5 * skew_t
    # compute B = C @ jlog_so3
    B = C * jlog_so3_T
    # for small angles, use Taylor expansion
    B_taylor = 0.5 * skew_t
    B_final = wp.where(use_taylor, B_taylor, B)

    # construct 6x6 matrix from blocks in row-major order
    # top-left 3x3: jlog_so3^T, top-right 3x3: B_final
    # bottom-left 3x3: zeros, bottom-right 3x3: jlog_so3^T
    # fmt: off
    return wp_mat66(
        jlog_so3_T[0, 0], jlog_so3_T[0, 1], jlog_so3_T[0, 2], B_final[0, 0], B_final[0, 1], B_final[0, 2],
        jlog_so3_T[1, 0], jlog_so3_T[1, 1], jlog_so3_T[1, 2], B_final[1, 0], B_final[1, 1], B_final[1, 2],
        jlog_so3_T[2, 0], jlog_so3_T[2, 1], jlog_so3_T[2, 2], B_final[2, 0], B_final[2, 1], B_final[2, 2],
        0.0, 0.0, 0.0, jlog_so3_T[0, 0], jlog_so3_T[0, 1], jlog_so3_T[0, 2],
        0.0, 0.0, 0.0, jlog_so3_T[1, 0], jlog_so3_T[1, 1], jlog_so3_T[1, 2],
        0.0, 0.0, 0.0, jlog_so3_T[2, 0], jlog_so3_T[2, 1], jlog_so3_T[2, 2]
    )
    # fmt: on


@wp.kernel
def se3_jlog_kernel(
    xyz_wxyz: wp.array(dtype=wp_vec7),
    eps: wp.float32,
    result: wp.array(dtype=wp_mat66),
):
    """SE(3) jlog kernel."""
    tid = wp.tid()
    result[tid] = se3_jlog_func(xyz_wxyz[tid], eps)
