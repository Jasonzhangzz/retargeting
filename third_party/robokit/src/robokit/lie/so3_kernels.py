# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
import warp as wp


@wp.func
def hat_func(v: wp.vec3) -> wp.mat33:
    """Warp implementation of hat operator (skew-symmetric matrix)"""
    return wp.mat33(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)


@wp.func
def so3_jac_left_inv_func(theta: wp.vec3, eps: wp.float32) -> wp.mat33:
    """Warp implementation of SO(3) left jacobian inverse."""
    theta_squared = wp.dot(theta, theta)
    use_taylor = theta_squared < eps

    # safe versions to avoid division by zero
    theta_squared_safe = wp.where(use_taylor, 1.0, theta_squared)
    theta_safe = wp.sqrt(theta_squared_safe)
    half_theta_safe = theta_safe / 2.0

    # skew-symmetric matrix and its square
    skew_omega = hat_func(theta)
    skew_omega_squared = skew_omega * skew_omega

    eye = wp.identity(n=3, dtype=theta.dtype)

    # taylor series
    taylor_result = eye - 0.5 * skew_omega + theta_squared_safe * skew_omega_squared / 12.0

    # full computation
    cos_half_theta = wp.cos(half_theta_safe)
    sin_half_theta = wp.sin(half_theta_safe)
    coeff = (1.0 - theta_safe * cos_half_theta / (2.0 * sin_half_theta)) / theta_squared_safe

    full_result = eye - 0.5 * skew_omega + coeff * skew_omega_squared
    result = wp.where(use_taylor, taylor_result, full_result)
    return result


@wp.kernel
def so3_jlog_kernel(
    theta: wp.array(dtype=wp.vec3),
    eps: wp.float32,
    result: wp.array(dtype=wp.mat33),
):
    i = wp.tid()

    V_inv = so3_jac_left_inv_func(theta[i], eps)
    V_inv_T = wp.transpose(V_inv)
    result[i] = V_inv_T
