# pyright: reportArgumentType=false
# pyright: reportOptionalOperand=false
import logging
from typing import Optional, Tuple, Union

import numpy as np
import warp as wp

from robokit.lie.se3_kernels import (
    se3_adjoint_kernel,
    se3_apply_kernel,
    se3_compose_kernel,
    se3_exp_map_kernel,
    se3_exp_map_to_matrix_kernel,
    se3_from_matrix_kernel,
    se3_inverse_kernel,
    se3_jlog_kernel,
    se3_log_map_kernel,
    se3_to_matrix_kernel,
)
from robokit.opt.var_values import Var
from robokit.types import ArrayLike
from robokit.utils.warp_utils import gather, masked_copy, wp_device_type, wp_mat66, wp_vec6, wp_vec7


logger = logging.getLogger("robokit")


@wp.kernel
def _float32_to_vec6_kernel(
    src: wp.array2d(dtype=wp.float32),
    dst: wp.array(dtype=wp_vec6),
):
    i = wp.tid()
    dst[i] = wp_vec6(src[i, 0], src[i, 1], src[i, 2], src[i, 3], src[i, 4], src[i, 5])


def se3_identity(
    shape: Union[int, Tuple[int, ...]],
    device: Optional[wp_device_type] = None,
    requires_grad: bool = False,
) -> wp.array:
    """Create identity transforms as a typed `wp_vec7` array.

    Example:
        >>> se3_identity(1).numpy()[0].tolist()
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    """
    if isinstance(shape, int):
        shape = (shape,)
    identity = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return wp.from_numpy(
        np.broadcast_to(identity, (*shape, 7)).copy(),
        dtype=wp_vec7,
        device=device,
        requires_grad=requires_grad,
    )


def se3_from_matrix(matrix: wp.array) -> wp.array:
    """Convert homogeneous matrices to typed `xyz_wxyz` transforms.

    Example:
        >>> matrix = wp.from_numpy(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44)
        >>> se3_from_matrix(matrix).numpy()[0].tolist()
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    """
    xyz_wxyz = wp.empty(
        matrix.shape,
        dtype=wp_vec7,
        device=matrix.device,
        requires_grad=matrix.requires_grad,
    )
    wp.launch(
        kernel=se3_from_matrix_kernel,
        dim=matrix.size,
        inputs=[matrix.flatten()],
        outputs=[xyz_wxyz.flatten()],
        device=matrix.device,
    )
    return xyz_wxyz


def se3_to_matrix(T_dst_src: wp.array) -> wp.array:
    """Convert typed `xyz_wxyz` transforms to homogeneous matrices.

    Example:
        >>> np.allclose(se3_to_matrix(se3_identity(1)).numpy()[0], np.eye(4))
        True
    """
    matrix = wp.empty(
        T_dst_src.shape,
        dtype=wp.mat44,
        device=T_dst_src.device,
        requires_grad=T_dst_src.requires_grad,
    )
    wp.launch(
        kernel=se3_to_matrix_kernel,
        dim=T_dst_src.size,
        inputs=[T_dst_src.flatten()],
        outputs=[matrix.flatten()],
        device=T_dst_src.device,
    )
    return matrix


def se3_compose(T_dst_mid: wp.array, T_mid_src: wp.array, out: Optional[wp.array] = None) -> wp.array:
    """Compose typed transforms as `T_dst_src = T_dst_mid * T_mid_src`.

    Example:
        >>> se3_compose(se3_identity(1), se3_identity(1)).numpy()[0].tolist()
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    """
    if out is None:
        out = wp.empty(
            T_dst_mid.shape,
            dtype=wp_vec7,
            device=T_dst_mid.device,
            requires_grad=T_dst_mid.requires_grad or T_mid_src.requires_grad,
        )
    wp.launch(
        kernel=se3_compose_kernel,
        dim=T_dst_mid.size,
        inputs=[T_dst_mid.flatten(), T_mid_src.flatten()],
        outputs=[out.flatten()],
        device=T_dst_mid.device,
    )
    return out


def se3_inverse(T_dst_src: wp.array) -> wp.array:
    """Invert typed transforms.

    Example:
        >>> np.allclose(se3_inverse(se3_identity(1)).numpy()[0], [0, 0, 0, 1, 0, 0, 0])
        True
    """
    result = wp.empty(
        T_dst_src.shape,
        dtype=wp_vec7,
        device=T_dst_src.device,
        requires_grad=T_dst_src.requires_grad,
    )
    wp.launch(
        kernel=se3_inverse_kernel,
        dim=T_dst_src.size,
        inputs=[T_dst_src.flatten()],
        outputs=[result.flatten()],
        device=T_dst_src.device,
    )
    return result


def se3_apply(T_dst_src: wp.array, points_src: wp.array) -> wp.array:
    """Apply typed transforms to points.

    Example:
        >>> points = wp.from_numpy(np.array([[1, 2, 3]], dtype=np.float32), dtype=wp.vec3)
        >>> se3_apply(se3_identity(1), points).numpy()[0].tolist()
        [1.0, 2.0, 3.0]
    """
    points_dst = wp.empty(
        points_src.shape,
        dtype=wp.vec3,
        device=points_src.device,
        requires_grad=points_src.requires_grad or T_dst_src.requires_grad,
    )
    wp.launch(
        kernel=se3_apply_kernel,
        dim=points_src.size,
        inputs=[T_dst_src.flatten(), points_src.flatten()],
        outputs=[points_dst.flatten()],
        device=points_src.device,
    )
    return points_dst


def se3_exp(log_transform: wp.array, eps: float = 1e-4) -> wp.array:
    """Map translation-first twists to typed transforms.

    Example:
        >>> twist = wp.from_numpy(np.zeros((1, 6), dtype=np.float32), dtype=wp_vec6)
        >>> se3_exp(twist).numpy()[0].tolist()
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    """
    T = wp.empty(
        log_transform.shape,
        dtype=wp_vec7,
        device=log_transform.device,
        requires_grad=log_transform.requires_grad,
    )
    wp.launch(
        kernel=se3_exp_map_kernel,
        dim=log_transform.size,
        inputs=[log_transform.flatten(), eps],
        outputs=[T.flatten()],
        device=log_transform.device,
    )
    return T


def se3_exp_to_matrix(log_transform: wp.array, eps: float = 1e-4) -> wp.array:
    """Map translation-first twists directly to homogeneous matrices.

    Example:
        >>> twist = wp.from_numpy(np.zeros((1, 6), dtype=np.float32), dtype=wp_vec6)
        >>> np.allclose(se3_exp_to_matrix(twist).numpy()[0], np.eye(4))
        True
    """
    matrix = wp.empty(
        log_transform.shape,
        dtype=wp.mat44,
        device=log_transform.device,
        requires_grad=log_transform.requires_grad,
    )
    wp.launch(
        kernel=se3_exp_map_to_matrix_kernel,
        dim=log_transform.size,
        inputs=[log_transform.flatten(), eps],
        outputs=[matrix.flatten()],
        device=log_transform.device,
    )
    return matrix


def se3_log(T_dst_src: wp.array, eps: float = 1e-4) -> wp.array:
    """Map typed transforms to translation-first twists.

    Example:
        >>> se3_log(se3_identity(1)).numpy()[0].tolist()
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    """
    log_transform = wp.empty(
        T_dst_src.shape,
        dtype=wp_vec6,
        device=T_dst_src.device,
        requires_grad=T_dst_src.requires_grad,
    )
    wp.launch(
        kernel=se3_log_map_kernel,
        dim=T_dst_src.size,
        inputs=[T_dst_src.flatten(), eps],
        outputs=[log_transform.flatten()],
        device=T_dst_src.device,
    )
    return log_transform


def se3_adjoint(T_dst_src: wp.array) -> wp.array:
    """Compute adjoint matrices for typed transforms.

    Example:
        >>> np.allclose(se3_adjoint(se3_identity(1)).numpy()[0], np.eye(6))
        True
    """
    result = wp.empty(
        T_dst_src.shape,
        dtype=wp_mat66,
        device=T_dst_src.device,
        requires_grad=T_dst_src.requires_grad,
    )
    wp.launch(
        kernel=se3_adjoint_kernel,
        dim=T_dst_src.size,
        inputs=[T_dst_src.flatten(), result.flatten()],
        device=T_dst_src.device,
    )
    return result


def se3_jlog(T_dst_src: wp.array, eps: float = 1e-4) -> wp.array:
    """Compute logarithm-map Jacobians for typed transforms.

    Example:
        >>> bool(np.linalg.norm(se3_jlog(se3_identity(1)).numpy()[0] - np.eye(6)) < 1e-2)
        True
    """
    result = wp.empty(
        T_dst_src.shape,
        dtype=wp_mat66,
        device=T_dst_src.device,
        requires_grad=T_dst_src.requires_grad,
    )
    wp.launch(
        kernel=se3_jlog_kernel,
        dim=T_dst_src.size,
        inputs=[T_dst_src.flatten(), eps, result.flatten()],
        device=T_dst_src.device,
    )
    return result


class SE3Var(Var):
    """SE(3) pose var: `wp_vec7 [x, y, z, qw, qx, qy, qz]`; grow behavior via `se3_*` functions, not new methods."""

    _xyz_wxyz: wp.array

    def __init__(self, xyz_wxyz: wp.array):
        self._xyz_wxyz = xyz_wxyz.contiguous()

    def __repr__(self) -> str:
        return f"SE3Var(xyz_wxyz={np.round(self._xyz_wxyz.numpy(), decimals=4)})"

    @property
    def xyz_wxyz(self) -> wp.array:
        return self._xyz_wxyz

    @xyz_wxyz.setter
    def xyz_wxyz(self, value: wp.array):
        self._xyz_wxyz = value.contiguous()

    @property
    def tangent_dim(self) -> int:
        return 6

    @property
    def batch_size(self) -> int:
        if self._xyz_wxyz.ndim == 0:
            return 1
        return int(self._xyz_wxyz.shape[0])

    @property
    def device(self) -> wp_device_type:
        return self._xyz_wxyz.device

    def clone(self) -> "SE3Var":
        return SE3Var(wp.clone(self._xyz_wxyz))

    def view(self, batch_size: int, offset: int = 0) -> "SE3Var":
        return SE3Var(self._xyz_wxyz[offset : offset + batch_size])

    def integrate(
        self,
        velocity: ArrayLike,
        out: Optional["SE3Var"] = None,
        tangent_mask: Optional[wp.array] = None,
        weight_decay: float = 0.0,
        out_velocity: Optional[ArrayLike] = None,
    ) -> "SE3Var":
        del tangent_mask, out_velocity  # SE(3) has no per-DOF freezing; the solvers pass one regardless
        if weight_decay != 0.0:
            raise ValueError("SE3Var does not support weight decay")
        if velocity.dtype == wp.float32 and velocity.ndim == 2:
            # VarValues hands out a strided float32 slice; se3_exp needs a typed vec6
            vec6_vel = wp.empty(
                velocity.shape[0], dtype=wp_vec6, device=velocity.device, requires_grad=velocity.requires_grad
            )
            wp.launch(
                kernel=_float32_to_vec6_kernel,
                dim=velocity.shape[0],
                inputs=[velocity],
                outputs=[vec6_vel],
                device=velocity.device,
            )
            velocity = vec6_vel
        delta = se3_exp(velocity)
        if out is None:
            return SE3Var(se3_compose(self._xyz_wxyz, delta))
        se3_compose(self._xyz_wxyz, delta, out=out.xyz_wxyz)
        return out

    def gather(self, indices: ArrayLike, out: Optional["SE3Var"] = None) -> "SE3Var":
        """Gather poses by batch index, into `out` when preallocated."""
        return SE3Var(gather(self._xyz_wxyz, indices, out._xyz_wxyz if out is not None else None))

    def accept(self, accept_mask: ArrayLike, proposed: "SE3Var") -> "SE3Var":
        """Accept proposed values where accept_mask[i] == 1, keep self unchanged otherwise. In-place."""
        masked_copy(accept_mask, proposed._xyz_wxyz, self._xyz_wxyz)
        return self
