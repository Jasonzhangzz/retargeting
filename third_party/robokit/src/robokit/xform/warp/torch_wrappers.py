"""Torch-autograd wrappers over the Warp kernels; tensors must be float32."""

from typing import Dict, Literal, Optional, Tuple, cast, get_args

import torch
import warp as wp
from jaxtyping import Float

from robokit.xform.warp.rotation_conversions import (
    axis_angle_to_quaternion_kernel,
    euler_angles_to_matrix_kernel,
    matrix_to_euler_angles_kernel,
    matrix_to_quaternion_kernel,
    quaternion_apply_kernel,
    quaternion_invert_kernel,
    quaternion_multiply_kernel,
    quaternion_to_axis_angle_kernel,
    quaternion_to_matrix_kernel,
)
from robokit.xform.warp.transforms import (
    inverse_tf_mat_kernel,
    rotate_points_kernel,
    transform_points_kernel,
)


def axis_angle_to_matrix(axis_angle: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 3 3"]:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle (torch.Tensor): Rotation vectors with shape [..., 3], whose magnitude is the angle in radians turned anticlockwise around their direction.

    Returns:
        torch.Tensor: Rotation matrices [..., 3, 3].

    Example:
        >>> axis_angle_to_matrix(torch.tensor([0.5, 0.0, 0.0]))
        tensor([[ 1.0000,  0.0000,  0.0000],
                [ 0.0000,  0.8776, -0.4794],
                [ 0.0000,  0.4794,  0.8776]])
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


class AxisAngleToQuaternion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, axis_angle: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 4"]:
        wp.init()

        # flatten input for kernel processing
        original_shape = axis_angle.shape
        axis_angle_flat = axis_angle.contiguous().view(-1, 3)

        axis_angle_wp = wp.from_torch(axis_angle_flat, dtype=wp.vec3, requires_grad=axis_angle.requires_grad)

        quaternions_wp = wp.from_torch(
            torch.empty(
                axis_angle_flat.shape[0],
                4,
                dtype=axis_angle.dtype,
                device=axis_angle.device,
                requires_grad=axis_angle.requires_grad,
            ),
            dtype=wp.vec4,
            requires_grad=axis_angle.requires_grad,
        )

        wp.launch(
            kernel=axis_angle_to_quaternion_kernel,
            dim=(axis_angle_wp.shape[0],),
            inputs=[axis_angle_wp],
            outputs=[quaternions_wp],
            device=axis_angle_wp.device,
        )

        if axis_angle.requires_grad:
            ctx.axis_angle_wp = axis_angle_wp
            ctx.quaternions_wp = quaternions_wp
            ctx.original_shape = original_shape

        return wp.to_torch(quaternions_wp).view(original_shape[:-1] + (4,))

    @staticmethod
    def backward(ctx, quaternion_grad: Float[torch.Tensor, "... 4"]) -> Tuple[Optional[Float[torch.Tensor, "... 3"]]]:
        wp.init()

        ctx.quaternions_wp.grad = wp.from_torch(quaternion_grad.contiguous().view(-1, 4), dtype=wp.vec4)

        wp.launch(
            kernel=axis_angle_to_quaternion_kernel,
            dim=(ctx.axis_angle_wp.shape[0],),
            inputs=[ctx.axis_angle_wp],
            outputs=[ctx.quaternions_wp],
            adj_inputs=[ctx.axis_angle_wp.grad],
            adj_outputs=[ctx.quaternions_wp.grad],
            adjoint=True,
            device=ctx.axis_angle_wp.device,
        )

        axis_angle_grad = wp.to_torch(ctx.axis_angle_wp.grad).view(ctx.original_shape)
        return (axis_angle_grad,)


def axis_angle_to_quaternion(axis_angle: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 4"]:
    """
    Convert rotations given as axis/angle to quaternions using Warp.

    Args:
        axis_angle: Rotations given as a vector in axis angle form, as a tensor of shape (..., 3), where the magnitude is the angle turned anticlockwise in radians around the vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).

    Example:
        >>> axis_angle = torch.tensor([1.5708, 0.0, 0.0])
        >>> expected = torch.tensor([0.7071, 0.7071, 0.0000, 0.0000])
        >>> torch.allclose(axis_angle_to_quaternion(axis_angle), expected, atol=1e-4)
        True
    """
    return AxisAngleToQuaternion.apply(axis_angle)  # type: ignore


# fmt: off
_AXES = Literal[
    "sxyz", "sxyx", "sxzy", "sxzx", "syzx", "syzy", "syxz", "syxy", "szxy", "szxz", "szyx", "szyz",
    "rzyx", "rxyx", "ryzx", "rxzx", "rxzy", "ryzy", "rzxy", "ryxy", "ryxz", "rzxz", "rxyz", "rzyz"
]
# fmt: on
_AXES_SPEC: Dict[_AXES, wp.vec4i] = {
    axes: wp.vec4i("sr".index(axes[0]), "xyz".index(axes[1]), "xyz".index(axes[2]), "xyz".index(axes[3]))  # type: ignore
    for axes in get_args(_AXES)
}


def _axes_spec(axes: str) -> wp.vec4i:
    """Kernel spec of an axes string; three-character forms get a leading "s"."""
    axes = axes.lower() if len(axes) == 4 else f"s{axes.lower()}"
    if axes not in _AXES_SPEC:
        raise ValueError(f"Invalid axes: {axes}")
    return _AXES_SPEC[axes]  # type: ignore


class EulerAnglesToMatrix(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, euler_angles: Float[torch.Tensor, "... 3"], axes: _AXES = "sxyz"
    ) -> Float[torch.Tensor, "... 3 3"]:
        axes_spec = _axes_spec(axes)
        wp.init()
        euler_angles_wp = wp.from_torch(
            euler_angles.contiguous().view(-1, 3), dtype=wp.vec3, requires_grad=euler_angles.requires_grad
        )
        rot_mat_wp = wp.from_torch(
            torch.empty(
                euler_angles.shape + (3,),
                dtype=euler_angles.dtype,
                device=euler_angles.device,
                requires_grad=euler_angles.requires_grad,
            ).view(-1, 3, 3),
            dtype=wp.mat33,
            requires_grad=euler_angles.requires_grad,
        )

        wp.launch(
            kernel=euler_angles_to_matrix_kernel,
            dim=(euler_angles_wp.shape[0],),
            inputs=[euler_angles_wp, axes_spec],
            outputs=[rot_mat_wp],
            device=euler_angles_wp.device,
        )
        if euler_angles.requires_grad:
            ctx.euler_angles_wp = euler_angles_wp
            ctx.rot_mat_wp = rot_mat_wp
            ctx.axes_spec = axes_spec
        return wp.to_torch(rot_mat_wp).view(euler_angles.shape + (3,))

    @staticmethod
    def backward(  # type: ignore
        ctx, rot_mat_grad: Float[torch.Tensor, "... 3 3"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 3"]], None]:
        wp.init()
        ctx.rot_mat_wp.grad = wp.from_torch(rot_mat_grad.contiguous().view(-1, 3, 3), dtype=wp.mat33)
        wp.launch(
            kernel=euler_angles_to_matrix_kernel,
            dim=(ctx.euler_angles_wp.shape[0],),
            inputs=[ctx.euler_angles_wp, ctx.axes_spec],
            outputs=[ctx.rot_mat_wp],
            adj_inputs=[ctx.euler_angles_wp.grad, ctx.axes_spec],
            adj_outputs=[ctx.rot_mat_wp.grad],
            adjoint=True,
            device=ctx.euler_angles_wp.device,
        )
        return wp.to_torch(ctx.euler_angles_wp.grad).view(rot_mat_grad.shape[:-1]), None


def euler_angles_to_matrix(
    euler_angles: Float[torch.Tensor, "... 3"], axes: _AXES = "sxyz"
) -> Float[torch.Tensor, "... 3 3"]:
    """Converts Euler angles to rotation matrices.

    Args:
        euler_angles (torch.Tensor): Tensor of Euler angles with shape [..., 3].
        axes (str): One of 24 axis sequences, e.g. "sxyz"; three-character forms get a leading "s".

    Returns:
        torch.Tensor: Rotation matrices with shape [..., 3, 3].

    Example:
        >>> euler_angles = torch.tensor([1.0, 0.5, 2.0])
        >>> euler_angles_to_matrix(euler_angles, axes="sxyz")
        tensor([[-0.3652, -0.6592,  0.6574],
                [ 0.7980,  0.1420,  0.5857],
                [-0.4794,  0.7385,  0.4742]])
        >>> euler_angles_to_matrix(euler_angles, axes="rxyz")
        tensor([[-0.3652, -0.7980,  0.4794],
                [ 0.3234, -0.5917, -0.7385],
                [ 0.8729, -0.1146,  0.4742]])
    """
    return EulerAnglesToMatrix.apply(euler_angles, axes)  # type: ignore


class MatrixToEulerAngles(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rot_mat: Float[torch.Tensor, "... 3 3"], axes: _AXES = "sxyz") -> Float[torch.Tensor, "... 3"]:
        axes_spec = _axes_spec(axes)
        wp.init()
        rot_mat_wp = wp.from_torch(
            rot_mat.contiguous().view(-1, 3, 3), dtype=wp.mat33, requires_grad=rot_mat.requires_grad
        )
        euler_angles_wp = wp.from_torch(
            torch.empty(
                rot_mat.shape[:-2] + (3,),
                dtype=rot_mat.dtype,
                device=rot_mat.device,
                requires_grad=rot_mat.requires_grad,
            ).view(-1, 3),
            dtype=wp.vec3,
            requires_grad=rot_mat.requires_grad,
        )

        wp.launch(
            kernel=matrix_to_euler_angles_kernel,
            dim=(rot_mat_wp.shape[0],),
            inputs=[rot_mat_wp, axes_spec],
            outputs=[euler_angles_wp],
            device=rot_mat_wp.device,
        )

        if rot_mat.requires_grad:
            ctx.rot_mat_wp = rot_mat_wp
            ctx.euler_angles_wp = euler_angles_wp
            ctx.axes_spec = axes_spec

        return wp.to_torch(euler_angles_wp).view(rot_mat.shape[:-2] + (3,))

    @staticmethod
    def backward(  # type: ignore
        ctx, euler_angles_grad: Float[torch.Tensor, "... 3"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 3 3"]], None]:
        wp.init()
        ctx.euler_angles_wp.grad = wp.from_torch(euler_angles_grad.contiguous().view(-1, 3), dtype=wp.vec3)
        wp.launch(
            kernel=matrix_to_euler_angles_kernel,
            dim=(ctx.rot_mat_wp.shape[0],),
            inputs=[ctx.rot_mat_wp, ctx.axes_spec],
            outputs=[ctx.euler_angles_wp],
            adj_inputs=[ctx.rot_mat_wp.grad, ctx.axes_spec],
            adj_outputs=[ctx.euler_angles_wp.grad],
            adjoint=True,
            device=ctx.rot_mat_wp.device,
        )
        return wp.to_torch(ctx.rot_mat_wp.grad).view(euler_angles_grad.shape[:-1] + (3, 3)), None


def matrix_to_euler_angles(
    rot_mat: Float[torch.Tensor, "... 3 3"], axes: _AXES = "sxyz"
) -> Float[torch.Tensor, "... 3"]:
    """Converts rotation matrices to Euler angles, the inverse of `euler_angles_to_matrix`.

    Args:
        rot_mat (torch.Tensor): Rotation matrices with shape [..., 3, 3].
        axes (str): One of 24 axis sequences, e.g. "sxyz"; three-character forms get a leading "s".

    Returns:
        torch.Tensor: Euler angles with shape [..., 3]. As in scipy, the middle angle lies in [-pi/2, pi/2] for Tait-Bryan axes and in [0, pi] for proper Euler axes, the others in [-pi, pi], and the third angle is zero at gimbal lock, where the gradient is undefined.

    Example:
        >>> euler_angles = torch.tensor([1.0, 0.5, 2.0])
        >>> rot_mat = euler_angles_to_matrix(euler_angles, axes="rxyz")
        >>> matrix_to_euler_angles(rot_mat, axes="rxyz")
        tensor([1.0000, 0.5000, 2.0000])
    """
    return MatrixToEulerAngles.apply(rot_mat, axes)  # type: ignore


def matrix_to_nearest_euler_angles(rot_mat: Float[torch.Tensor, "... t 3 3"]) -> Float[torch.Tensor, "... t 3"]:
    """Converts a rotation sequence to `rxyz` Euler angles, picking per frame the equivalent angles nearest the last.

    Args:
        rot_mat (torch.Tensor): Rotation matrices with shape [..., T, 3, 3], ordered along T.

    Returns:
        torch.Tensor: Continuous Euler angles with shape [..., T, 3] that reproduce `rot_mat` exactly.

    Example:
        >>> angles = torch.tensor([[3.1, 0.2, -3.1], [-3.1, 0.2, 3.1]])
        >>> matrix_to_nearest_euler_angles(euler_angles_to_matrix(angles, "rxyz"))
        tensor([[ 3.1000,  0.2000, -3.1000],
                [ 3.1832,  0.2000, -3.1832]])
    """
    principal = matrix_to_euler_angles(rot_mat, "rxyz")
    frames = [principal[..., 0, :]]
    for index in range(1, principal.shape[-2]):
        angles = principal[..., index, :]
        other = torch.stack([angles[..., 0] + torch.pi, torch.pi - angles[..., 1], angles[..., 2] + torch.pi], dim=-1)
        delta = torch.remainder(
            torch.stack([angles, other], dim=-2) - frames[-1].unsqueeze(-2) + torch.pi, 2 * torch.pi
        )
        delta = delta - torch.pi
        choice = delta.square().sum(dim=-1).argmin(dim=-1)
        frames.append(frames[-1] + delta.gather(-2, choice[..., None, None].expand(*choice.shape, 1, 3)).squeeze(-2))
    return torch.stack(frames, dim=-2)


class MatrixToQuaternion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rot_mat: Float[torch.Tensor, "... 3 3"]) -> Float[torch.Tensor, "... 4"]:
        wp.init()
        rot_mat_wp = wp.from_torch(
            rot_mat.contiguous().view(-1, 3, 3), dtype=wp.mat33, requires_grad=rot_mat.requires_grad
        )
        quat_wp = wp.from_torch(
            torch.empty(
                rot_mat.shape[:-2] + (4,),
                dtype=rot_mat.dtype,
                device=rot_mat.device,
                requires_grad=rot_mat.requires_grad,
            ).view(-1, 4),
            dtype=wp.vec4,
            requires_grad=rot_mat.requires_grad,
        )
        wp.launch(
            kernel=matrix_to_quaternion_kernel,
            dim=(rot_mat_wp.shape[0],),
            inputs=[rot_mat_wp],
            outputs=[quat_wp],
            device=rot_mat_wp.device,
        )
        if rot_mat.requires_grad:
            ctx.rot_mat_wp = rot_mat_wp
            ctx.quat_wp = quat_wp
        return wp.to_torch(quat_wp).view(rot_mat.shape[:-2] + (4,))

    @staticmethod
    def backward(  # type: ignore
        ctx, quat_grad: Float[torch.Tensor, "... 4"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 3 3"]]]:
        wp.init()
        ctx.quat_wp.grad = wp.from_torch(quat_grad.contiguous().view(-1, 4), dtype=wp.vec4)
        wp.launch(
            kernel=matrix_to_quaternion_kernel,
            dim=(ctx.rot_mat_wp.shape[0],),
            inputs=[ctx.rot_mat_wp],
            outputs=[ctx.quat_wp],
            adj_inputs=[ctx.rot_mat_wp.grad],
            adj_outputs=[ctx.quat_wp.grad],
            adjoint=True,
            device=ctx.rot_mat_wp.device,
        )
        return (wp.to_torch(ctx.rot_mat_wp.grad).view(quat_grad.shape[:-1] + (3, 3)),)


def matrix_to_quaternion(rot_mat: Float[torch.Tensor, "... 3 3"]) -> Float[torch.Tensor, "... 4"]:
    """
    Converts rotation matrices to quaternions (wxyz format).

    Args:
        rot_mat (torch.Tensor): Rotation matrices with shape [..., 3, 3].

    Returns:
        torch.Tensor: Quaternions with shape [..., 4] in wxyz format.

    Example:
        >>> rot_mat = torch.tensor([[-0.2533, -0.6075,  0.7529],
        ...                         [ 0.8445, -0.5185, -0.1343],
        ...                         [ 0.4720,  0.6017,  0.6443]])
        >>> matrix_to_quaternion(rot_mat)
        tensor([0.4671, 0.3940, 0.1503, 0.7772])

    Note:
        The gradient of this function differs from the pytorch3d implementation, but it should be okay for most use cases. See https://github.com/facebookresearch/pytorch3d/issues/503#issuecomment-755493515.
    """
    return MatrixToQuaternion.apply(rot_mat)  # type: ignore


class QuaternionToMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_wxyz: Float[torch.Tensor, "... 4"]) -> Float[torch.Tensor, "... 3 3"]:
        wp.init()
        q_wxyz_wp = wp.from_torch(q_wxyz.contiguous().view(-1, 4), dtype=wp.vec4, requires_grad=q_wxyz.requires_grad)
        rot_mat_wp = wp.from_torch(
            torch.empty(
                q_wxyz.shape[:-1] + (3, 3),
                dtype=q_wxyz.dtype,
                device=q_wxyz.device,
                requires_grad=q_wxyz.requires_grad,
            ).view(-1, 3, 3),
            dtype=wp.mat33,
            requires_grad=q_wxyz.requires_grad,
        )

        wp.launch(
            kernel=quaternion_to_matrix_kernel,
            dim=(q_wxyz_wp.shape[0],),
            inputs=[q_wxyz_wp],
            outputs=[rot_mat_wp],
            device=q_wxyz_wp.device,
        )

        if q_wxyz.requires_grad:
            ctx.q_wxyz_wp = q_wxyz_wp
            ctx.rot_mat_wp = rot_mat_wp
            ctx.quaternion_shape = q_wxyz.shape

        return wp.to_torch(rot_mat_wp).view(q_wxyz.shape[:-1] + (3, 3))

    @staticmethod
    def backward(  # type: ignore
        ctx, rot_mat_grad: Float[torch.Tensor, "... 3 3"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 4"]], None]:
        wp.init()
        ctx.rot_mat_wp.grad = wp.from_torch(rot_mat_grad.contiguous().view(-1, 3, 3), dtype=wp.mat33)

        wp.launch(
            kernel=quaternion_to_matrix_kernel,
            dim=(ctx.q_wxyz_wp.shape[0],),
            inputs=[ctx.q_wxyz_wp],
            outputs=[ctx.rot_mat_wp],
            adj_inputs=[ctx.q_wxyz_wp.grad],
            adj_outputs=[ctx.rot_mat_wp.grad],
            adjoint=True,
            device=ctx.q_wxyz_wp.device,
        )

        return wp.to_torch(ctx.q_wxyz_wp.grad).view(ctx.quaternion_shape), None


def quaternion_to_matrix(q_wxyz: Float[torch.Tensor, "... 4"]) -> Float[torch.Tensor, "... 3 3"]:
    """
    Converts quaternions to rotation matrices.

    Args:
        q_wxyz (torch.Tensor): Quaternions with shape [..., 4] in wxyz order.

    Returns:
        torch.Tensor: Rotation matrices with shape [..., 3, 3].

    Example:
        >>> q_wxyz = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068])  # 90° about z
        >>> expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        >>> torch.allclose(quaternion_to_matrix(q_wxyz), expected, atol=1e-6)
        True
    """
    return QuaternionToMatrix.apply(q_wxyz)  # type: ignore


class QuaternionApply(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_wxyz: Float[torch.Tensor, "... 4"],
        points: Float[torch.Tensor, "... 3"],
    ) -> Float[torch.Tensor, "... 3"]:
        wp.init()

        # flatten for kernel processing
        quat_wp = wp.from_torch(q_wxyz.contiguous().view(-1, 4), dtype=wp.vec4, requires_grad=q_wxyz.requires_grad)
        points_wp = wp.from_torch(points.contiguous().view(-1, 3), dtype=wp.vec3, requires_grad=points.requires_grad)

        out_points_wp = wp.from_torch(
            torch.empty_like(points).contiguous().view(-1, 3),
            dtype=wp.vec3,
            requires_grad=q_wxyz.requires_grad or points.requires_grad,
        )

        wp.launch(
            kernel=quaternion_apply_kernel,
            dim=(points_wp.shape[0],),
            inputs=[quat_wp, points_wp],
            outputs=[out_points_wp],
            device=quat_wp.device,
        )

        if q_wxyz.requires_grad or points.requires_grad:
            ctx.quat_wp = quat_wp
            ctx.points_wp = points_wp
            ctx.out_points_wp = out_points_wp
            ctx.output_shape = points.shape

        return wp.to_torch(out_points_wp).view(points.shape)

    @staticmethod
    def backward(  # type: ignore
        ctx, out_points_grad: Float[torch.Tensor, "... 3"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 4"]], Optional[Float[torch.Tensor, "... 3"]]]:
        wp.init()

        ctx.out_points_wp.grad = wp.from_torch(out_points_grad.contiguous().view(-1, 3), dtype=wp.vec3)

        wp.launch(
            kernel=quaternion_apply_kernel,
            dim=(ctx.points_wp.shape[0],),
            inputs=[ctx.quat_wp, ctx.points_wp],
            outputs=[ctx.out_points_wp],
            adj_inputs=[ctx.quat_wp.grad, ctx.points_wp.grad],
            adj_outputs=[ctx.out_points_wp.grad],
            adjoint=True,
            device=ctx.points_wp.device,
        )

        quat_grad = (
            wp.to_torch(ctx.quat_wp.grad).view(ctx.output_shape[:-1] + (4,)) if ctx.quat_wp.requires_grad else None
        )
        points_grad = wp.to_torch(ctx.points_wp.grad).view(ctx.output_shape) if ctx.points_wp.requires_grad else None

        return quat_grad, points_grad


def quaternion_apply(
    q_wxyz: Float[torch.Tensor, "... 4"], points: Float[torch.Tensor, "... 3"]
) -> Float[torch.Tensor, "... 3"]:
    """Apply quaternion rotation to 3D points.

    Args:
        q_wxyz (torch.Tensor): Quaternions with shape [..., 4] in wxyz order; normalized before use.
        points (torch.Tensor): 3D points with shape [..., 3].

    Returns:
        torch.Tensor: Rotated points with shape [..., 3].

    Example:
        >>> quaternion = torch.tensor([0.7071, 0.0, 0.0, 0.7071])
        >>> point = torch.tensor([[1.0, 0.0, 0.0]])
        >>> expected = torch.tensor([[0., 1., 0.]])
        >>> torch.allclose(quaternion_apply(quaternion, point), expected, atol=1e-4)
        True

    Note:
        The batch dimensions (...) of quaternions and points must be broadcastable.
        The quaternion rotation is applied using the efficient formula:
        v' = v + 2 * w * (qv × v) + 2 * qv × (qv × v)
        where quat = [w, x, y, z] and qv = [x, y, z].
    """
    if q_wxyz.device != points.device:
        raise ValueError(f"quaternions and points must be on the same device, got {q_wxyz.device} and {points.device}")

    # ensure broadcastable shapes
    q_wxyz = q_wxyz / torch.linalg.vector_norm(q_wxyz, dim=-1, keepdim=True)
    quat_batch_shape = q_wxyz.shape[:-1]
    points_batch_shape = points.shape[:-1]

    broadcasted_shape = torch.broadcast_shapes(quat_batch_shape, points_batch_shape)

    quat_expanded = q_wxyz.expand(broadcasted_shape + (4,))
    points_expanded = points.expand(broadcasted_shape + (3,))

    return QuaternionApply.apply(quat_expanded, points_expanded)  # type: ignore


class QuaternionInvert(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quat_wxyz: Float[torch.Tensor, "... 4"]) -> Float[torch.Tensor, "... 4"]:
        wp.init()

        quat_wxyz_wp = wp.from_torch(
            quat_wxyz.contiguous().view(-1, 4), dtype=wp.vec4, requires_grad=quat_wxyz.requires_grad
        )

        quat_inv_wp = wp.from_torch(
            torch.empty_like(quat_wxyz).view(-1, 4), dtype=wp.vec4, requires_grad=quat_wxyz.requires_grad
        )

        wp.launch(
            kernel=quaternion_invert_kernel,
            dim=(quat_wxyz_wp.shape[0],),
            inputs=[quat_wxyz_wp],
            outputs=[quat_inv_wp],
            device=quat_wxyz_wp.device,
        )

        if quat_wxyz.requires_grad:
            ctx.quat_wxyz_wp = quat_wxyz_wp
            ctx.quat_inv_wp = quat_inv_wp

        return wp.to_torch(quat_inv_wp).view(quat_wxyz.shape)

    @staticmethod
    def backward(  # type: ignore
        ctx, quat_inv_grad: Float[torch.Tensor, "... 4"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 4"]]]:
        wp.init()

        ctx.quat_inv_wp.grad = wp.from_torch(quat_inv_grad.contiguous().view(-1, 4), dtype=wp.vec4)

        wp.launch(
            kernel=quaternion_invert_kernel,
            dim=(ctx.quat_wxyz_wp.shape[0],),
            inputs=[ctx.quat_wxyz_wp],
            outputs=[ctx.quat_inv_wp],
            adj_inputs=[ctx.quat_wxyz_wp.grad],
            adj_outputs=[ctx.quat_inv_wp.grad],
            adjoint=True,
            device=ctx.quat_wxyz_wp.device,
        )

        return (wp.to_torch(ctx.quat_wxyz_wp.grad).view(quat_inv_grad.shape),)


def quaternion_invert(q_wxyz: Float[torch.Tensor, "... 4"]) -> Float[torch.Tensor, "... 4"]:
    """
    Invert quaternions (compute conjugate for unit quaternions).

    Args:
        q_wxyz (torch.Tensor): Quaternions with shape [..., 4] in wxyz format.

    Returns:
        torch.Tensor: Inverted quaternions with shape [..., 4] in wxyz format.

    Note:
        For unit quaternions, the inverse is the conjugate: q^-1 = [w, -x, -y, -z]

    Example:
        >>> q_wxyz = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068])
        >>> torch.allclose(quaternion_invert(q_wxyz), torch.tensor([0.7071068, 0.0, 0.0, -0.7071068]))
        True
    """
    return QuaternionInvert.apply(q_wxyz)  # type: ignore


class QuaternionToAxisAngle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quat_wxyz: Float[torch.Tensor, "... 4"]) -> Float[torch.Tensor, "... 3"]:
        wp.init()

        quat_wxyz_wp = wp.from_torch(
            quat_wxyz.contiguous().view(-1, 4), dtype=wp.vec4, requires_grad=quat_wxyz.requires_grad
        )

        axis_angle_wp = wp.from_torch(
            torch.empty(
                quat_wxyz.shape[:-1] + (3,),
                dtype=quat_wxyz.dtype,
                device=quat_wxyz.device,
                requires_grad=quat_wxyz.requires_grad,
            ).view(-1, 3),
            dtype=wp.vec3,
            requires_grad=quat_wxyz.requires_grad,
        )

        wp.launch(
            kernel=quaternion_to_axis_angle_kernel,
            dim=(quat_wxyz_wp.shape[0],),
            inputs=[quat_wxyz_wp],
            outputs=[axis_angle_wp],
            device=quat_wxyz_wp.device,
        )

        if quat_wxyz.requires_grad:
            ctx.quat_wxyz_wp = quat_wxyz_wp
            ctx.axis_angle_wp = axis_angle_wp

        return wp.to_torch(axis_angle_wp).view(quat_wxyz.shape[:-1] + (3,))

    @staticmethod
    def backward(  # type: ignore
        ctx, axis_angle_grad: Float[torch.Tensor, "... 3"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 4"]]]:
        wp.init()

        ctx.axis_angle_wp.grad = wp.from_torch(axis_angle_grad.contiguous().view(-1, 3), dtype=wp.vec3)

        wp.launch(
            kernel=quaternion_to_axis_angle_kernel,
            dim=(ctx.quat_wxyz_wp.shape[0],),
            inputs=[ctx.quat_wxyz_wp],
            outputs=[ctx.axis_angle_wp],
            adj_inputs=[ctx.quat_wxyz_wp.grad],
            adj_outputs=[ctx.axis_angle_wp.grad],
            adjoint=True,
            device=ctx.quat_wxyz_wp.device,
        )

        return (wp.to_torch(ctx.quat_wxyz_wp.grad).view(axis_angle_grad.shape[:-1] + (4,)),)


def quaternion_to_axis_angle(q_wxyz: Float[torch.Tensor, "... 4"]) -> Float[torch.Tensor, "... 3"]:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        q_wxyz: quaternions with shape (..., 4) in wxyz format.

    Returns:
        Rotations given as a vector in axis angle form, as a tensor of shape (..., 3), where the magnitude is the angle turned anticlockwise in radians around the vector's direction.

    Example:
        >>> q_wxyz = torch.tensor([0.7071, 0.7071, 0.0, 0.0])
        >>> expected = torch.tensor([1.5708, 0.0000, 0.0000])
        >>> torch.allclose(quaternion_to_axis_angle(q_wxyz), expected, atol=1e-4)
        True
    """
    return QuaternionToAxisAngle.apply(q_wxyz)  # type: ignore


class QuaternionMultiply(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q1_wxyz: Float[torch.Tensor, "... 4"],
        q2_wxyz: Float[torch.Tensor, "... 4"],
    ) -> Float[torch.Tensor, "... 4"]:
        wp.init()
        q1_wxyz_wp = wp.from_torch(q1_wxyz.contiguous().view(-1, 4), dtype=wp.vec4, requires_grad=q1_wxyz.requires_grad)
        q2_wxyz_wp = wp.from_torch(q2_wxyz.contiguous().view(-1, 4), dtype=wp.vec4, requires_grad=q2_wxyz.requires_grad)
        out_wp = wp.empty_like(q1_wxyz_wp)

        wp.launch(
            kernel=quaternion_multiply_kernel,
            dim=q1_wxyz_wp.shape[0],
            inputs=[q1_wxyz_wp, q2_wxyz_wp],
            outputs=[out_wp],
            device=q1_wxyz_wp.device,
        )

        if q1_wxyz.requires_grad or q2_wxyz.requires_grad:
            ctx.q1_wxyz_wp = q1_wxyz_wp
            ctx.q2_wxyz_wp = q2_wxyz_wp
            ctx.out_wp = out_wp

        return wp.to_torch(out_wp).view(q1_wxyz.shape)

    @staticmethod
    def backward(
        ctx, grad_out: Float[torch.Tensor, "... 4"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 4"]], Optional[Float[torch.Tensor, "... 4"]]]:
        wp.init()
        ctx.out_wp.grad = wp.from_torch(grad_out.contiguous().view(-1, 4), dtype=wp.vec4)

        wp.launch(
            kernel=quaternion_multiply_kernel,
            dim=ctx.q1_wxyz_wp.shape[0],
            inputs=[ctx.q1_wxyz_wp, ctx.q2_wxyz_wp],
            outputs=[ctx.out_wp],
            adj_inputs=[ctx.q1_wxyz_wp.grad, ctx.q2_wxyz_wp.grad],
            adj_outputs=[ctx.out_wp.grad],
            adjoint=True,
            device=ctx.q1_wxyz_wp.device,
        )

        grad_q1_wxyz = wp.to_torch(ctx.q1_wxyz_wp.grad).view(grad_out.shape) if ctx.q1_wxyz_wp.requires_grad else None
        grad_q2_wxyz = wp.to_torch(ctx.q2_wxyz_wp.grad).view(grad_out.shape) if ctx.q2_wxyz_wp.requires_grad else None
        return grad_q1_wxyz, grad_q2_wxyz


def quaternion_multiply(
    q_wxyz_1: Float[torch.Tensor, "... 4"],
    q_wxyz_2: Float[torch.Tensor, "... 4"],
) -> Float[torch.Tensor, "... 4"]:
    """Multiply two quaternions (wxyz).

    Args:
        q_wxyz_1 (torch.Tensor): First quaternion with shape [..., 4] in wxyz format.
        q_wxyz_2 (torch.Tensor): Second quaternion with shape [..., 4] in wxyz format.

    Returns:
        torch.Tensor: Resulting quaternion with shape [..., 4] in wxyz format.

    Example:
        >>> q_wxyz_1 = torch.tensor([0.9238795, 0.0, 0.0, 0.3826834])
        >>> q_wxyz_2 = torch.tensor([0.9238795, 0.0, 0.0, 0.3826834])
        >>> expected = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068])
        >>> torch.allclose(quaternion_multiply(q_wxyz_1, q_wxyz_2), expected, atol=1e-6)
        True
    """
    return QuaternionMultiply.apply(q_wxyz_1, q_wxyz_2)  # type: ignore


def matrix_to_axis_angle(rot_mat: Float[torch.Tensor, "... 3 3"]) -> Float[torch.Tensor, "... 3"]:
    """Convert rotation matrices to axis-angle vectors, whose magnitude is the rotation angle.

    Args:
        rot_mat (torch.Tensor): Rotation matrices with shape [..., 3, 3].

    Returns:
        torch.Tensor: Axis-angle vectors with shape [..., 3].

    Example:
        >>> rot_mat = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        >>> torch.allclose(matrix_to_axis_angle(rot_mat), torch.tensor([1.5708, 0.0, 0.0]), atol=1e-4)
        True
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(rot_mat))


def matrix_to_rotation_angle(rot_mat: Float[torch.Tensor, "... 3 3"]) -> Float[torch.Tensor, "..."]:
    """Compute the rotation angle of rotation matrices; pass `R_a.mT @ R_b` for the angle between two rotations.

    Args:
        rot_mat (torch.Tensor): Rotation matrices with shape [..., 3, 3].

    Returns:
        torch.Tensor: Rotation angles in radians, in [0, pi], with shape [...]; not differentiable at 0.

    Example:
        >>> rot_mat = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        >>> torch.allclose(matrix_to_rotation_angle(rot_mat), torch.tensor(torch.pi / 2))
        True
    """
    return torch.linalg.vector_norm(matrix_to_axis_angle(rot_mat), dim=-1)


def rotation_between_vectors(
    src: Float[torch.Tensor, "... 3"], dst: Float[torch.Tensor, "... 3"]
) -> Float[torch.Tensor, "... 3 3"]:
    """Compute the smallest rotation that turns the direction of `src` into the direction of `dst`.

    Args:
        src (torch.Tensor): Source vectors with shape [..., 3]; only their direction is used.
        dst (torch.Tensor): Destination vectors with shape [..., 3]; only their direction is used.

    Returns:
        torch.Tensor: Rotation matrices [..., 3, 3] with `rot_mat @ src` parallel to `dst`; a half turn for opposite vectors.

    Example:
        >>> rot_mat = rotation_between_vectors(torch.tensor([0.0, 0.0, 1.0]), torch.tensor([1.0, 0.0, 0.0]))
        >>> torch.allclose(rot_mat @ torch.tensor([0.0, 0.0, 1.0]), torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
        True
    """
    u = src / torch.linalg.vector_norm(src, dim=-1, keepdim=True)
    v = dst / torch.linalg.vector_norm(dst, dim=-1, keepdim=True)
    u, v = torch.broadcast_tensors(u, v)
    w, xyz = 1.0 + (u * v).sum(dim=-1, keepdim=True), torch.linalg.cross(u, v)
    # (nearly) opposite vectors: turn half way around any axis perpendicular to `src`
    helper = torch.where(u[..., :1].abs() < 0.9, u.new_tensor([1.0, 0.0, 0.0]), u.new_tensor([0.0, 1.0, 0.0]))
    opposite = w + torch.linalg.vector_norm(xyz, dim=-1, keepdim=True) < 1e-6
    xyz = torch.where(opposite, torch.linalg.cross(u, helper), xyz)
    return quaternion_to_matrix(torch.cat([torch.where(opposite, 0.0, w), xyz], dim=-1))


def quaternion_slerp(
    q_wxyz_1: Float[torch.Tensor, "... 4"], q_wxyz_2: Float[torch.Tensor, "... 4"], t: Float[torch.Tensor, "..."]
) -> Float[torch.Tensor, "... 4"]:
    """Spherically interpolate between unit quaternions along the shorter arc.

    Args:
        q_wxyz_1 (torch.Tensor): Unit quaternions with shape [..., 4] in wxyz format, returned at `t = 0`.
        q_wxyz_2 (torch.Tensor): Unit quaternions with shape [..., 4] in wxyz format, reached at `t = 1` (up to sign).
        t (torch.Tensor): Interpolation fractions with shape [...], broadcast against the quaternion batch dimensions.

    Returns:
        torch.Tensor: Interpolated quaternions with shape [..., 4] in wxyz format with non-negative real part.

    Example:
        >>> q_z90 = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068])
        >>> quaternion_slerp(torch.tensor([1.0, 0.0, 0.0, 0.0]), q_z90, torch.tensor(0.5))
        tensor([0.9239, 0.0000, 0.0000, 0.3827])
    """
    t = torch.as_tensor(t, dtype=q_wxyz_1.dtype, device=q_wxyz_1.device)
    shape = torch.broadcast_shapes(q_wxyz_1.shape[:-1], q_wxyz_2.shape[:-1], t.shape)
    q_wxyz_1, q_wxyz_2 = q_wxyz_1.expand(shape + (4,)), q_wxyz_2.expand(shape + (4,))
    delta = quaternion_to_axis_angle(quaternion_multiply(quaternion_invert(q_wxyz_1), q_wxyz_2))
    return quaternion_multiply(q_wxyz_1, axis_angle_to_quaternion(t.expand(shape)[..., None] * delta))


class TransformPoints(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pts: Float[torch.Tensor, "... n 3"], tf_mat: Float[torch.Tensor, "... 4 4"]):
        n_pts = pts.shape[-2]
        wp.init()
        pts_wp = wp.from_torch(pts.contiguous().view(-1, 3), dtype=wp.vec3, requires_grad=pts.requires_grad)
        tf_mat_wp = wp.from_torch(
            tf_mat.contiguous().view(-1, 4, 4), dtype=wp.mat44, requires_grad=tf_mat.requires_grad
        )
        # new_pts_wp = wp.zeros_like(pts_wp)  # NOTE somehow this will cause a bug in multi-processing
        new_pts_wp = wp.from_torch(
            torch.empty_like(pts).view(-1, 3), dtype=wp.vec3, requires_grad=pts.requires_grad or tf_mat.requires_grad
        )  # note do not use `torch.empty_like(pts.view(-1, 3))`, pts may not be contiguous
        wp.launch(
            kernel=transform_points_kernel,
            dim=(pts_wp.shape[0],),
            inputs=[pts_wp, tf_mat_wp, n_pts],
            outputs=[new_pts_wp],
            device=pts_wp.device,
        )
        if pts.requires_grad or tf_mat.requires_grad:
            ctx.pts_wp = pts_wp
            ctx.tf_mat_wp = tf_mat_wp
            ctx.new_pts_wp = new_pts_wp
            ctx.n_pts = n_pts
        return wp.to_torch(new_pts_wp).view(pts.shape)

    @staticmethod
    def backward(  # type: ignore
        ctx, new_pts_grad: Float[torch.Tensor, "... n 3"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... n 3"]], Optional[Float[torch.Tensor, "... 4 4"]]]:
        ctx.new_pts_wp.grad = wp.from_torch(new_pts_grad.contiguous().view(-1, 3), dtype=wp.vec3)
        wp.launch(
            kernel=transform_points_kernel,
            dim=(ctx.pts_wp.shape[0],),
            inputs=[ctx.pts_wp, ctx.tf_mat_wp, ctx.n_pts],
            outputs=[ctx.new_pts_wp],
            adj_inputs=[ctx.pts_wp.grad, ctx.tf_mat_wp.grad, ctx.n_pts],
            adj_outputs=[ctx.new_pts_wp.grad],
            adjoint=True,
            device=ctx.pts_wp.device,
        )
        pts_grad = wp.to_torch(ctx.pts_wp.grad).view(new_pts_grad.shape) if ctx.pts_wp.requires_grad else None
        tf_mat_grad = (
            wp.to_torch(ctx.tf_mat_wp.grad.contiguous()).view(new_pts_grad.shape[:-2] + (4, 4))
            if ctx.tf_mat_wp.requires_grad
            else None
        )
        return pts_grad, tf_mat_grad


def transform_points(
    pts: Float[torch.Tensor, "... n 3"], tf_mat: Float[torch.Tensor, "... 4 4"]
) -> Float[torch.Tensor, "... n 3"]:
    """Apply a transformation matrix on a set of 3D points.

    Args:
        pts (torch.Tensor): 3D points, could be [... n 3]
        tf_mat (torch.Tensor): Transformation matrix, could be [... 4 4]

    Returns:
        Transformed pts in shape of [... n 3]

    Examples:
        >>> pts = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> tf_mat = torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 2.0], [1.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 1.0]])
        >>> transform_points(pts, tf_mat)
        tensor([[3., 5., 4.],
                [6., 8., 7.]])

    Note:
        The dimension number of `pts` and `tf_mat` should be the same. The batch dimensions (...) are broadcast (https://pytorch.org/docs/stable/notes/broadcasting.html; thus must be broadcastable). We don't adopt the shapes [... 3] and [... 4 4] because there is no real broadcasted vector-matrix multiplication in pytorch. [... 3] and [... 4 4] will be converted to [... 1 3] and [... 4 4] and apply a broadcasted matrix-matrix multiplication.
    """
    if pts.device != tf_mat.device:
        raise ValueError(f"pts and tf_mat must be on the same device, got {pts.device} and {tf_mat.device}")

    broadcasted_shape = torch.broadcast_shapes(pts.shape[:-2], tf_mat.shape[:-2])
    return TransformPoints.apply(
        pts.expand(broadcasted_shape + pts.shape[-2:]), tf_mat.expand(broadcasted_shape + tf_mat.shape[-2:])
    )  # type: ignore


class RotatePoints(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pts: Float[torch.Tensor, "... n 3"], rot_mat: Float[torch.Tensor, "... 3 3"]):
        n_pts = pts.shape[-2]
        wp.init()
        pts_wp = wp.from_torch(pts.contiguous().view(-1, 3), dtype=wp.vec3, requires_grad=pts.requires_grad)
        rot_mat_wp = wp.from_torch(
            rot_mat.contiguous().view(-1, 3, 3), dtype=wp.mat33, requires_grad=rot_mat.requires_grad
        )
        new_pts_wp = wp.from_torch(
            torch.empty_like(pts).view(-1, 3), dtype=wp.vec3, requires_grad=pts.requires_grad or rot_mat.requires_grad
        )
        wp.launch(
            kernel=rotate_points_kernel,
            dim=(pts_wp.shape[0],),
            inputs=[pts_wp, rot_mat_wp, n_pts],
            outputs=[new_pts_wp],
            device=pts_wp.device,
        )
        if pts.requires_grad or rot_mat.requires_grad:
            ctx.pts_wp = pts_wp
            ctx.rot_mat_wp = rot_mat_wp
            ctx.new_pts_wp = new_pts_wp
            ctx.n_pts = n_pts
        return wp.to_torch(new_pts_wp).view(pts.shape)

    @staticmethod
    def backward(  # type: ignore
        ctx, new_pts_grad: Float[torch.Tensor, "... n 3"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... n 3"]], Optional[Float[torch.Tensor, "... 3 3"]]]:
        wp.init()
        ctx.new_pts_wp.grad = wp.from_torch(new_pts_grad.contiguous().view(-1, 3), dtype=wp.vec3)
        wp.launch(
            kernel=rotate_points_kernel,
            dim=(ctx.pts_wp.shape[0],),
            inputs=[ctx.pts_wp, ctx.rot_mat_wp, ctx.n_pts],
            outputs=[ctx.new_pts_wp],
            adj_inputs=[ctx.pts_wp.grad, ctx.rot_mat_wp.grad, ctx.n_pts],
            adj_outputs=[ctx.new_pts_wp.grad],
            adjoint=True,
            device=ctx.pts_wp.device,
        )
        pts_grad = wp.to_torch(ctx.pts_wp.grad).view(new_pts_grad.shape) if ctx.pts_wp.requires_grad else None
        rot_mat_grad = (
            wp.to_torch(ctx.rot_mat_wp.grad).view(new_pts_grad.shape[:-2] + (3, 3))
            if ctx.rot_mat_wp.requires_grad
            else None
        )
        return pts_grad, rot_mat_grad


def rotate_points(
    pts: Float[torch.Tensor, "... n 3"], rot_mat: Float[torch.Tensor, "... 3 3"]
) -> Float[torch.Tensor, "... n 3"]:
    """Apply a rotation matrix on a set of 3D points.

    Args:
        pts (torch.Tensor): 3D points in shape [... n 3].
        rot_mat (torch.Tensor): Rotation matrix in shape [... 3 3].

    Returns:
        torch.Tensor: Rotated points in shape [... n 3].

    Example:
        >>> R_z90 = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        >>> rotate_points(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), R_z90)
        tensor([[ 0.,  1.,  0.],
                [-1.,  0.,  0.]])
    """
    return RotatePoints.apply(pts, rot_mat)  # type: ignore


def project_points(
    pts: Float[torch.Tensor, "... n 3"], intrinsic: Float[torch.Tensor, "... 3 3"]
) -> Float[torch.Tensor, "... n 2"]:
    """Project camera-frame 3D points to pixels with pinhole intrinsics (OpenCV axes); depth is clamped at 1e-3 m.

    Args:
        pts (torch.Tensor): Camera-frame 3D points in shape [... n 3].
        intrinsic (torch.Tensor): Intrinsic matrix in shape [... 3 3]; batch dimensions broadcast with `pts`.

    Returns:
        torch.Tensor: Pixel coordinates (u, v) in shape [... n 2].

    Example:
        >>> K = torch.tensor([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
        >>> project_points(torch.tensor([[0.0, 0.0, 1.0], [0.1, -0.2, 2.0]]), K)
        tensor([[32., 24.],
                [37., 14.]])
    """
    xy = pts[..., :2] / pts[..., 2:].clamp(min=1e-3)
    return xy @ intrinsic[..., :2, :2].transpose(-1, -2) + intrinsic[..., None, :2, 2]


def intr_to_proj_mat(
    intr: Float[torch.Tensor, "3 3"], H: int, W: int, near: float = 0.001, far: float = 10.0
) -> Float[torch.Tensor, "4 4"]:
    """Convert a 3x3 camera intrinsic matrix to a 4x4 OpenGL projection matrix.

    Example:
        >>> intr = torch.tensor([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
        >>> intr_to_proj_mat(intr, H=48, W=64)[0, 0].item()
        3.125
    """
    fu, fv, cu, cv = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    return torch.tensor(
        [
            [2 * fu / W, 0, -2 * cu / W + 1, 0],
            [0, 2 * fv / H, 2 * cv / H - 1, 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0],
        ],
        dtype=torch.float32,
        device=intr.device,
    )


def rot_tl_to_tf_mat(
    rot_mat: Optional[Float[torch.Tensor, "... 3 3"]] = None,
    tl: Optional[Float[torch.Tensor, "... 3"]] = None,
) -> Float[torch.Tensor, "... 4 4"]:
    """Build a 4x4 transform from a rotation and/or translation (identity/zero defaults; at least one required).

    Examples:
        >>> rot_mat = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float32)
        >>> tl = torch.tensor([1, 2, 3], dtype=torch.float32)
        >>> rot_tl_to_tf_mat(rot_mat, tl)
        tensor([[0., 1., 0., 1.],
                [0., 0., 1., 2.],
                [1., 0., 0., 3.],
                [0., 0., 0., 1.]])
        >>> rot_tl_to_tf_mat(tl=tl)
        tensor([[1., 0., 0., 1.],
                [0., 1., 0., 2.],
                [0., 0., 1., 3.],
                [0., 0., 0., 1.]])
        >>> rot_tl_to_tf_mat(rot_mat=rot_mat)
        tensor([[0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [1., 0., 0., 0.],
                [0., 0., 0., 1.]])
    """
    if rot_mat is None and tl is None:
        raise ValueError("Either rot_mat or tl should be provided.")
    if tl is None:
        tl = torch.zeros(rot_mat.shape[:-2] + (3,), device=rot_mat.device, dtype=rot_mat.dtype)
    if rot_mat is None:
        rot_mat = torch.eye(3, device=tl.device, dtype=tl.dtype).expand(tl.shape[:-1] + (3, 3))
    if rot_mat.device != tl.device:
        raise ValueError(f"rot_mat and tl must be on the same device, got {rot_mat.device} and {tl.device}")

    b_shape = torch.broadcast_shapes(rot_mat.shape[:-2], tl.shape[:-1])
    rot_mat = rot_mat.expand(b_shape + (3, 3))
    tl = tl.expand(b_shape + (3,))

    tf_mat_3x4 = torch.cat([rot_mat, tl.unsqueeze(-1)], dim=-1)
    last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=tl.device, dtype=tl.dtype)
    last_row = last_row.expand(b_shape + (1, 4))
    return torch.cat([tf_mat_3x4, last_row], dim=-2)


class InverseTfMat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tf_mat: Float[torch.Tensor, "... 4 4"]) -> Float[torch.Tensor, "... 4 4"]:
        wp.init()
        tf_mat_wp = wp.from_torch(
            tf_mat.contiguous().view(-1, 4, 4), dtype=wp.mat44, requires_grad=tf_mat.requires_grad
        )
        tf_mat_inv_wp = wp.from_torch(
            torch.empty_like(tf_mat).view(-1, 4, 4), dtype=wp.mat44, requires_grad=tf_mat.requires_grad
        )
        wp.launch(
            kernel=inverse_tf_mat_kernel,
            dim=(tf_mat_wp.shape[0],),
            inputs=[tf_mat_wp],
            outputs=[tf_mat_inv_wp],
            device=tf_mat_wp.device,
        )
        if tf_mat.requires_grad:
            ctx.tf_mat_wp = tf_mat_wp
            ctx.tf_mat_inv_wp = tf_mat_inv_wp
        return wp.to_torch(tf_mat_inv_wp).view(tf_mat.shape)

    @staticmethod
    def backward(  # type: ignore
        ctx, tf_mat_inv_grad: Float[torch.Tensor, "... 4 4"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 4 4"]]]:
        wp.init()
        if ctx.tf_mat_wp.grad is not None:
            ctx.tf_mat_wp.grad.zero_()
        ctx.tf_mat_inv_wp.grad = wp.from_torch(tf_mat_inv_grad.contiguous().view(-1, 4, 4), dtype=wp.mat44)
        wp.launch(
            kernel=inverse_tf_mat_kernel,
            dim=(ctx.tf_mat_wp.shape[0],),
            inputs=[ctx.tf_mat_wp],
            outputs=[ctx.tf_mat_inv_wp],
            adj_inputs=[ctx.tf_mat_wp.grad],
            adj_outputs=[ctx.tf_mat_inv_wp.grad],
            adjoint=True,
            device=ctx.tf_mat_wp.device,
        )
        return (wp.to_torch(cast(wp.array, ctx.tf_mat_wp.grad)).view(tf_mat_inv_grad.shape),)


def inverse_tf_mat(tf_mat: Float[torch.Tensor, "... 4 4"]) -> Float[torch.Tensor, "... 4 4"]:
    """Invert a batch of 4x4 rigid transformation matrices.

    Args:
        tf_mat (torch.Tensor): Transformation matrices with shape [..., 4, 4].

    Returns:
        torch.Tensor: Inverted transformation matrices with shape [..., 4, 4].

    Example:
        >>> tf_mat = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 2], [1, 0, 0, 3], [0, 0, 0, 1]], dtype=torch.float32)
        >>> torch.allclose(inverse_tf_mat(tf_mat) @ tf_mat, torch.eye(4), atol=1e-5)
        True
    """
    return InverseTfMat.apply(tf_mat)  # type: ignore


def pose7_to_tf_mat(pose7: Float[torch.Tensor, "... 7"]) -> Float[torch.Tensor, "... 4 4"]:
    """Convert translation-first, scalar-first poses `[x, y, z, qw, qx, qy, qz]` to matrices.

    Args:
        pose7 (torch.Tensor): Poses in shape [..., 7].

    Returns:
        torch.Tensor: Transformation matrices in shape [..., 4, 4].

    Example:
        >>> pose7 = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]])
        >>> tf_mat = pose7_to_tf_mat(pose7)
        >>> tf_mat.shape, torch.allclose(tf_mat[0, :3, 3], torch.tensor([1.0, 2.0, 3.0]))
        (torch.Size([1, 4, 4]), True)
    """
    return rot_tl_to_tf_mat(quaternion_to_matrix(pose7[..., 3:]), pose7[..., :3])


def tf_mat_to_pose7(tf_mat: Float[torch.Tensor, "... 4 4"]) -> Float[torch.Tensor, "... 7"]:
    """Convert matrices to translation-first, scalar-first poses `[x, y, z, qw, qx, qy, qz]`.

    Args:
        tf_mat (torch.Tensor): Transformation matrices in shape [..., 4, 4].

    Returns:
        torch.Tensor: Poses in shape [..., 7].

    Example:
        >>> pose7 = torch.tensor([[0.1, 0.2, 0.3, 0.0, 1.0, 0.0, 0.0]])
        >>> back = tf_mat_to_pose7(pose7_to_tf_mat(pose7))
        >>> torch.allclose(back.abs(), pose7.abs(), atol=1e-6)
        True
    """
    return torch.cat([tf_mat[..., :3, 3], matrix_to_quaternion(tf_mat[..., :3, :3])], dim=-1)
