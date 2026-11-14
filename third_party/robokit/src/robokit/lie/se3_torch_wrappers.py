"""Torch autograd wrappers for SE(3) warp kernels."""

from typing import Optional, Tuple, cast

import torch
import warp as wp
from jaxtyping import Float

from robokit.lie.se3_kernels import (
    se3_adjoint_kernel,
    se3_apply_kernel,
    se3_compose_kernel,
    se3_exp_map_kernel,
    se3_exp_map_to_matrix_kernel,
    se3_inverse_kernel,
    se3_jlog_kernel,
    se3_log_map_kernel,
    se3_to_matrix_kernel,
)
from robokit.lie.so3_torch_wrappers import get_epsilon
from robokit.utils.warp_utils import wp_mat66, wp_vec6, wp_vec7


class SE3Compose(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        xyz_wxyz1: Float[torch.Tensor, "... 7"],
        xyz_wxyz2: Float[torch.Tensor, "... 7"],
    ) -> Float[torch.Tensor, "... 7"]:
        wp.init()
        xyz_wxyz1_wp = wp.from_torch(
            xyz_wxyz1.contiguous().view(-1, 7),
            dtype=wp_vec7,
            requires_grad=xyz_wxyz1.requires_grad,
        )
        xyz_wxyz2_wp = wp.from_torch(
            xyz_wxyz2.contiguous().view(-1, 7),
            dtype=wp_vec7,
            requires_grad=xyz_wxyz2.requires_grad,
        )
        result_wp = wp.from_torch(
            torch.empty_like(xyz_wxyz1).view(-1, 7),
            dtype=wp_vec7,
            requires_grad=xyz_wxyz1.requires_grad or xyz_wxyz2.requires_grad,
        )
        wp.launch(
            kernel=se3_compose_kernel,
            dim=(xyz_wxyz1_wp.shape[0],),
            inputs=[xyz_wxyz1_wp, xyz_wxyz2_wp, result_wp],
            device=xyz_wxyz1_wp.device,
        )
        if xyz_wxyz1.requires_grad or xyz_wxyz2.requires_grad:
            ctx.xyz_wxyz1_wp = xyz_wxyz1_wp
            ctx.xyz_wxyz2_wp = xyz_wxyz2_wp
            ctx.result_wp = result_wp
            ctx.input_shape = xyz_wxyz1.shape
        return wp.to_torch(result_wp).view(xyz_wxyz1.shape)

    @staticmethod
    def backward(
        ctx, xyz_wxyz_grad: Float[torch.Tensor, "... 7"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 7"]], Optional[Float[torch.Tensor, "... 7"]]]:
        wp.init()
        if ctx.xyz_wxyz1_wp.grad is not None:
            ctx.xyz_wxyz1_wp.grad.zero_()
        if ctx.xyz_wxyz2_wp.grad is not None:
            ctx.xyz_wxyz2_wp.grad.zero_()
        ctx.result_wp.grad = wp.from_torch(xyz_wxyz_grad.contiguous().view(-1, 7), dtype=wp_vec7)
        wp.launch(
            kernel=se3_compose_kernel,
            dim=(ctx.xyz_wxyz1_wp.shape[0],),
            inputs=[ctx.xyz_wxyz1_wp, ctx.xyz_wxyz2_wp, ctx.result_wp],
            adj_inputs=[ctx.xyz_wxyz1_wp.grad, ctx.xyz_wxyz2_wp.grad, ctx.result_wp.grad],
            adjoint=True,
            device=ctx.xyz_wxyz1_wp.device,
        )
        xyz_wxyz1_grad = (
            wp.to_torch(cast(wp.array, ctx.xyz_wxyz1_wp.grad)).view(ctx.input_shape)
            if ctx.xyz_wxyz1_wp.requires_grad
            else None
        )
        xyz_wxyz2_grad = (
            wp.to_torch(cast(wp.array, ctx.xyz_wxyz2_wp.grad)).view(ctx.input_shape)
            if ctx.xyz_wxyz2_wp.requires_grad
            else None
        )
        return xyz_wxyz1_grad, xyz_wxyz2_grad


SE3Multiply = SE3Compose


class SE3Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz_wxyz: Float[torch.Tensor, "... 7"]) -> Float[torch.Tensor, "... 6 6"]:
        wp.init()
        xyz_wxyz_wp = wp.from_torch(
            xyz_wxyz.contiguous().view(-1, 7), dtype=wp_vec7, requires_grad=xyz_wxyz.requires_grad
        )
        adjoint_output = torch.zeros(
            xyz_wxyz.shape[:-1] + (6, 6),
            dtype=xyz_wxyz.dtype,
            device=xyz_wxyz.device,
            requires_grad=xyz_wxyz.requires_grad,
        )
        adjoint_output_wp = wp.from_torch(adjoint_output.view(-1, 6, 6), dtype=wp_mat66)
        wp.launch(
            kernel=se3_adjoint_kernel,
            dim=(xyz_wxyz_wp.shape[0],),
            inputs=[xyz_wxyz_wp, adjoint_output_wp],
            device=xyz_wxyz_wp.device,
        )
        if xyz_wxyz.requires_grad:
            ctx.xyz_wxyz_wp = xyz_wxyz_wp
            ctx.adjoint_output_wp = adjoint_output_wp
            ctx.input_shape = xyz_wxyz.shape
        return adjoint_output

    @staticmethod
    def backward(ctx, adjoint_grad: Float[torch.Tensor, "... 6 6"]) -> Optional[Float[torch.Tensor, "... 7"]]:
        wp.init()
        ctx.xyz_wxyz_wp.grad.zero_()
        ctx.adjoint_output_wp.grad = wp.from_torch(adjoint_grad.contiguous().view(-1, 6, 6), dtype=wp_mat66)
        wp.launch(
            kernel=se3_adjoint_kernel,
            dim=(ctx.xyz_wxyz_wp.shape[0],),
            inputs=[ctx.xyz_wxyz_wp, ctx.adjoint_output_wp],
            adj_inputs=[ctx.xyz_wxyz_wp.grad, ctx.adjoint_output_wp.grad],
            adjoint=True,
            device=ctx.xyz_wxyz_wp.device,
        )
        if ctx.xyz_wxyz_wp.requires_grad:
            return wp.to_torch(ctx.xyz_wxyz_wp.grad).view(ctx.input_shape)
        return None


class SE3Jlog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz_wxyz: Float[torch.Tensor, "... 7"]) -> Float[torch.Tensor, "... 6 6"]:
        wp.init()
        eps = get_epsilon(xyz_wxyz.dtype)
        ctx.save_for_backward(xyz_wxyz)
        ctx.eps = eps
        xyz_wxyz_wp = wp.from_torch(
            xyz_wxyz.contiguous().view(-1, 7), dtype=wp_vec7, requires_grad=xyz_wxyz.requires_grad
        )
        jlog_output = torch.zeros(
            xyz_wxyz.shape[:-1] + (6, 6),
            dtype=xyz_wxyz.dtype,
            device=xyz_wxyz.device,
            requires_grad=xyz_wxyz.requires_grad,
        )
        jlog_output_wp = wp.from_torch(jlog_output.view(-1, 6, 6), dtype=wp_mat66)
        wp.launch(
            kernel=se3_jlog_kernel,
            dim=(xyz_wxyz_wp.shape[0],),
            inputs=[xyz_wxyz_wp, eps, jlog_output_wp],
            device=xyz_wxyz_wp.device,
        )
        return jlog_output

    @staticmethod
    def backward(ctx, grad_output):
        (xyz_wxyz,) = ctx.saved_tensors
        eps = ctx.eps
        grad_xyz_wxyz = torch.zeros_like(xyz_wxyz)
        xyz_wxyz_wp = wp.from_torch(
            xyz_wxyz.contiguous().view(-1, 7),
            dtype=wp_vec7,
            grad=wp.from_torch(grad_xyz_wxyz.contiguous().view(-1, 7), dtype=wp_vec7),
        )
        jlog_output = torch.zeros(xyz_wxyz.shape[:-1] + (6, 6), dtype=xyz_wxyz.dtype, device=xyz_wxyz.device)
        jlog_output_wp = wp.from_torch(
            jlog_output.view(-1, 6, 6),
            dtype=wp_mat66,
            grad=wp.from_torch(grad_output.contiguous().view(-1, 6, 6), dtype=wp_mat66),
        )
        wp.launch(
            kernel=se3_jlog_kernel,
            dim=(xyz_wxyz_wp.shape[0],),
            inputs=[xyz_wxyz_wp, eps, jlog_output_wp],
            adj_inputs=[None, eps, None],
            adjoint=True,
            device=xyz_wxyz_wp.device,
        )
        return grad_xyz_wxyz


class SE3ExpMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_transforms: Float[torch.Tensor, "... 6"], eps: float = 1e-4) -> Float[torch.Tensor, "... 7"]:
        wp.init()
        log_transform_wp = wp.from_torch(
            log_transforms.contiguous().view(-1, 6), dtype=wp_vec6, requires_grad=log_transforms.requires_grad
        )
        xyz_wxyz_output_wp = wp.from_torch(
            torch.empty(
                log_transforms.shape[:-1] + (7,), dtype=log_transforms.dtype, device=log_transforms.device
            ).view(-1, 7),
            dtype=wp_vec7,
            requires_grad=log_transforms.requires_grad,
        )
        wp.launch(
            kernel=se3_exp_map_kernel,
            dim=(log_transform_wp.shape[0],),
            inputs=[log_transform_wp, eps],
            outputs=[xyz_wxyz_output_wp],
            device=log_transform_wp.device,
        )
        if log_transforms.requires_grad:
            ctx.log_transform_wp = log_transform_wp
            ctx.xyz_wxyz_output_wp = xyz_wxyz_output_wp
            ctx.eps = eps
        return wp.to_torch(xyz_wxyz_output_wp).view(log_transforms.shape[:-1] + (7,))

    @staticmethod
    def backward(
        ctx, xyz_wxyz_grad: Float[torch.Tensor, "... 7"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 6"]], None]:
        wp.init()
        if ctx.log_transform_wp.grad is not None:
            ctx.log_transform_wp.grad.zero_()
        ctx.xyz_wxyz_output_wp.grad = wp.from_torch(xyz_wxyz_grad.contiguous().view(-1, 7), dtype=wp_vec7)
        wp.launch(
            kernel=se3_exp_map_kernel,
            dim=(ctx.log_transform_wp.shape[0],),
            inputs=[ctx.log_transform_wp, ctx.eps],
            outputs=[ctx.xyz_wxyz_output_wp],
            adj_inputs=[ctx.log_transform_wp.grad, None],
            adj_outputs=[ctx.xyz_wxyz_output_wp.grad],
            adjoint=True,
            device=ctx.log_transform_wp.device,
        )
        return wp.to_torch(cast(wp.array, ctx.log_transform_wp.grad)).view(xyz_wxyz_grad.shape[:-1] + (6,)), None


class SE3ExpMapToMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_transforms: Float[torch.Tensor, "... 6"], eps: float = 1e-4) -> Float[torch.Tensor, "... 4 4"]:
        wp.init()
        log_transform_wp = wp.from_torch(
            log_transforms.contiguous().view(-1, 6), dtype=wp_vec6, requires_grad=log_transforms.requires_grad
        )
        matrix_output_wp = wp.from_torch(
            torch.empty(
                log_transforms.shape[:-1] + (4, 4), dtype=log_transforms.dtype, device=log_transforms.device
            ).view(-1, 4, 4),
            dtype=wp.mat44,
            requires_grad=log_transforms.requires_grad,
        )
        wp.launch(
            kernel=se3_exp_map_to_matrix_kernel,
            dim=(log_transform_wp.shape[0],),
            inputs=[log_transform_wp, eps],
            outputs=[matrix_output_wp],
            device=log_transform_wp.device,
        )
        if log_transforms.requires_grad:
            ctx.log_transform_wp = log_transform_wp
            ctx.matrix_output_wp = matrix_output_wp
            ctx.eps = eps
        return wp.to_torch(matrix_output_wp).view(log_transforms.shape[:-1] + (4, 4))

    @staticmethod
    def backward(
        ctx, matrix_grad: Float[torch.Tensor, "... 4 4"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 6"]], None]:
        wp.init()
        if ctx.log_transform_wp.grad is not None:
            ctx.log_transform_wp.grad.zero_()
        ctx.matrix_output_wp.grad = wp.from_torch(matrix_grad.contiguous().view(-1, 4, 4), dtype=wp.mat44)
        wp.launch(
            kernel=se3_exp_map_to_matrix_kernel,
            dim=(ctx.log_transform_wp.shape[0],),
            inputs=[ctx.log_transform_wp, ctx.eps],
            outputs=[ctx.matrix_output_wp],
            adj_inputs=[ctx.log_transform_wp.grad, None],
            adj_outputs=[ctx.matrix_output_wp.grad],
            adjoint=True,
            device=ctx.log_transform_wp.device,
        )
        return wp.to_torch(cast(wp.array, ctx.log_transform_wp.grad)).view(matrix_grad.shape[:-2] + (6,)), None


class SE3LogMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz_wxyz: Float[torch.Tensor, "... 7"], eps: float = 1e-4) -> Float[torch.Tensor, "... 6"]:
        wp.init()
        xyz_wxyz_wp = wp.from_torch(
            xyz_wxyz.contiguous().view(-1, 7), dtype=wp_vec7, requires_grad=xyz_wxyz.requires_grad
        )
        log_transform_wp = wp.from_torch(
            torch.empty(xyz_wxyz.shape[:-1] + (6,), dtype=xyz_wxyz.dtype, device=xyz_wxyz.device).view(-1, 6),
            dtype=wp_vec6,
            requires_grad=xyz_wxyz.requires_grad,
        )
        wp.launch(
            kernel=se3_log_map_kernel,
            dim=(xyz_wxyz_wp.shape[0],),
            inputs=[xyz_wxyz_wp, eps],
            outputs=[log_transform_wp],
            device=xyz_wxyz_wp.device,
        )
        if xyz_wxyz.requires_grad:
            ctx.xyz_wxyz_wp = xyz_wxyz_wp
            ctx.log_transform_wp = log_transform_wp
            ctx.eps = eps
            ctx.input_shape = xyz_wxyz.shape
        return wp.to_torch(log_transform_wp).view(xyz_wxyz.shape[:-1] + (6,))

    @staticmethod
    def backward(
        ctx, log_transforms_grad: Float[torch.Tensor, "... 6"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 7"]], None]:
        wp.init()
        if ctx.xyz_wxyz_wp.grad is not None:
            ctx.xyz_wxyz_wp.grad.zero_()
        ctx.log_transform_wp.grad = wp.from_torch(log_transforms_grad.contiguous().view(-1, 6), dtype=wp_vec6)
        wp.launch(
            kernel=se3_log_map_kernel,
            dim=(ctx.xyz_wxyz_wp.shape[0],),
            inputs=[ctx.xyz_wxyz_wp, ctx.eps],
            outputs=[ctx.log_transform_wp],
            adj_inputs=[ctx.xyz_wxyz_wp.grad, ctx.eps],
            adj_outputs=[ctx.log_transform_wp.grad],
            adjoint=True,
            device=ctx.xyz_wxyz_wp.device,
        )
        return wp.to_torch(cast(wp.array, ctx.xyz_wxyz_wp.grad)).view(ctx.input_shape), None


class SE3Inverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz_wxyz: Float[torch.Tensor, "... 7"]) -> Float[torch.Tensor, "... 7"]:
        wp.init()
        xyz_wxyz_wp = wp.from_torch(
            xyz_wxyz.contiguous().view(-1, 7), dtype=wp_vec7, requires_grad=xyz_wxyz.requires_grad
        )
        result_wp = wp.from_torch(
            torch.empty_like(xyz_wxyz).view(-1, 7), dtype=wp_vec7, requires_grad=xyz_wxyz.requires_grad
        )
        wp.launch(
            kernel=se3_inverse_kernel,
            dim=(xyz_wxyz_wp.shape[0],),
            inputs=[xyz_wxyz_wp, result_wp],
            device=xyz_wxyz_wp.device,
        )
        if xyz_wxyz.requires_grad:
            ctx.xyz_wxyz_wp = xyz_wxyz_wp
            ctx.result_wp = result_wp
            ctx.input_shape = xyz_wxyz.shape
        return wp.to_torch(result_wp).view(xyz_wxyz.shape)

    @staticmethod
    def backward(ctx, result_grad: Float[torch.Tensor, "... 7"]) -> Optional[Float[torch.Tensor, "... 7"]]:
        wp.init()
        if ctx.xyz_wxyz_wp.grad is not None:
            ctx.xyz_wxyz_wp.grad.zero_()
        ctx.result_wp.grad = wp.from_torch(result_grad.contiguous().view(-1, 7), dtype=wp_vec7)
        wp.launch(
            kernel=se3_inverse_kernel,
            dim=(ctx.xyz_wxyz_wp.shape[0],),
            inputs=[ctx.xyz_wxyz_wp, ctx.result_wp],
            adj_inputs=[ctx.xyz_wxyz_wp.grad, ctx.result_wp.grad],
            adjoint=True,
            device=ctx.xyz_wxyz_wp.device,
        )
        if ctx.xyz_wxyz_wp.requires_grad:
            return wp.to_torch(cast(wp.array, ctx.xyz_wxyz_wp.grad)).view(ctx.input_shape)
        return None


class SE3Apply(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, xyz_wxyz: Float[torch.Tensor, "... 7"], points: Float[torch.Tensor, "... 3"]
    ) -> Float[torch.Tensor, "... 3"]:
        wp.init()
        xyz_wxyz_wp = wp.from_torch(
            xyz_wxyz.contiguous().view(-1, 7), dtype=wp_vec7, requires_grad=xyz_wxyz.requires_grad
        )
        points_wp = wp.from_torch(points.contiguous().view(-1, 3), dtype=wp.vec3, requires_grad=points.requires_grad)
        result_wp = wp.from_torch(
            torch.empty_like(points).view(-1, 3),
            dtype=wp.vec3,
            requires_grad=xyz_wxyz.requires_grad or points.requires_grad,
        )
        wp.launch(
            kernel=se3_apply_kernel,
            dim=(points_wp.shape[0],),
            inputs=[xyz_wxyz_wp, points_wp, result_wp],
            device=xyz_wxyz_wp.device,
        )
        if xyz_wxyz.requires_grad or points.requires_grad:
            ctx.xyz_wxyz_wp = xyz_wxyz_wp
            ctx.points_wp = points_wp
            ctx.result_wp = result_wp
            ctx.input_shape_xyz_wxyz = xyz_wxyz.shape
            ctx.input_shape_points = points.shape
        return wp.to_torch(result_wp).view(points.shape)

    @staticmethod
    def backward(
        ctx, result_grad: Float[torch.Tensor, "... 3"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... 7"]], Optional[Float[torch.Tensor, "... 3"]]]:
        wp.init()
        if ctx.xyz_wxyz_wp.grad is not None:
            ctx.xyz_wxyz_wp.grad.zero_()
        if ctx.points_wp.grad is not None:
            ctx.points_wp.grad.zero_()
        ctx.result_wp.grad = wp.from_torch(result_grad.contiguous().view(-1, 3), dtype=wp.vec3)
        wp.launch(
            kernel=se3_apply_kernel,
            dim=(ctx.points_wp.shape[0],),
            inputs=[ctx.xyz_wxyz_wp, ctx.points_wp, ctx.result_wp],
            adj_inputs=[ctx.xyz_wxyz_wp.grad, ctx.points_wp.grad, ctx.result_wp.grad],
            adjoint=True,
            device=ctx.xyz_wxyz_wp.device,
        )
        xyz_wxyz_grad = (
            wp.to_torch(cast(wp.array, ctx.xyz_wxyz_wp.grad)).view(ctx.input_shape_xyz_wxyz)
            if ctx.xyz_wxyz_wp.requires_grad
            else None
        )
        points_grad = (
            wp.to_torch(cast(wp.array, ctx.points_wp.grad)).view(ctx.input_shape_points)
            if ctx.points_wp.requires_grad
            else None
        )
        return xyz_wxyz_grad, points_grad


class SE3ToMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz_wxyz: Float[torch.Tensor, "... 7"]) -> Float[torch.Tensor, "... 4 4"]:
        wp.init()
        xyz_wxyz_wp = wp.from_torch(
            xyz_wxyz.contiguous().view(-1, 7), dtype=wp_vec7, requires_grad=xyz_wxyz.requires_grad
        )
        matrix = torch.empty(xyz_wxyz.shape[:-1] + (4, 4), dtype=xyz_wxyz.dtype, device=xyz_wxyz.device)
        matrix_wp = wp.from_torch(matrix.view(-1, 4, 4), dtype=wp.mat44, requires_grad=xyz_wxyz.requires_grad)
        wp.launch(
            kernel=se3_to_matrix_kernel,
            dim=(xyz_wxyz_wp.shape[0],),
            inputs=[xyz_wxyz_wp],
            outputs=[matrix_wp],
            device=xyz_wxyz_wp.device,
        )
        if xyz_wxyz.requires_grad:
            ctx.xyz_wxyz_wp = xyz_wxyz_wp
            ctx.matrix_wp = matrix_wp
            ctx.input_shape = xyz_wxyz.shape
        return matrix

    @staticmethod
    def backward(ctx, matrix_grad: Float[torch.Tensor, "... 4 4"]) -> Optional[Float[torch.Tensor, "... 7"]]:
        wp.init()
        if ctx.xyz_wxyz_wp.grad is not None:
            ctx.xyz_wxyz_wp.grad.zero_()
        ctx.matrix_wp.grad = wp.from_torch(matrix_grad.contiguous().view(-1, 4, 4), dtype=wp.mat44)
        wp.launch(
            kernel=se3_to_matrix_kernel,
            dim=(ctx.xyz_wxyz_wp.shape[0],),
            inputs=[ctx.xyz_wxyz_wp],
            outputs=[ctx.matrix_wp],
            adj_inputs=[ctx.xyz_wxyz_wp.grad],
            adj_outputs=[ctx.matrix_wp.grad],
            adjoint=True,
            device=ctx.xyz_wxyz_wp.device,
        )
        if ctx.xyz_wxyz_wp.requires_grad:
            return wp.to_torch(cast(wp.array, ctx.xyz_wxyz_wp.grad)).view(ctx.input_shape)
        return None
