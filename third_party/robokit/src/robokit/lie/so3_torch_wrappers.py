"""Torch autograd wrappers for SO(3) warp kernels."""

import torch
import warp as wp
from jaxtyping import Float

from robokit.lie.so3_kernels import so3_jlog_kernel


def get_epsilon(dtype: torch.dtype) -> float:
    if dtype == torch.float32:
        return 1e-4
    elif dtype == torch.float64:
        return 1e-8
    else:
        return 1e-4


class SO3Jlog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 3 3"]:
        wp.init()

        eps = get_epsilon(theta.dtype)

        original_shape = theta.shape
        theta_flat = theta.view(-1, 3).contiguous()

        theta_wp = wp.from_torch(theta_flat, dtype=wp.vec3, requires_grad=theta.requires_grad)

        jac_output = torch.empty(
            (theta_flat.shape[0], 3, 3),
            dtype=theta.dtype,
            device=theta.device,
            requires_grad=theta.requires_grad,
        )

        jac_output_wp = wp.from_torch(jac_output, dtype=wp.mat33)

        wp.launch(
            kernel=so3_jlog_kernel,
            dim=(theta_flat.shape[0],),
            inputs=[theta_wp, eps, jac_output_wp],
            device=theta_wp.device,
        )

        result_shape = original_shape[:-1] + (3, 3)
        return jac_output.view(result_shape)

    @staticmethod
    def backward(ctx, grad_output):
        return None
