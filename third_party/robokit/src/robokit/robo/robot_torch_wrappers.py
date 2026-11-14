# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
import math
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import warp as wp
from jaxtyping import Float
from torch.autograd import Function

from robokit.robo.robot_kernels import (
    compute_forward_kinematics_matrix_accum_kernel,
    compute_forward_kinematics_matrix_backward_kernel,
    compute_forward_kinematics_matrix_local_kernel,
    compute_forward_kinematics_matrix_sequential_kernel,
    transform_link_points_matrix_kernel,
)


if TYPE_CHECKING:
    from robokit.robo.robot_spec import RobotSpec
    from robokit.robo.robot_spec_tensors import RobotSpecTensors


class WarpForwardKinematicsMatrixAutodiff(Function):
    @staticmethod
    def forward(
        ctx,
        robot_spec: "RobotSpec",
        q: Float[torch.Tensor, "... num_actuated_joints"],
        T_world_base: Float[torch.Tensor, "... 4 4"],
    ):
        batch_shape = q.shape[:-1]
        num_instances = math.prod(batch_shape) if batch_shape else 1
        num_actuated = q.shape[-1]
        requires_grad = q.requires_grad or T_world_base.requires_grad
        device = q.device
        wp.init()  # device_from_torch needs the warp runtime; forward() may be the process's first warp call
        wp_device = wp.device_from_torch(device)

        q_wp = wp.from_torch(
            q.contiguous().view(num_instances, num_actuated),
            dtype=wp.float32,
            requires_grad=requires_grad,
        )
        T_world_base_wp = wp.from_torch(
            T_world_base.to(device=device, dtype=torch.float32).contiguous().view(num_instances, 4, 4),
            dtype=wp.mat44,
            requires_grad=requires_grad,
        )

        T_world_joint_wp = wp.empty(
            (num_instances, robot_spec.num_joints), dtype=wp.mat44, device=wp_device, requires_grad=requires_grad
        )
        T_world_link_wp = wp.empty(
            (num_instances, robot_spec.num_links), dtype=wp.mat44, device=wp_device, requires_grad=requires_grad
        )

        spec_t = robot_spec.get_tensors(str(wp_device))
        spec_tensors_wp: Tuple[wp.array, ...] = (
            spec_t.actuated_joint_indices,
            spec_t.mimic_actuated_joint_indices,
            spec_t.mimic_multipliers,
            spec_t.mimic_offsets,
            spec_t.joint_twists,
            spec_t.parent_joint_transforms_matrix,
            spec_t.topological_order_joint_indices,
            spec_t.parent_joint_indices,
            spec_t.link_parent_joint_indices,
        )

        wp.launch(
            kernel=compute_forward_kinematics_matrix_sequential_kernel,
            dim=[num_instances],
            inputs=[q_wp, T_world_base_wp, *spec_tensors_wp],
            outputs=[T_world_joint_wp, T_world_link_wp],
            device=wp_device,
        )

        if requires_grad:
            ctx.meta_info = (batch_shape, num_instances, num_actuated, robot_spec.num_joints, device)
            ctx.robot_spec = robot_spec
            ctx.joint_positions_wp = q_wp
            ctx.T_world_base_wp = T_world_base_wp
            ctx.T_world_joint_wp = T_world_joint_wp
            ctx.T_world_link_wp = T_world_link_wp
            ctx.spec_tensors_wp = spec_tensors_wp

        return wp.to_torch(T_world_link_wp).view(*batch_shape, robot_spec.num_links, 4, 4)

    @staticmethod
    def backward(
        ctx, link_poses_grad: Float[torch.Tensor, "... num_links 4 4"]
    ) -> Tuple[
        None,
        Optional[Float[torch.Tensor, "... num_actuated_joints"]],
        Optional[Float[torch.Tensor, "... 4 4"]],
    ]:
        batch_shape, num_instances, num_actuated, num_joints, device = ctx.meta_info

        ctx.T_world_link_wp.grad = wp.from_torch(
            link_poses_grad.contiguous().view(-1, ctx.robot_spec.num_links, 4, 4), dtype=wp.mat44, requires_grad=False
        )
        ctx.T_world_joint_wp.grad = wp.zeros_like(ctx.T_world_joint_wp)
        ctx.joint_positions_wp.grad = wp.zeros_like(ctx.joint_positions_wp)

        base_grad_input = None
        if ctx.needs_input_grad[2]:
            ctx.T_world_base_wp.grad = wp.zeros_like(ctx.T_world_base_wp)
            base_grad_input = ctx.T_world_base_wp.grad

        wp_device = ctx.joint_positions_wp.device
        spec_tensors: Tuple[wp.array, ...] = ctx.spec_tensors_wp

        wp.launch(
            kernel=compute_forward_kinematics_matrix_sequential_kernel,
            dim=[num_instances],
            inputs=[ctx.joint_positions_wp, ctx.T_world_base_wp, *spec_tensors],
            outputs=[ctx.T_world_joint_wp, ctx.T_world_link_wp],
            adj_inputs=[ctx.joint_positions_wp.grad, base_grad_input, *([None] * 9)],
            adj_outputs=[ctx.T_world_joint_wp.grad, ctx.T_world_link_wp.grad],
            adjoint=True,
            device=wp_device,
        )

        joint_values_grad = None
        if ctx.needs_input_grad[1]:
            joint_values_grad = wp.to_torch(ctx.joint_positions_wp.grad).view(*batch_shape, num_actuated)

        base_values_grad = None
        if ctx.needs_input_grad[2]:
            base_values_grad = wp.to_torch(ctx.T_world_base_wp.grad).view(*batch_shape, 4, 4)

        return None, joint_values_grad, base_values_grad


class WarpForwardKinematicsMatrix(Function):
    @staticmethod
    @lru_cache(maxsize=128)
    def _get_cached_fk(spec_t: "RobotSpecTensors", num_instances: int, record_cmd: bool):
        """Cache FK buffers by spec, batch size, and recording mode to bound memory across batches."""
        n_act = spec_t.spec.num_actuated_joints
        n_joints = spec_t.spec.num_joints
        n_links = spec_t.spec.num_links
        device = spec_t.device
        q_buf = torch.zeros((num_instances, n_act), dtype=torch.float32, device=device)
        base_buf = torch.zeros((num_instances, 4, 4), dtype=torch.float32, device=device)
        q_wp = wp.from_torch(q_buf, dtype=wp.float32, requires_grad=False)
        base_wp = wp.from_torch(base_buf, dtype=wp.mat44, requires_grad=False)
        X_local_wp = wp.zeros((num_instances, n_joints), dtype=wp.mat44, device=device, requires_grad=False)
        T_world_joint_wp = wp.zeros((num_instances, n_joints), dtype=wp.mat44, device=device, requires_grad=False)
        T_world_link_wp = wp.zeros((num_instances, n_links), dtype=wp.mat44, device=device, requires_grad=False)
        cmd1 = wp.launch(
            compute_forward_kinematics_matrix_local_kernel,
            dim=(num_instances, n_joints),
            inputs=[
                q_wp,
                spec_t.actuated_joint_indices,
                spec_t.mimic_actuated_joint_indices,
                spec_t.mimic_multipliers,
                spec_t.mimic_offsets,
                spec_t.joint_types,
                spec_t.joint_axes,
                spec_t.parent_joint_transforms_matrix,
            ],
            outputs=[X_local_wp],
            device=device,
            record_cmd=record_cmd,
        )
        cmd2 = wp.launch(
            compute_forward_kinematics_matrix_accum_kernel,
            dim=(num_instances, n_links),
            inputs=[X_local_wp, base_wp, spec_t.parent_joint_indices, spec_t.link_parent_joint_indices],
            outputs=[T_world_joint_wp, T_world_link_wp],
            device=device,
            record_cmd=record_cmd,
        )
        return q_buf, base_buf, q_wp, base_wp, X_local_wp, T_world_joint_wp, T_world_link_wp, spec_t, cmd1, cmd2

    @staticmethod
    def forward(
        ctx,
        robot_spec: "RobotSpec",
        q: Float[torch.Tensor, "... num_actuated_joints"],
        T_world_base: Float[torch.Tensor, "... 4 4"],
        record_cmd: bool = False,
    ):
        batch_shape = q.shape[:-1]
        num_instances = math.prod(batch_shape) if batch_shape else 1
        num_actuated = q.shape[-1]
        device = q.device
        wp.init()  # device_from_torch needs the warp runtime; forward() may be the process's first warp call
        wp_device = wp.device_from_torch(device)

        spec_t = robot_spec.get_tensors(str(wp_device))
        q_buf, base_buf, q_wp, base_wp, X_local_wp, T_world_joint_wp, T_world_link_wp, spec_t, cmd1, cmd2 = (
            WarpForwardKinematicsMatrix._get_cached_fk(spec_t, num_instances, record_cmd)
        )
        q_buf.copy_(q.contiguous().view(num_instances, num_actuated))
        base_buf.copy_(T_world_base.to(device=device, dtype=torch.float32).contiguous().view(num_instances, 4, 4))
        if cmd1 is not None:
            cmd1.launch()
            cmd2.launch()
        else:
            wp.launch(
                compute_forward_kinematics_matrix_local_kernel,
                dim=(num_instances, robot_spec.num_joints),
                inputs=[
                    q_wp,
                    spec_t.actuated_joint_indices,
                    spec_t.mimic_actuated_joint_indices,
                    spec_t.mimic_multipliers,
                    spec_t.mimic_offsets,
                    spec_t.joint_types,
                    spec_t.joint_axes,
                    spec_t.parent_joint_transforms_matrix,
                ],
                outputs=[X_local_wp],
                device=str(wp_device),
            )
            wp.launch(
                compute_forward_kinematics_matrix_accum_kernel,
                dim=(num_instances, robot_spec.num_links),
                inputs=[X_local_wp, base_wp, spec_t.parent_joint_indices, spec_t.link_parent_joint_indices],
                outputs=[T_world_joint_wp, T_world_link_wp],
                device=str(wp_device),
            )

        result = wp.to_torch(T_world_link_wp).view(*batch_shape, robot_spec.num_links, 4, 4)

        requires_grad = q.requires_grad or T_world_base.requires_grad
        if requires_grad:
            # snapshot outputs so a later forward on the same cached buffers doesn't overwrite them
            ctx.robot_spec = robot_spec
            ctx.save_for_backward(
                wp.to_torch(T_world_joint_wp).view(*batch_shape, robot_spec.num_joints, 4, 4).clone(),
                result.clone(),
                T_world_base.to(device=device, dtype=torch.float32).contiguous().view(*batch_shape, 4, 4).clone(),
            )
            result = result.clone()

        return result

    @staticmethod
    def backward(
        ctx, link_poses_grad: Float[torch.Tensor, "... num_links 4 4"]
    ) -> Tuple[
        None,
        Optional[Float[torch.Tensor, "... num_actuated_joints"]],
        Optional[Float[torch.Tensor, "... 4 4"]],
        None,
    ]:
        T_world_joint, T_world_link, T_world_base = ctx.saved_tensors
        grad_q, grad_base = _WarpFKAnalyticalBackward.apply(
            ctx.robot_spec,
            link_poses_grad,
            T_world_joint,
            T_world_link,
            T_world_base,
        )
        return None, grad_q, grad_base, None


class WarpTransformLinkPoints(Function):
    @staticmethod
    def forward(
        ctx,
        T_world_link: Float[torch.Tensor, "... num_links 4 4"],
        local_points: Float[torch.Tensor, "num_points 3"],
        point_link_indices: torch.Tensor,
    ) -> Float[torch.Tensor, "... num_points 3"]:
        batch_shape = T_world_link.shape[:-3]
        num_instances = math.prod(batch_shape) if batch_shape else 1
        num_links = T_world_link.shape[-3]
        num_points = local_points.shape[0]
        device = T_world_link.device
        wp.init()  # device_from_torch needs the warp runtime; forward() may be the process's first warp call
        wp_device = wp.device_from_torch(device)

        requires_grad = T_world_link.requires_grad or local_points.requires_grad

        T_world_link_wp = wp.from_torch(
            T_world_link.contiguous().view(num_instances, num_links, 4, 4),
            dtype=wp.mat44,
            requires_grad=requires_grad,
        )

        local_points_wp = wp.from_torch(
            local_points.contiguous().to(device=device, dtype=torch.float32),
            dtype=wp.vec3,
            requires_grad=requires_grad,
        )

        point_link_indices_wp = wp.from_torch(
            point_link_indices.contiguous().to(device=device, dtype=torch.int32),
            dtype=wp.int32,
            requires_grad=False,
        )

        world_points_wp = wp.empty(
            (num_instances, num_points), dtype=wp.vec3, device=wp_device, requires_grad=requires_grad
        )

        wp.launch(
            kernel=transform_link_points_matrix_kernel,
            dim=(num_instances, num_points),
            inputs=[T_world_link_wp, local_points_wp, point_link_indices_wp, world_points_wp],
            device=wp_device,
        )

        if requires_grad:
            ctx.batch_shape = batch_shape
            ctx.num_instances = num_instances
            ctx.num_links = num_links
            ctx.num_points = num_points
            ctx.T_world_link_wp = T_world_link_wp
            ctx.local_points_wp = local_points_wp
            ctx.point_link_indices_wp = point_link_indices_wp
            ctx.world_points_wp = world_points_wp

        return wp.to_torch(world_points_wp).view(*batch_shape, num_points, 3)

    @staticmethod
    def backward(
        ctx, world_points_grad: Float[torch.Tensor, "... num_points 3"]
    ) -> Tuple[Optional[Float[torch.Tensor, "... num_links 4 4"]], Optional[Float[torch.Tensor, "num_points 3"]], None]:
        batch_shape = ctx.batch_shape
        num_instances = ctx.num_instances
        num_links = ctx.num_links
        num_points = ctx.num_points

        ctx.world_points_wp.grad = wp.from_torch(
            world_points_grad.contiguous().view(num_instances, num_points, 3),
            dtype=wp.vec3,
            requires_grad=False,
        )
        ctx.T_world_link_wp.grad = wp.zeros_like(ctx.T_world_link_wp)
        ctx.local_points_wp.grad = wp.zeros_like(ctx.local_points_wp)

        wp_device = ctx.T_world_link_wp.device

        wp.launch(
            kernel=transform_link_points_matrix_kernel,
            dim=(num_instances, num_points),
            inputs=[ctx.T_world_link_wp, ctx.local_points_wp, ctx.point_link_indices_wp, ctx.world_points_wp],
            adj_inputs=[ctx.T_world_link_wp.grad, ctx.local_points_wp.grad, None, ctx.world_points_wp.grad],
            adjoint=True,
            device=wp_device,
        )

        T_world_link_grad = None
        if ctx.needs_input_grad[0]:
            T_world_link_grad = wp.to_torch(ctx.T_world_link_wp.grad).view(*batch_shape, num_links, 4, 4)

        local_points_grad = None
        if ctx.needs_input_grad[1]:
            local_points_grad = wp.to_torch(ctx.local_points_wp.grad).view(num_points, 3)

        return T_world_link_grad, local_points_grad, None


class _WarpFKAnalyticalBackward(Function):
    """Analytical FK backward as autograd.Function."""

    @staticmethod
    def forward(
        robot_spec: "RobotSpec",
        grad_T_link: torch.Tensor,
        T_world_joint: torch.Tensor,
        T_world_link: torch.Tensor,
        T_world_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_shape = grad_T_link.shape[:-3]
        num_instances = math.prod(batch_shape) if batch_shape else 1
        num_actuated = robot_spec.num_actuated_joints
        device = grad_T_link.device
        wp.init()  # device_from_torch needs the warp runtime; forward() may be the process's first warp call
        wp_device = wp.device_from_torch(device)
        spec_t = robot_spec.get_tensors(str(wp_device))

        T_world_joint_wp = wp.from_torch(
            T_world_joint.contiguous().view(num_instances, robot_spec.num_joints, 4, 4),
            dtype=wp.mat44,
            requires_grad=False,
        )
        T_world_link_wp = wp.from_torch(
            T_world_link.contiguous().view(num_instances, robot_spec.num_links, 4, 4),
            dtype=wp.mat44,
            requires_grad=False,
        )
        T_world_base_wp = wp.from_torch(
            T_world_base.contiguous().view(num_instances, 4, 4),
            dtype=wp.mat44,
            requires_grad=False,
        )
        grad_T_link_wp = wp.from_torch(
            grad_T_link.contiguous().to(dtype=torch.float32).view(num_instances, robot_spec.num_links, 4, 4),
            dtype=wp.mat44,
            requires_grad=False,
        )

        grad_q_wp = wp.zeros((num_instances, num_actuated), dtype=wp.float32, device=wp_device)
        grad_base_wp = wp.zeros((num_instances, 16), dtype=wp.float32, device=wp_device)

        wp.launch(
            kernel=compute_forward_kinematics_matrix_backward_kernel,
            dim=[num_instances],
            inputs=[
                T_world_joint_wp,
                T_world_link_wp,
                T_world_base_wp,
                grad_T_link_wp,
                spec_t.joint_twists,
                spec_t.link_ancestor_joints_mask,
                spec_t.joints_to_actuated_mapping,
                spec_t.link_parent_joint_indices,
            ],
            outputs=[grad_q_wp, grad_base_wp],
            device=wp_device,
        )

        grad_q = wp.to_torch(grad_q_wp).view(*batch_shape, num_actuated)
        grad_base = wp.to_torch(grad_base_wp).view(*batch_shape, 4, 4)
        return grad_q, grad_base

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_grad_q, grad_grad_base):
        raise NotImplementedError("Second derivatives not supported")
