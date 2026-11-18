"""Projected link-origin constraints shared by frame and trajectory qpos fitting."""

from typing import List, Optional, cast

import numpy as np
import torch
import warp as wp

from robokit.lie.se3 import SE3Var, se3_to_matrix
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.task import EagerTask
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.warp.torch_wrappers import project_points


@wp.kernel
def _accumulate_keypoints(
    residual: wp.array2d(dtype=wp.float32),
    jacobian: wp.array3d(dtype=wp.float32),
    normal: wp.array3d(dtype=wp.float32),
    gradient: wp.array3d(dtype=wp.float32),
    costs: wp.array1d(dtype=wp.float32),
    build: int,
):
    b, row = wp.tid()  # pyright: ignore[reportGeneralTypeIssues, reportAssignmentType]
    value = residual[b, row]
    wp.atomic_add(costs, b, 0.5 * value * value)
    if build:
        for i in range(jacobian.shape[2]):
            wp.atomic_add(gradient, b, i, 0, jacobian[b, row, i] * value)
            for j in range(jacobian.shape[2]):
                wp.atomic_add(normal, b, i, j, jacobian[b, row, i] * jacobian[b, row, j])


class KeypointAlignmentTask(EagerTask):
    """Huber image residuals using native link-position Jacobians.

    Observations:
        - Pixels have shape `[views,points,2]` or per-batch `[batch,views,points,2]`.
        - NaN coordinates or zero weights exclude missing observations.
        - Variables follow the mask task's `[camera, absolute q]` tangent order.

    Example:
        >>> # xdoctest: +REQUIRES(module:nvdiffrast)
        >>> from robokit.assets.robots.arms import xarm7
        >>> robot = Robot.load(str(xarm7.URDF_PATH))
        >>> pixels = torch.tensor([[[100., 100.]]])
        >>> task = KeypointAlignmentTask(robot, [robot.spec.link_names[-1]], pixels, torch.eye(3))
        >>> task.residual_dim
        2
    """

    def __init__(
        self,
        robot: Robot,
        link_names: List[str],
        pixels: torch.Tensor,
        intrinsic: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        T_mount_base: Optional[torch.Tensor] = None,
        weight: float = 0.03,
        sigma: float = 8.0,
        huber_delta: float = 12.0,
    ):
        wp.init()
        if pixels.ndim not in (3, 4) or pixels.shape[-2:] != (len(link_names), 2) or not link_names:
            raise ValueError("expected keypoints [N,P,2] or [B,N,P,2] with P link names")
        if not np.isfinite([weight, sigma, huber_delta]).all() or weight < 0 or sigma <= 0 or huber_delta <= 0:
            raise ValueError("keypoint weight must be nonnegative; sigma and Huber delta must be positive")
        self.robot, self.weight, self.sigma, self.delta = robot, weight, sigma, huber_delta / sigma
        self.views, self.points = pixels.shape[-3:-1]
        pixels = pixels.detach().to(dtype=torch.float32)
        valid = torch.isfinite(pixels).all(-1)
        weights = valid.float() if weights is None else weights.to(pixels)
        if weights.shape != pixels.shape[:-1] or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("expected nonnegative keypoint weights matching [*,N,P]")
        if ((weights > 0) & ~valid).any():
            raise ValueError("visible keypoints must have finite coordinates")
        self.pixels = torch.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)
        self.weights = weights
        self.intrinsic = intrinsic.to(pixels).expand(self.views, 3, 3)
        self.mount = (
            torch.eye(4, device=pixels.device).expand(self.views, 4, 4)
            if T_mount_base is None
            else T_mount_base.to(pixels)
        )
        target = torch.zeros((1, self.points, 7), device=pixels.device)
        target[..., 3] = 1
        self.position = PositionTask(
            robot,
            [robot.spec.link_names.index(n) for n in link_names],
            wp.from_torch(target, dtype=wp_vec7),
            weight=1.0,
        )
        self.position.var_key = VarKey.QPOS_OFFSET
        self.batch_size = 0

    @property
    def residual_dim(self) -> int:
        return 2 * self.views * self.points

    def prepare(self, var_values: VarValues, proposed: bool, iter_idx: int):
        """Project link origins and prepare robust image residuals and optional Jacobians."""
        del iter_idx
        state = cast(RobotState, var_values.get(VarKey.QPOS_OFFSET))
        size, dimension = var_values.batch_size, var_values.tangent_dim
        if size != self.batch_size:
            device = str(self.pixels.device)
            self.world = wp.empty((size, 3 * self.points), device=device)
            self.spatial = torch.empty(
                (size, 3 * self.points, self.robot.num_actuated_joints), device=self.pixels.device
            )
            self.current = torch.empty((size, self.residual_dim), device=self.pixels.device)
            self.proposed = torch.empty_like(self.current)
            self.jacobian = torch.empty((size, self.residual_dim, dimension), device=self.pixels.device)
            self.batch_size = size
        # --- project link origins ---
        state.invalidate()
        native = VarValues.from_var(state, VarKey.QPOS_OFFSET)
        self.position.compute_weighted_residual(native, out_residual=self.world)
        world = -wp.to_torch(self.world).reshape(size, self.points, 3)
        mounted = torch.einsum("nij,spj->snpi", self.mount[:, :3, :3], world) + self.mount[None, :, None, :3, 3]
        camera = wp.to_torch(se3_to_matrix(cast(SE3Var, var_values.get(VarKey.ROBOT)).xyz_wxyz))
        point = torch.einsum("sij,snpj->snpi", camera[:, :3, :3], mounted) + camera[:, None, None, :3, 3]
        z = point[..., 2].clamp(min=0.001)
        focal = self.intrinsic[:, [0, 1], [0, 1]]
        error = (project_points(point, self.intrinsic[None]) - self.pixels) / self.sigma
        # --- robust image residuals ---
        magnitude = error.abs()
        root = (2 * self.delta * magnitude - self.delta**2).clamp(min=1e-12).sqrt()
        normalization = (self.weight * self.weights / (self.weights > 0).sum(-1, keepdim=True).clamp(min=1)).sqrt()[
            ..., None
        ]
        residual = torch.where(magnitude <= self.delta, error, error.sign() * root) * normalization
        (self.proposed if proposed else self.current).copy_(residual.reshape(size, -1))
        if proposed:
            return
        # --- analytic projection Jacobians ---
        self.spatial.zero_()
        self.position.compute_weighted_jacobian_analytic(native, out_jacobian=wp.from_torch(self.spatial))
        projection = torch.zeros((*point.shape[:-1], 2, 3), device=point.device)
        projection[..., 0, 0] = focal[None, :, None, 0] / z
        projection[..., 1, 1] = focal[None, :, None, 1] / z
        projection[..., 2] = (-point[..., :2] * focal[None, :, None] / z[..., None] ** 2) * (point[..., 2:] > 0.001)
        x, y, zz = mounted.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack([zero, -zz, y, zz, zero, -x, -y, x, zero], dim=-1).reshape(*mounted.shape[:-1], 3, 3)
        rotation = camera[:, None, None, :3, :3].expand(size, self.views, self.points, 3, 3)
        camera_jacobian = torch.cat([rotation, -rotation @ skew], dim=-1)
        joint_jacobian = (
            rotation @ self.mount[None, :, None, :3, :3] @ (-self.spatial.reshape(size, 1, self.points, 3, -1))
        )
        jacobian = projection @ torch.cat([camera_jacobian, joint_jacobian], dim=-1)
        robust = torch.where(magnitude <= self.delta, torch.ones_like(error), self.delta / root)
        self.jacobian.copy_((jacobian * (robust * normalization / self.sigma)[..., None]).reshape_as(self.jacobian))

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        self.prepare(var_values, True, 0)
        if out_residual is None:
            return wp.from_torch(self.proposed)
        wp.to_torch(out_residual)[:, row_offset : row_offset + self.residual_dim].copy_(self.proposed)
        return out_residual

    def accumulate_normal_equations(
        self,
        var_values: VarValues,
        *,
        costs: wp.array,
        JtJ: Optional[wp.array] = None,
        Jtr: Optional[wp.array] = None,
    ):
        wp.launch(
            _accumulate_keypoints,
            dim=self.current.shape,
            inputs=[
                wp.from_torch(self.current if JtJ is not None else self.proposed),
                wp.from_torch(self.jacobian),
                JtJ,
                Jtr,
                costs,
                int(JtJ is not None),
            ],
            device=costs.device,
        )
