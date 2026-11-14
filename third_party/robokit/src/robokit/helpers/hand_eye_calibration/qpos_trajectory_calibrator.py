"""Unknown joint trajectories from calibrated, synchronized silhouette views."""

from dataclasses import replace
from enum import Enum
from time import perf_counter
from typing import Any, Dict, List, Optional, Type, cast

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from robokit.helpers.hand_eye_calibration.calibrator import (
    HandEyeCalibrator,
    _torch_warp_stream,
    build_robot_render_data,
)
from robokit.helpers.hand_eye_calibration.config import QposTrajectoryCalibratorConfig
from robokit.helpers.hand_eye_calibration.keypoint_alignment_task import KeypointAlignmentTask
from robokit.helpers.hand_eye_calibration.mask_alignment_task import (
    MaskAlignmentOffsetTask,
    _precompute_merged_mesh,
    _rasterize_context,
    _render_batch,
)
from robokit.helpers.hand_eye_calibration.trajectory_alignment import (
    initialization_frames,
    trajectory_normal_equations,
)
from robokit.lie.se3 import SE3Var, se3_from_matrix, se3_to_matrix
from robokit.opt.population_solver import PopulationSolver
from robokit.opt.var_values import Var, VarKey, VarValues
from robokit.robo import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask, _temporal_differences
from robokit.terms.task import EagerTask
from robokit.utils.warp_utils import wp_vec7


class _FrameMasks(MaskAlignmentOffsetTask):
    """Flatten trajectory frames into native per-pose mask normal blocks."""

    def _alloc_buffers(self, batch_size: int, device: str):
        super()._alloc_buffers(batch_size, device)
        self._target_blurred = self._blur(self.sequence_masks, cast(float, self.blur_sigma)).contiguous()
        self._wp_target_blurred = wp.from_torch(self._target_blurred)
        self._wp_observation_weights = wp.from_torch(self.sequence_weights.contiguous())


class _TrajectoryMasks(EagerTask):
    """Couple native image derivatives across frames and shared cameras."""

    @property
    def residual_dim(self) -> int:
        return 0  # Streaming equations do not allocate pixel Jacobians.

    def compute_weighted_residual(self, *args, **kwargs) -> wp.array:
        raise NotImplementedError("Trajectory silhouettes stream normal equations.")

    def __init__(
        self,
        robot: Robot,
        intrinsic: torch.Tensor,
        cameras: torch.Tensor,
        masks: torch.Tensor,
        weights: torch.Tensor,
        times: torch.Tensor,
        blur: float,
        patience: int,
        config: QposTrajectoryCalibratorConfig,
        camera_keys: list,
        render_link_names: Optional[List[str]],
        render_data: tuple,
        keypoint_links: Optional[List[str]] = None,
        target_keypoints: Optional[torch.Tensor] = None,
        keypoint_weights: Optional[torch.Tensor] = None,
        downscale: int = 1,
    ):
        # --- bind frame observations ---
        self.times, self.camera_keys = times, camera_keys
        self.patience = patience
        self.cameras: List[SE3Var] = []
        self.frames, joints = masks.shape[0], robot.num_actuated_joints
        self.limits = torch.as_tensor(robot.spec.actuated_joint_limits, device=masks.device, dtype=masks.dtype)
        nominal = torch.as_tensor(robot.spec.midrange_q, device=masks.device)[None]
        poses, vertices, faces = render_data
        mesh_names = robot.spec.get_link_meshes(mode="visual")
        names = render_link_names or [n for n in robot.spec.link_names if n in mesh_names]
        indices = np.array([robot.spec.link_names.index(n) for n in names])
        self.terms = []
        self.keypoint_terms = []
        for view in range(masks.shape[1]):
            term = _FrameMasks(
                offset_joints=range(joints),
                optimize_q=True,
                robot=robot,
                q=nominal,
                link_indices=indices,
                T_mount_base=None,
                camera_intrinsic=intrinsic[view],
                target_masks=masks[:1, view],
                target_depths=None,
                target_link_mask=None,
                observation_weights=None,
                link_poses=poses,
                link_vertices=vertices,
                link_faces=faces,
                height=masks.shape[-2],
                width=masks.shape[-1],
                blur_sigma=blur,
                patience=1000000,
            )
            term.normalize_by_union = config.frame.candidate_metric == "iou" and downscale > 1
            term._optimize_camera = bool(camera_keys)
            term._contour_geometry = downscale == 1
            term.sequence_masks, term.sequence_weights = masks[:, view].contiguous(), weights[:, view]
            self.cameras.append(SE3Var(se3_from_matrix(wp.from_torch(cameras[view : view + 1], dtype=wp.mat44))))
            term.resolve(
                VarValues(
                    {
                        VarKey.ROBOT: self.cameras[view],
                        VarKey.QPOS_OFFSET: robot.state(wp.from_torch(nominal.contiguous())),
                    }
                ),
                {},
            )
            self.terms.append(term)
            self.keypoint_terms.append(
                None
                if target_keypoints is None
                else KeypointAlignmentTask(
                    robot,
                    cast(List[str], keypoint_links),
                    target_keypoints[:, view : view + 1],
                    intrinsic[view],
                    None if keypoint_weights is None else keypoint_weights[:, view : view + 1],
                    weight=config.frame.keypoint_weight,
                    sigma=config.frame.keypoint_sigma / downscale,
                    huber_delta=config.frame.keypoint_huber_delta / downscale,
                )
            )
        # --- allocate coupled buffers ---
        dimension = self.frames * joints + 6 * len(camera_keys)
        device = str(masks.device)
        self.normal = wp.zeros((1, dimension, dimension), device=device)
        self.gradient = wp.zeros((1, dimension, 1), device=device)
        self.current_cost, self.proposed_cost = wp.zeros(1, device=device), wp.zeros(1, device=device)
        self.blocks = wp.zeros((self.frames, 6 + joints, 6 + joints), device=device)
        self.vectors = wp.zeros((self.frames, 6 + joints, 1), device=device)
        self.costs = wp.zeros(self.frames, device=device)
        self.temporal_differences = _temporal_differences(times, config.velocity_weight, config.acceleration_weight)
        self.temporal_normal = torch.zeros((self.frames * joints, self.frames * joints), device=masks.device)
        for difference in self.temporal_differences:
            self.temporal_normal += torch.kron(difference.T @ difference, torch.eye(joints, device=masks.device))

    def prepare(self, var_values: VarValues, proposed: bool, iter_idx: int):
        """Prepare frame losses; build the coupled system only for a current-state step."""
        if not proposed and iter_idx == 0:
            self._best_cost, self._last_improvement = float("inf"), 0
        # --- accumulate image observations ---
        q = cast(RobotState, var_values.get(VarKey.ROBOT)).flatten()
        normals, gradients, cost = [], [], torch.zeros((), device=self.times.device)
        for view, term in enumerate(self.terms):
            camera = cast(SE3Var, var_values.get(self.camera_keys[view])) if self.camera_keys else self.cameras[view]
            # Each frame is a batch row with one image; native normal blocks retain [camera, q] order.
            camera = SE3Var(
                wp.from_torch(wp.to_torch(camera.xyz_wxyz).repeat(self.frames, 1).contiguous(), dtype=wp_vec7)
            )
            mapped = VarValues({VarKey.ROBOT: camera, VarKey.QPOS_OFFSET: q})
            self.costs.zero_()
            if not proposed:
                self.blocks.zero_()
                self.vectors.zero_()
            for observation in (term, self.keypoint_terms[view]):
                if observation is not None:
                    observation.prepare(mapped, proposed, iter_idx)
                    observation.accumulate_normal_equations(
                        mapped,
                        costs=self.costs,
                        JtJ=None if proposed else self.blocks,
                        Jtr=None if proposed else self.vectors,
                    )
            if not proposed:
                normals.append(wp.to_torch(self.blocks).clone())
                gradients.append(wp.to_torch(self.vectors)[..., 0].clone())
            cost += wp.to_torch(self.costs).sum()
        # --- add temporal losses ---
        positions = wp.to_torch(q.q)
        temporal = torch.zeros((), device=positions.device)
        temporal_gradient = torch.zeros_like(positions)
        for difference in self.temporal_differences:
            residual = difference @ positions
            temporal += 0.5 * residual.square().sum()
            if not proposed:
                temporal_gradient += difference.T @ residual
        wp.to_torch(self.proposed_cost if proposed else self.current_cost)[0] = cost + temporal
        if proposed:
            return
        # --- assemble shared camera and joint blocks ---
        matrix, vector, _ = trajectory_normal_equations(
            positions,
            torch.stack(normals),
            torch.stack(gradients),
            self.times,
            0.0,
            0.0,
            bool(self.camera_keys),
        )
        joint_dimension = positions.numel()
        matrix[:joint_dimension, :joint_dimension] += self.temporal_normal
        vector[:joint_dimension] += temporal_gradient.flatten()
        # Keep joints whose descent direction points outside an active limit fixed in this solve.
        joint_gradient = vector[:joint_dimension].reshape_as(positions)
        bounded = ((positions <= self.limits[:, 0]) & (joint_gradient > 0)) | (
            (positions >= self.limits[:, 1]) & (joint_gradient < 0)
        )
        indices = bounded.flatten().nonzero().flatten()
        matrix[indices] = 0
        matrix[:, indices] = 0
        vector[indices] = 0
        wp.to_torch(self.normal)[0].copy_(matrix)
        wp.to_torch(self.gradient)[0, :, 0].copy_(vector)

    def accumulate_normal_equations(
        self, var_values: VarValues, *, costs: wp.array, JtJ: Optional[wp.array] = None, Jtr: Optional[wp.array] = None
    ):
        """Add prepared costs and optional normal equations on Warp's current stream."""
        cost = self.proposed_cost if JtJ is None else self.current_cost
        with torch.cuda.stream(cast(torch.cuda.Stream, wp.stream_to_torch(costs.device))):
            wp.to_torch(costs).add_(wp.to_torch(cost))
            if JtJ is not None:
                wp.to_torch(JtJ).add_(wp.to_torch(self.normal))
                wp.to_torch(cast(wp.array, Jtr)).add_(wp.to_torch(self.gradient))

    def on_step(self, accept_mask: wp.array, iter_idx: int, costs: wp.array) -> bool:
        """Accept every frame's render and stop when aggregate cost stagnates."""
        frame_accept = wp.from_torch(wp.to_torch(accept_mask).repeat(self.frames))
        for term in self.terms:
            term._accept_render(frame_accept)
        cost = float(costs.numpy()[0])
        if cost < self._best_cost - np.finfo(np.float32).eps * max(1.0, abs(cost)):
            self._best_cost, self._last_improvement = cost, iter_idx
        return iter_idx - self._last_improvement >= self.patience


class QposTrajectoryCalibrator:
    """Fit unknown q[T,J], optionally refining static cameras[V,4,4] shared over time; temporal penalties apply only to q.

    Example (CUDA, robot meshes and observations required):
        >>> # xdoctest: +REQUIRES(module:nvdiffrast)
        >>> from robokit.helpers.hand_eye_calibration import presets
        >>> config = presets.rgb_qpos_trajectory
        >>> config.optimize_camera
        False
        >>> # helper = QposTrajectoryCalibrator(config, robot, K, H, W)
        >>> # cameras = helper.solve(cameras, masks, timestamps)
        >>> # helper.last_q.shape  # (timestamps, joints)
    """

    def __init__(
        self,
        config: QposTrajectoryCalibratorConfig,
        robot: Robot,
        camera_intrinsic: torch.Tensor,
        height: int,
        width: int,
        render_link_names: Optional[List[str]] = None,
    ):
        wp.init()
        self.config, self.robot, self.camera_intrinsic = config, robot, camera_intrinsic
        self.height, self.width, self.render_link_names = height, width, render_link_names
        self._render_geometry: Optional[tuple] = None
        self._trajectory_cost = TrajectorySmoothnessTask(robot)

    def _geometry(self) -> tuple:
        """Reuse static meshes across initialization, path ranking and coupled fitting."""
        if self._render_geometry is None:
            self._render_geometry = build_robot_render_data(
                self.robot,
                torch.as_tensor(self.robot.spec.midrange_q, device=self.camera_intrinsic.device)[None],
                link_names=self.render_link_names,
            )
        return self._render_geometry

    def _build(
        self,
        q: torch.Tensor,
        cameras: torch.Tensor,
        intrinsic: torch.Tensor,
        masks: torch.Tensor,
        weights: torch.Tensor,
        times: torch.Tensor,
        keypoint_links: Optional[List[str]],
        target_keypoints: Optional[torch.Tensor],
        keypoint_weights: Optional[torch.Tensor],
    ):
        """Build the coupled trajectory solver."""
        # --- optimization variables ---
        frames, views = masks.shape[:2]
        state = self.robot.state(wp.from_torch(q[None].contiguous()))
        state.clamp_q_on_integrate = True
        camera_keys = (
            list(cast(Type[Enum], Enum("CameraKey", {f"VIEW{i}": i for i in range(views)})))
            if self.config.optimize_camera
            else []
        )
        named: Dict[Enum, Var] = {VarKey.ROBOT: state}
        for view, key in enumerate(camera_keys):
            named[key] = SE3Var(se3_from_matrix(wp.from_torch(cameras[view : view + 1], dtype=wp.mat44)))
        self._initial_var = VarValues(named)
        self._camera_keys = camera_keys
        # --- shared camera priors ---
        camera_priors = []
        if camera_keys and self.config.camera_prior_weight:
            scale = (
                self.config.camera_prior_weight**0.5
                * np.r_[np.full(3, 1 / self.config.camera_prior_t), np.full(3, 1 / self.config.camera_prior_r)]
            )
            for key in camera_keys:
                prior = ConfigurationTask(
                    T_world_base_rest=wp.clone(cast(SE3Var, self._initial_var.get(key)).xyz_wxyz),
                    base_weight=scale,
                    include_joints=False,
                    var_key=key,
                )
                prior.resolve(self._initial_var, {})
                camera_priors.append(prior)
        # --- image pyramid ---
        stages = []
        render_data = self._geometry()
        for level in self.config.levels:
            ds = min(level.downscale, max(1, min(self.height, self.width) // 32))
            small = F.avg_pool2d(masks.flatten(0, 1)[:, None], ds)[:, 0].reshape(
                frames, views, self.height // ds, self.width // ds
            )
            k = intrinsic.clone()
            k[:, :2] /= ds
            stages.append(
                [
                    _TrajectoryMasks(
                        self.robot,
                        k,
                        cameras,
                        small,
                        weights * ds**2,
                        times,
                        level.blur_sigma,
                        level.patience,
                        self.config,
                        camera_keys,
                        self.render_link_names,
                        render_data,
                        keypoint_links,
                        None if target_keypoints is None else target_keypoints / ds,
                        keypoint_weights,
                        ds,
                    )
                ]
                + camera_priors
            )
        self._solver = PopulationSolver(stages, self.config.solver, device=str(masks.device))

    @torch.no_grad()
    def _rank_paths(
        self,
        paths: np.ndarray,
        cameras: torch.Tensor,
        masks: torch.Tensor,
        times: torch.Tensor,
        valid: torch.Tensor,
        keypoint_links: Optional[List[str]],
        target_keypoints: Optional[torch.Tensor],
        keypoint_weights: Optional[torch.Tensor],
        *,
        downscale: Optional[int] = None,
    ) -> tuple:
        """Rank paths using source masks, keypoints and temporal losses."""
        q = torch.as_tensor(paths, device=masks.device, dtype=masks.dtype)
        count, frames, joints = q.shape
        ds = min(
            self.config.levels[0].downscale if downscale is None else downscale,
            max(1, min(self.height, self.width) // 32),
        )
        target = F.avg_pool2d(masks.flatten(0, 1)[:, None], ds)[:, 0].reshape(
            frames, masks.shape[1], self.height // ds, self.width // ds
        )
        intrinsic = self.camera_intrinsic.to(masks).expand(masks.shape[1], 3, 3).clone()
        intrinsic[:, :2] /= ds
        flat = q.reshape(-1, joints)
        poses = self.robot.forward_kinematics_via_matrix_torch(flat)
        meshes = self.robot.spec.get_link_meshes(mode="visual")
        names = self.render_link_names or [n for n in self.robot.spec.link_names if n in meshes]
        indices = [self.robot.spec.link_names.index(n) for n in names]
        _, vertices, faces = self._geometry()
        errors = torch.zeros((count * frames, masks.shape[1]), device=masks.device)
        context = _rasterize_context(masks.device.index)
        # --- rendered mask scores ---
        for start in range(0, len(flat), 32):
            end = min(start + 32, len(flat))
            triangles, transformed = _precompute_merged_mesh(vertices, faces, poses[start:end, indices])
            source_rows = torch.arange(start, end, device=masks.device) % frames
            for view in range(masks.shape[1]):
                rendered, _ = _render_batch(
                    context,
                    cameras[view],
                    intrinsic[view],
                    triangles,
                    transformed,
                    self.height // ds,
                    self.width // ds,
                    False,
                )
                expected = target[source_rows, view]
                intersection = torch.minimum(rendered, expected).sum((-2, -1))
                union = torch.maximum(rendered, expected).sum((-2, -1))
                mismatch = torch.where(union > 0, 1 - intersection / union.clamp(min=1e-8), 0)
                mismatch = torch.where(
                    expected.sum((-2, -1)) > 0,
                    mismatch,
                    rendered.square().sum((-2, -1)) / (expected.shape[-2] * expected.shape[-1] * 0.05),
                )
                errors[start:end, view] = mismatch
        errors = errors.reshape(count, frames, -1)
        weighted = (errors * valid[None]).sum(-1)
        cost = 0.5 * weighted.sum(-1)
        # --- keypoint scores ---
        if target_keypoints is not None:
            state = self.robot.state(wp.from_torch(flat.contiguous()))
            for view in range(masks.shape[1]):
                camera = SE3Var(se3_from_matrix(wp.from_torch(cameras[view : view + 1], dtype=wp.mat44)))
                camera = SE3Var(wp.from_torch(wp.to_torch(camera.xyz_wxyz).repeat(len(flat), 1), dtype=wp_vec7))
                variables = VarValues({VarKey.ROBOT: camera, VarKey.QPOS_OFFSET: state})
                term = KeypointAlignmentTask(
                    self.robot,
                    cast(List[str], keypoint_links),
                    target_keypoints[:, view : view + 1].repeat(count, 1, 1, 1),
                    self.camera_intrinsic.to(masks).expand(masks.shape[1], 3, 3)[view],
                    None if keypoint_weights is None else keypoint_weights[:, view : view + 1].repeat(count, 1, 1),
                    weight=self.config.frame.keypoint_weight,
                    sigma=self.config.frame.keypoint_sigma,
                    huber_delta=self.config.frame.keypoint_huber_delta,
                )
                residual = wp.to_torch(term.compute_weighted_residual(variables)).reshape(count, frames, -1)
                cost += 0.5 * residual.square().sum((1, 2))
        for difference in _temporal_differences(times, self.config.velocity_weight, self.config.acceleration_weight):
            cost += 0.5 * torch.einsum("ab,kbj->kaj", difference, q).square().sum((1, 2))
        return (cost / valid.sum().clamp(min=1)).cpu().numpy(), (weighted / valid.sum(-1).clamp(min=1)).cpu().numpy()

    def _initialize_q(
        self,
        cameras: torch.Tensor,
        intrinsic: torch.Tensor,
        masks: torch.Tensor,
        weights: torch.Tensor,
        times: torch.Tensor,
        valid: torch.Tensor,
        mask_area: torch.Tensor,
        keypoint_links: Optional[List[str]],
        target_keypoints: Optional[torch.Tensor],
        keypoint_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Retain image-fitted seed branches beside broad samples, then select a coherent trajectory."""
        # --- frame sampler and repair deadline ---
        deadline = (
            None if self.config.initialization_seconds is None else perf_counter() + self.config.initialization_seconds
        )
        frames = masks.shape[0]
        helper = HandEyeCalibrator(
            replace(self.config.frame, optimize_q=True, optimize_camera=False),
            self.robot,
            intrinsic,
            self.height,
            self.width,
            render_link_names=self.render_link_names,
        )
        seed_helper = None
        if target_keypoints is None:
            solver = helper.config.solver  # half the schedule, one candidate per level
            ends = [max(1, i // 2) for i in solver.select_at] + [solver.num_iters // 2]
            seed_helper = HandEyeCalibrator(
                replace(
                    helper.config,
                    solver=replace(
                        solver,
                        stages=[
                            replace(stage, iters=end - start, num_candidates=1)
                            for stage, start, end in zip(solver.stages, [0, *ends], ends)
                        ],
                    ),
                    max_restarts=0,
                ),
                self.robot,
                intrinsic,
                self.height,
                self.width,
                render_link_names=self.render_link_names,
            )
        identity = torch.eye(4, device=masks.device)
        bank = {}
        clock = times.cpu().numpy()
        indices = np.arange(frames)
        spec = self.robot.spec

        def interpolate(values: np.ndarray) -> np.ndarray:
            result = np.stack([np.interp(clock, clock[indices], column) for column in values.T], axis=1)
            if len(indices) > 1:
                for endpoint, neighbor, outside in (
                    (0, 1, clock < clock[indices[0]]),
                    (-1, -2, clock > clock[indices[-1]]),
                ):
                    velocity = (values[neighbor] - values[endpoint]) / (
                        clock[indices[neighbor]] - clock[indices[endpoint]]
                    )
                    result[outside] = values[endpoint] + (clock[outside, None] - clock[indices[endpoint]]) * velocity
            return np.clip(result, spec.actuated_joint_limits[:, 0], spec.actuated_joint_limits[:, 1])

        def sample(frame: int):
            observations: Dict[str, Any] = dict(
                target_masks=masks[frame],
                observation_weights=weights[frame],
                T_mount_base=cameras,
                keypoint_links=keypoint_links,
                target_keypoints=None if target_keypoints is None else target_keypoints[frame],
                keypoint_weights=None if keypoint_weights is None else keypoint_weights[frame],
            )
            hint = None if not bank else bank[min(bank, key=lambda i: abs(clock[i] - clock[frame]))][3]
            if seed_helper is not None:
                seed_helper.solve(identity, q=hint, **observations)
                hint = cast(torch.Tensor, seed_helper.last_q)
            helper.solve(identity, q=hint, **observations)
            candidates, costs = helper.last_candidates, helper.last_candidate_costs
            seed = cast(torch.Tensor, helper.last_q)
            if seed_helper is not None:
                seed = cast(torch.Tensor, seed_helper.last_q)
                candidates = torch.cat([candidates, seed[None]])
                costs = torch.cat([costs, masks.new_tensor([seed_helper.last_final_cost])])
            bank[frame] = (
                candidates.cpu().numpy(),
                costs.cpu().numpy(),
                candidates[costs.argmin()].clone(),
                seed.clone(),
            )

        # --- source frame searches ---
        observed_views = ((mask_area > 0) & (valid > 0)).cpu().numpy()
        visible = observed_views.any(-1)
        if self.config.initialization_frames is not None and self.config.initialization_frames < frames:
            guided = np.ones(frames, dtype=bool)
            if target_keypoints is not None:
                observed = torch.isfinite(target_keypoints).all(-1)
                if keypoint_weights is not None:
                    observed &= keypoint_weights > 0
                guided = observed.any((-2, -1)).cpu().numpy()
            indices = initialization_frames(observed_views, clock, self.config.initialization_frames, guided)
        information = (mask_area * valid).sum(-1).cpu().numpy()
        for frame in indices[np.argsort(-information[indices], kind="stable")]:
            sample(int(frame))
        # --- path ranking and repair ---
        for repair in range(self.config.initialization_repairs + 1):
            indices = np.array(sorted(bank))
            self.last_candidates = np.stack([bank[i][0] for i in indices])
            self.last_candidate_costs = np.stack([bank[i][1] for i in indices])
            paths = self._trajectory_cost.compute_paths(
                self.last_candidates,
                self.last_candidate_costs,
                clock[indices],
                self.config.path_weight,
                self.config.initialization_paths,
            )
            paths = np.stack([interpolate(path) for path in paths])
            chosen = 0
            if self.config.initialization_paths > 1 or self.config.initialization_repairs:
                self.last_path_scores, errors = self._rank_paths(
                    paths, cameras, masks, times, valid, keypoint_links, target_keypoints, keypoint_weights
                )
                chosen = int(self.last_path_scores.argmin())
            path = paths[chosen]
            eligible = visible.copy()
            eligible[indices] = False  # pyright: ignore[reportIndexIssue]
            if (
                repair == self.config.initialization_repairs
                or not eligible.any()
                or (deadline is not None and perf_counter() >= deadline)
            ):
                break
            worst = int(np.where(eligible, errors[chosen], -np.inf).argmax())
            self.last_repair_indices.append(worst)
            sample(worst)
        self.last_initialization_indices = indices.copy()
        independent = torch.stack([bank[i][2] for i in indices]).cpu().numpy()
        self.last_frame_q = torch.as_tensor(
            interpolate(independent),
            device=masks.device,
            dtype=masks.dtype,
        )
        return torch.as_tensor(path, device=masks.device, dtype=masks.dtype)

    def solve(
        self,
        initial_extrinsics: torch.Tensor,
        target_masks: torch.Tensor,
        timestamps: torch.Tensor,
        observation_weights: Optional[torch.Tensor] = None,
        q: Optional[torch.Tensor] = None,
        keypoint_links: Optional[List[str]] = None,
        target_keypoints: Optional[torch.Tensor] = None,
        keypoint_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fit synchronized silhouettes and optional keypoints to a joint trajectory; an empty mask with positive weight constrains absence.

        Lifecycle:
            1. Validate calibration, timestamps and observations; normalize mask costs by area.
            2. Initialize a coherent trajectory, or use supplied q without frame sampling.
            3. Fit the coupled image, temporal and optional camera system.
            4. Resolve shared joint offsets against source frames and retain joints and cameras.

        Args:
            initial_extrinsics: Static calibrated cameras `[V,4,4]`.
            target_masks: Silhouettes `[T,V,H,W]` in `[0,1]`.
            timestamps: Finite, strictly increasing source times `[T]`.
            observation_weights: Nonnegative mask weights `[T,V]`; zero marks a missing view.
            q: Optional joint trajectory `[T,J]`; skips frame sampling.
            keypoint_links: Link-frame origins `[P]` observed by the keypoints.
            target_keypoints: Image coordinates `[T,V,P,2]`; NaNs mark missing points.
            keypoint_weights: Optional confidence weights `[T,V,P]`; zero excludes a point.

        Returns:
            Cameras `[V,4,4]` on the input device; fixed unless camera refinement is enabled.

        Outputs:
            last_frame_q: Independent frame fits, or the supplied q.
            last_initial_q: Coherent path before coupled refinement, or the supplied q.
            last_q: Final joint result `[T,J]` on the input device.
        """

        with _torch_warp_stream(initial_extrinsics.device):
            # --- prepare observations ---
            masks = target_masks.detach().to(dtype=torch.float32)
            times = timestamps.to(device=masks.device)
            cameras = initial_extrinsics.detach().to(device=masks.device, dtype=torch.float32).contiguous()
            intrinsic = self.camera_intrinsic.to(device=masks.device, dtype=torch.float32)
            if (
                masks.ndim != 4
                or masks.shape[-2:] != (self.height, self.width)
                or masks.shape[0] < 2
                or masks.shape[1] < 1
            ):
                raise ValueError("expected masks [T>=2,V,H,W]")
            frames, views = masks.shape[:2]
            if times.shape != (frames,) or not torch.isfinite(times).all() or (times.diff() <= 0).any():
                raise ValueError("timestamps must be finite and strictly increasing")
            times = (times - times[0]).to(dtype=torch.float32)
            if cameras.shape != (views, 4, 4) or not torch.isfinite(cameras).all():
                raise ValueError("expected finite static camera extrinsics [V,4,4]")
            if intrinsic.shape not in [(3, 3), (views, 3, 3)] or not torch.isfinite(intrinsic).all():
                raise ValueError("expected intrinsic [3,3] or [V,3,3]")
            intrinsic = intrinsic.expand(views, 3, 3).contiguous()
            valid = (
                torch.ones((frames, views), device=masks.device)
                if observation_weights is None
                else observation_weights.to(masks)
            )
            if (
                valid.shape != (frames, views)
                or not torch.isfinite(valid).all()
                or (valid < 0).any()
                or valid.sum() <= 0
            ):
                raise ValueError("expected nonnegative weights [T,V] with at least one observation")
            mask_area = masks.sum((-2, -1))
            weights = valid / mask_area.clamp(min=max(1, self.height * self.width * 0.05))
            if target_keypoints is not None:
                target_keypoints = target_keypoints.to(masks)
                if not keypoint_links or target_keypoints.shape != (frames, views, len(keypoint_links), 2):
                    raise ValueError("expected keypoints [T,V,P,2] and P keypoint_links")
                if keypoint_weights is not None:
                    keypoint_weights = keypoint_weights.to(masks)
                    if keypoint_weights.shape != (frames, views, len(keypoint_links)):
                        raise ValueError("expected keypoint weights [T,V,P]")
            # --- initialize joint trajectory ---
            self.last_candidates = self.last_candidate_costs = self.last_path_scores = None
            self.last_initialization_indices = np.empty(0, dtype=int)
            self.last_repair_indices = []
            if q is None:
                q = self._initialize_q(
                    cameras,
                    intrinsic,
                    masks,
                    weights,
                    times,
                    valid,
                    mask_area,
                    keypoint_links,
                    target_keypoints,
                    keypoint_weights,
                )
            else:
                q = q.detach().to(masks)
                limits = torch.as_tensor(self.robot.spec.actuated_joint_limits, device=masks.device, dtype=masks.dtype)
                if (
                    q.shape != (frames, self.robot.num_actuated_joints)
                    or not torch.isfinite(q).all()
                    or ((q < limits[:, 0]) | (q > limits[:, 1])).any()
                ):
                    raise ValueError("expected finite initial q [T,J] within joint limits")
                self.last_frame_q = q.clone()
            self.last_initial_q = q.clone()
            # --- build and solve the coupled system ---
            self._build(
                q, cameras, intrinsic, masks, weights, times, keypoint_links, target_keypoints, keypoint_weights
            )
            solved, cost = self._solver.solve(self._initial_var)
            if self._camera_keys:
                cameras = torch.stack(
                    [wp.to_torch(se3_to_matrix(cast(SE3Var, solved.get(key)).xyz_wxyz))[0] for key in self._camera_keys]
                )
            # --- shared joint offset search ---
            indices = np.unique(np.linspace(0, frames - 1, min(16, frames)).round().astype(int))
            q_path = wp.to_torch(cast(RobotState, solved.get(VarKey.ROBOT)).q)[0].cpu().numpy()
            q_path = self._trajectory_cost.compute_paths(
                q_path[:, None],
                np.zeros((frames, 1)),
                times.cpu().numpy(),
            )[0]
            candidates = self._trajectory_cost.compute_offset_candidates(q_path)
            selected = torch.as_tensor(indices, device=masks.device)
            scores, _ = self._rank_paths(
                candidates[:, indices],
                cameras,
                masks[selected],
                times[selected],
                valid[selected],
                keypoint_links,
                None if target_keypoints is None else target_keypoints[selected],
                None if keypoint_weights is None else keypoint_weights[selected],
                downscale=max(1, (min(self.height, self.width) + 127) // 128),
            )
            finalists = np.unique(np.r_[0, np.argsort(scores)[: min(4, self.config.initialization_paths)]])
            scores, _ = self._rank_paths(
                candidates[finalists],
                cameras,
                masks,
                times,
                valid,
                keypoint_links,
                target_keypoints,
                keypoint_weights,
                downscale=1,
            )
            chosen = int(finalists[scores.argmin()])
            baseline_cost = float(cost.numpy()[0])
            for candidate in (chosen, 0):
                path = self._trajectory_cost.compute_paths(
                    candidates[candidate][:, None],
                    np.zeros((frames, 1)),
                    times.cpu().numpy(),
                )[0]
                shifted_q = torch.as_tensor(path, device=masks.device, dtype=masks.dtype)
                wp.to_torch(cast(RobotState, solved.get(VarKey.ROBOT)).q)[0].copy_(shifted_q)
                solved.invalidate()
                if not candidate and np.allclose(path, q_path):
                    break
                costs = wp.zeros(1, device=str(masks.device))
                for term in self._solver.terms[-1]:
                    if isinstance(term, EagerTask):
                        term.prepare(solved, True, self.config.solver.num_iters)
                    else:
                        term.precompute(solved, need_gradient=False)
                    term.accumulate_normal_equations(solved, costs=costs)
                if not candidate or float(costs.numpy()[0]) < baseline_cost:
                    cost = costs
                    break
            wp.copy(self._solver.best_costs, cost)
            self.last_q = wp.to_torch(cast(RobotState, solved.get(VarKey.ROBOT)).q)[0].clone()
            self.last_final_cost = float(cost.numpy()[0])
            if self._camera_keys:
                cameras = torch.stack(
                    [wp.to_torch(se3_to_matrix(cast(SE3Var, solved.get(key)).xyz_wxyz))[0] for key in self._camera_keys]
                )
            return cameras
