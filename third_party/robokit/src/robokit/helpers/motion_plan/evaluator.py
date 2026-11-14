# pyright: reportArgumentType=false, reportIndexIssue=false, reportOperatorIssue=false
from dataclasses import dataclass
from typing import Generic, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import warp as wp

from robokit.geom.scene import WarpScene
from robokit.lie.se3_kernels import se3_compose_func, se3_inverse_func
from robokit.robo.robot import Robot
from robokit.robo.robot_state import RobotState
from robokit.utils.warp_utils import resolve_device, wp_device_type, wp_vec7


ArrayT = TypeVar("ArrayT")


# --- results ---------------------------------------------------------------
@dataclass
class MotionPlanMetrics(Generic[ArrayT]):
    """Endpoint metrics with omitted outputs allocated and inapplicable fields set to None.

    Metric definitions:
    - ``q_error``: Euclidean norm in native joint coordinates, mixing units for robots with prismatic joints.
    - Success: endpoint reach and interior-frame world penetration only.
    """

    success: Optional[ArrayT]
    position_error_m: Optional[ArrayT]
    orientation_error_rad: Optional[ArrayT]
    penetration_mm: ArrayT  # `[batch, frames]`: start, interior frames, goal
    q_error: Optional[ArrayT] = None


# --- kernels ---------------------------------------------------------------
@wp.func
def _worst_planned_penetration(penetration_mm: wp.array2d(dtype=wp.float32), batch: int) -> float:
    """Deepest penetration over the frames the planner chose; the given start and goal frames are excluded."""
    worst = float(0.0)
    for frame in range(1, penetration_mm.shape[1] - 1):
        worst = wp.max(worst, penetration_mm[batch, frame])
    return worst


@wp.kernel
def _compute_pose_goal_metrics_kernel(
    T_world_link: wp.array3d(dtype=wp_vec7),
    T_world_target: wp.array2d(dtype=wp_vec7),
    target_link_indices: wp.array1d(dtype=wp.int32),
    last_frame: int,
    position_threshold_m: float,
    orientation_threshold_rad: float,
    penetration_threshold_mm: float,
    penetration_mm: wp.array2d(dtype=wp.float32),
    success: wp.array1d(dtype=wp.bool),
    position_error_m: wp.array1d(dtype=wp.float32),
    orientation_error_rad: wp.array1d(dtype=wp.float32),
):
    batch = wp.tid()
    position, angle = float(0.0), float(0.0)
    for link in range(target_link_indices.shape[0]):
        error = se3_compose_func(
            se3_inverse_func(T_world_link[batch, last_frame, target_link_indices[link]]), T_world_target[batch, link]
        )
        position = wp.max(position, wp.sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2]))
        angle = wp.max(
            angle, 2.0 * wp.asin(wp.min(wp.sqrt(error[4] * error[4] + error[5] * error[5] + error[6] * error[6]), 1.0))
        )
        if not wp.isfinite(error[0] + error[1] + error[2] + error[3] + error[4] + error[5] + error[6]):
            position, angle = wp.inf, wp.inf
    position_error_m[batch] = position
    orientation_error_rad[batch] = angle
    success[batch] = (
        position < position_threshold_m
        and angle < orientation_threshold_rad
        and _worst_planned_penetration(penetration_mm, batch) < penetration_threshold_mm
    )


@wp.kernel
def _compute_joint_goal_metrics_kernel(
    q: wp.array3d(dtype=wp.float32),
    target_q: wp.array2d(dtype=wp.float32),
    last_frame: int,
    num_dofs: int,
    q_threshold: float,
    penetration_threshold_mm: float,
    penetration_mm: wp.array2d(dtype=wp.float32),
    success: wp.array1d(dtype=wp.bool),
    q_error: wp.array1d(dtype=wp.float32),
):
    batch = wp.tid()
    error_sq = float(0.0)
    for dof in range(num_dofs):
        error = q[batch, last_frame, dof] - target_q[batch, dof]
        error_sq += error * error
    error = wp.sqrt(error_sq)
    q_error[batch] = error
    success[batch] = (
        error < q_threshold and _worst_planned_penetration(penetration_mm, batch) < penetration_threshold_mm
    )


@wp.kernel
def _compute_penetration_kernel(
    signed_dists: wp.array3d(dtype=wp.float32),
    radii: wp.array1d(dtype=wp.float32),
    link_indices: wp.array1d(dtype=wp.int32),
    has_floating_base: int,
    penetration_mm: wp.array2d(dtype=wp.float32),
):
    """Deepest sphere-into-scene overlap per frame; base-link spheres never move, so skip them."""
    batch, frame = wp.tid()  # pyright: ignore
    worst = float(0.0)
    for sphere in range(radii.shape[0]):
        if has_floating_base != 0 or link_indices[sphere] > 0:
            worst = wp.max(worst, radii[sphere] - signed_dists[batch, frame, sphere])
    penetration_mm[batch, frame] = worst * 1000.0


# --- evaluator -------------------------------------------------------------
class MotionPlanEvaluator:
    """Compute endpoint success independently from trajectory optimization and benchmarking."""

    def __init__(
        self,
        robot: Robot,
        target_links: Union[str, int, Sequence[Union[str, int]]],
        num_frames: int,
        *,
        scene: Optional[WarpScene] = None,
        device: Optional[wp_device_type] = None,
    ):
        device = str(resolve_device(device))
        self.robot = robot
        links = [target_links] if isinstance(target_links, (str, int)) else list(target_links)
        self.target_link_indices = [
            robot.link_names.index(link) if isinstance(link, str) else int(link) for link in links
        ]
        if not self.target_link_indices or any(i < 0 or i >= len(robot.link_names) for i in self.target_link_indices):
            raise ValueError("target_links must contain valid robot links.")
        self._target_link_indices = wp.array(self.target_link_indices, dtype=wp.int32, device=device)
        if num_frames < 1:
            raise ValueError("num_frames must be positive.")
        self.num_frames = num_frames
        self.scene = scene
        self.device = wp.get_device(device)
        self._batch_size: Optional[int] = None
        self._sphere_buffers: Optional[Tuple[wp.array, wp.array, wp.array, wp.array]] = None

    def warmup(self, batch_size: int):
        """Allocate batch-shaped evaluation buffers once."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self._batch_size is not None:
            if batch_size != self._batch_size:
                raise ValueError(f"Batch size mismatch: expected {self._batch_size}, got {batch_size}")
            return
        self._batch_size = batch_size
        if self.scene is not None:
            num_spheres = len(self.robot.spec.local_collision_sphere_radii)
            count = batch_size * self.num_frames * num_spheres
            self._sphere_buffers = (
                wp.empty(count, dtype=wp.float32, device=self.device),
                wp.zeros(count, dtype=wp.int32, device=self.device),
                wp.from_numpy(
                    self.robot.spec.local_collision_sphere_radii.astype(np.float32),
                    dtype=wp.float32,
                    device=self.device,
                ),
                wp.from_numpy(
                    self.robot.spec.collision_spheres_link_indices.astype(np.int32),
                    dtype=wp.int32,
                    device=self.device,
                ),
            )

    def compute_metrics(
        self,
        state: RobotState,
        *,
        T_world_target: Optional[wp.array] = None,
        target_q: Optional[wp.array] = None,
        scene_indices: Optional[wp.array] = None,
        position_threshold_m: float = 0.005,
        orientation_threshold_rad: float = 0.05,
        q_threshold: float = 0.05,
        penetration_threshold_mm: float = 5.0,
        out: Optional[MotionPlanMetrics[wp.array]] = None,
    ) -> MotionPlanMetrics[wp.array]:
        """Evaluate trajectories independently of cost weights, refreshing state FK and filling ``out`` or new buffers.

        Goal-dependent outputs:
        - Pose goal: position/orientation errors.
        - Joint goal: ``q_error``.
        - No goal: penetration only, with success=None.

        Collision-check scope:
        - Check interior frames only, excluding start and goal.
        - Check collision success only, without certifying other task constraints.
        """
        if T_world_target is not None and target_q is not None:
            raise ValueError("Provide at most one of T_world_target or target_q.")
        has_goal = T_world_target is not None or target_q is not None
        if not has_goal and self.scene is None:
            raise ValueError("Penetration-only scoring needs a scene.")
        q = state.q
        batch_size = state.batch_size
        if (
            state.robot is not self.robot
            or q.ndim != 3
            or q.shape[1:] != (self.num_frames, self.robot.num_actuated_joints)
        ):
            raise ValueError("state must belong to this robot with shape (batch, frames, dofs).")
        if q.dtype != wp.float32 or q.device != self.device:
            raise ValueError("state.q must be float32 on the evaluator device.")
        if state.T_world_base.shape != q.shape[:2] or state.T_world_base.device != self.device:
            raise ValueError("state.T_world_base must match the trajectory batch and frames.")
        for value, dtype, shape, name in (
            (target_q, wp.float32, (batch_size, self.robot.num_actuated_joints), "target_q"),
            (scene_indices, wp.int32, (batch_size,), "scene_indices"),
        ):
            if value is not None and (value.dtype, value.shape, value.device) != (dtype, shape, self.device):
                raise ValueError(f"{name} must have shape {shape}, dtype {dtype}, and device {self.device}.")
        if T_world_target is not None and (
            T_world_target.dtype != wp_vec7
            or T_world_target.device != self.device
            or T_world_target.shape
            not in (
                (batch_size, len(self.target_link_indices)),
                (batch_size,) if len(self.target_link_indices) == 1 else (),
            )
        ):
            raise ValueError("T_world_target must match the batch and target links on the evaluator device.")
        if scene_indices is not None:
            indices = scene_indices.numpy()
            count = self.scene.num_scenes if self.scene is not None else 1
            if np.any(indices < 0) or np.any(indices >= count):
                raise ValueError("scene_indices contains an out-of-range scene index.")
        if any(
            not np.isfinite(v) or v < 0
            for v in (position_threshold_m, orientation_threshold_rad, q_threshold, penetration_threshold_mm)
        ):
            raise ValueError("Metric thresholds must be finite and nonnegative.")
        self.warmup(batch_size)
        metrics = (
            out
            if out is not None
            else MotionPlanMetrics(
                success=None,
                position_error_m=None,
                orientation_error_rad=None,
                penetration_mm=wp.empty((batch_size, self.num_frames), dtype=wp.float32, device=self.device),
            )
        )
        for name, active, dtype in (
            ("success", has_goal, wp.bool),
            ("position_error_m", T_world_target is not None, wp.float32),
            ("orientation_error_rad", T_world_target is not None, wp.float32),
            ("q_error", target_q is not None, wp.float32),
        ):
            value = getattr(metrics, name)
            if not active:
                setattr(metrics, name, None)
            elif value is None:
                setattr(metrics, name, wp.empty(batch_size, dtype=dtype, device=self.device))
            elif value.shape != (batch_size,) or value.dtype != dtype or value.device != self.device:
                raise ValueError(f"out.{name} has an incompatible shape, dtype, or device.")
        if (
            metrics.penetration_mm.shape != (batch_size, self.num_frames)
            or metrics.penetration_mm.dtype != wp.float32
            or metrics.penetration_mm.device != self.device
        ):
            raise ValueError("out.penetration_mm has an incompatible shape, dtype, or device.")
        metrics.penetration_mm.zero_()
        state.invalidate()
        self.robot.forward_kinematics(state)
        if self.scene is not None and self._sphere_buffers is not None:
            signed_dists, query_scene_indices, radii, link_indices = self._sphere_buffers
            self.robot.transform_collision_spheres(state)
            num_spheres = radii.shape[0]
            centers = state.collision_sphere_centers_world.reshape((batch_size * self.num_frames * num_spheres,))
            if scene_indices is None:
                query_scene_indices.zero_()
            else:
                from robokit.utils.warp_utils import repeat

                repeat(scene_indices, self.num_frames * num_spheres, out=query_scene_indices)
            self.scene.query_sdf(
                centers,
                scene_indices=query_scene_indices.reshape(centers.shape),
                distance_only=True,
                out_signed_dists=signed_dists.reshape(centers.shape),
            )
            wp.launch(
                _compute_penetration_kernel,
                dim=(batch_size, self.num_frames),
                inputs=[
                    signed_dists.reshape((batch_size, self.num_frames, num_spheres)),
                    radii,
                    link_indices,
                    int(state.has_floating_base),
                    metrics.penetration_mm,
                ],
                device=self.device,
            )
        if T_world_target is None and target_q is None:
            return metrics
        if target_q is not None:
            wp.launch(
                _compute_joint_goal_metrics_kernel,
                dim=q.shape[0],
                inputs=[
                    q,
                    target_q,
                    self.num_frames - 1,
                    self.robot.num_actuated_joints,
                    q_threshold,
                    penetration_threshold_mm,
                    metrics.penetration_mm,
                    metrics.success,
                    metrics.q_error,
                ],
                device=self.device,
            )
            return metrics
        wp.launch(
            _compute_pose_goal_metrics_kernel,
            dim=q.shape[0],
            inputs=[
                state.T_world_link,
                T_world_target.reshape((batch_size, len(self.target_link_indices))),
                self._target_link_indices,
                self.num_frames - 1,
                position_threshold_m,
                orientation_threshold_rad,
                penetration_threshold_mm,
                metrics.penetration_mm,
                metrics.success,
                metrics.position_error_m,
                metrics.orientation_error_rad,
            ],
            device=self.device,
        )
        return metrics


__all__ = ["MotionPlanEvaluator", "MotionPlanMetrics"]
