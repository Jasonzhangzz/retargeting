"""Offline (whole-trajectory) humanoid retargeting."""

import copy
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import warp as wp

from robokit.helpers.humanoid_retargeting._kernels import compute_targets_kernel, set_trajectory_solution_kernel
from robokit.helpers.humanoid_retargeting.config import (
    HumanoidContext,
    HumanoidRetargeterOfflineConfig,
    HumanoidRetargeterOnlineConfig,
)
from robokit.helpers.ik import IKSolver
from robokit.lie.se3 import se3_identity
from robokit.opt.lm_optimizer import LMOptimizer
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.task import Context
from robokit.utils.warp_utils import resolve_device, wp_device_type, wp_vec7


class HumanoidRetargeterOffline:
    """Whole-clip retargeting with input joints ordered by `human_joint_names`.

    Lifecycle:
        1. `__init__` stores the robot and configuration.
        2. `warmup(batch_size, num_frames)` builds shape-specific buffers and optimizers.
        3. `solve(T_world_human)` retargets one fixed-shape Warp trajectory batch.
    """

    def __init__(
        self,
        config: HumanoidRetargeterOnlineConfig,
        offline_config: HumanoidRetargeterOfflineConfig,
        robot: Optional[Robot] = None,
        device: Optional[wp_device_type] = None,
    ) -> None:
        if config.interaction_terms or config.correspondence_edges:
            raise ValueError("interaction and correspondence_edges are online-only; offline retargeting ignores them.")
        self.retargeting_config = config
        self.offline_config = offline_config
        self.robot = robot if robot is not None else config.load_robot()
        self._wp_device = resolve_device(device)
        self._shape = None

        self.human_joint_names = config.human_joint_names
        self._targets = config.target_arrays(self.human_joint_names)
        self._num_targets = len(self._targets.joint_indices)

    def warmup(self, batch_size: int, num_frames: int):
        """Build and cache all state for `(batch_size, num_frames)`."""
        if batch_size < 1 or num_frames < 1:
            raise ValueError("batch_size and num_frames must be positive")
        shape = (batch_size, num_frames)
        if self._shape == shape:
            return
        self._shape = None

        robot, config = self.robot, self.retargeting_config
        links = list(config.link_mapping)
        link_indices = [robot.spec.link_names.index(name) for name in links]
        flat = batch_size * num_frames

        self._target_joint_indices_wp = wp.from_numpy(
            self._targets.joint_indices, dtype=wp.int32, device=self._wp_device
        )
        self._target_scales_wp = wp.from_numpy(self._targets.scales, dtype=wp.float32, device=self._wp_device)
        self._rotation_offsets_wp = wp.from_numpy(self._targets.rotation_offsets, dtype=wp.vec4, device=self._wp_device)
        self._position_offsets_wp = wp.from_numpy(self._targets.position_offsets, dtype=wp.vec3, device=self._wp_device)
        # Share float32 position/rotation slices with alignment terms and a wp_vec7 view with target kernels and IK.
        self._target_buf_wp = wp.zeros(
            (batch_size, num_frames, len(links), 7), dtype=wp.float32, device=self._wp_device
        )
        # Contiguous groups of seven float32 values match wp_vec7's 28-byte layout without copying.
        self._T_world_target_wp = wp.array(
            ptr=self._target_buf_wp.ptr,
            shape=(flat, len(links)),
            dtype=wp_vec7,
            device=self._wp_device,
            copy=False,
        )
        # orientation terms follow the mapped targets; `target_overrides` move the positions and the warm start
        self._T_world_mapped_wp = wp.empty_like(self._T_world_target_wp)
        self._T_world_base_target_wp = wp.from_numpy(
            np.empty((flat, 7), dtype=np.float32), dtype=wp_vec7, device=self._wp_device
        )
        self._unused_indices_wp = wp.zeros((1,), dtype=wp.int32, device=self._wp_device)
        self._unused_scales_wp = wp.zeros((1,), dtype=wp.float32, device=self._wp_device)
        self._unused_vectors_wp = wp.zeros((flat, 1, 3), dtype=wp.float32, device=self._wp_device)
        self._default_valid_lengths_wp = wp.from_numpy(
            np.full(batch_size, num_frames, dtype=np.int32), dtype=wp.int32, device=self._wp_device
        )
        self._default_human_heights_wp = wp.from_numpy(
            np.full(batch_size, config.human_height_assumption, dtype=np.float32),
            dtype=wp.float32,
            device=self._wp_device,
        )
        self._joint_limits_wp = wp.from_numpy(
            robot.spec.actuated_joint_limits.astype(np.float32, copy=False),
            dtype=wp.float32,
            device=self._wp_device,
        )
        velocity_limits = (
            np.array(config.velocity_limit_override, dtype=np.float32)
            if config.velocity_limit_override is not None
            else robot.spec.actuated_joint_velocity_limits.astype(np.float32, copy=False)
        )
        self._max_dq_wp = wp.from_numpy(
            velocity_limits * config.velocity_limit_dt * config.velocity_clamp_scale,
            dtype=wp.float32,
            device=self._wp_device,
        )
        self._output_wp = wp.empty(
            (batch_size, num_frames, 7 + robot.spec.num_actuated_joints),
            dtype=wp.float32,  # type: ignore[reportArgumentType]
            device=self._wp_device,
        )

        self._ik = IKSolver(
            self.offline_config.warm_start, robot=robot, target_links=link_indices, device=self._wp_device
        )
        self._ik.warmup(flat)
        ik_init_q = wp.from_numpy(
            np.tile(robot.spec.midrange_q.astype(np.float32), (flat, 1)),
            dtype=wp.float32,
            device=self._wp_device,
        )
        self._ik_init_state = robot.state(
            q=ik_init_q,
            T_world_base=self._T_world_base_target_wp,
        )

        q_init = np.tile(
            robot.spec.midrange_q.astype(np.float32),
            (batch_size, num_frames, 1),
        )
        self._state = robot.state(
            q=wp.from_numpy(q_init, dtype=wp.float32, device=self._wp_device),
            T_world_base=se3_identity(shape=(batch_size, num_frames), device=self._wp_device),
        )
        self._var = VarValues({VarKey.ROBOT: self._state})
        ctx = HumanoidContext
        context: Dict[Context, Any] = {
            ctx.TARGET_LINKS: link_indices,
            ctx.TARGETS: self._T_world_mapped_wp,
            ctx.TARGET_POSITIONS: self._target_buf_wp[:, :, :, 0:3],
            ctx.OVERRIDDEN_TARGETS: self._T_world_target_wp,
            ctx.VELOCITY_LIMITS: velocity_limits,
            ctx.STEP: config.velocity_limit_dt,
        }
        self._tasks = []
        for template in self.offline_config.terms:
            if template.active:
                task = copy.copy(template)
                task.resolve(self._var, context)
                if task.active:
                    task.init_buffers(self._wp_device)
                    task.set_valid_lengths(self._default_valid_lengths_wp)
                    self._tasks.append(task)
        self._optimizer = LMOptimizer(self._tasks, device=self._wp_device, config=self.offline_config.solver)
        self._shape = shape

    def _compute_targets(
        self,
        T_world_human: wp.array,
        human_heights: wp.array,
    ) -> wp.array:
        """Compute every trajectory target with one Warp launch."""
        batch_size, num_frames = cast(Tuple[int, int], self._shape)
        wp.launch(
            compute_targets_kernel,
            dim=(batch_size * num_frames, self._num_targets + 1),
            inputs=[
                T_world_human.reshape((batch_size * num_frames, len(self.human_joint_names))),
                human_heights,
                self._target_joint_indices_wp,
                self._target_scales_wp,
                self._rotation_offsets_wp,
                self._position_offsets_wp,
                self._unused_indices_wp,
                self._unused_indices_wp,
                self._unused_scales_wp,
                self._unused_scales_wp,
                self._T_world_target_wp,
                self._T_world_base_target_wp,
                self._unused_vectors_wp,
                self._num_targets,
                0,
                self._targets.root_joint_index,
                self._targets.root_target_index,
                self._targets.root_scale,
                float(self.retargeting_config.human_height_assumption),
                num_frames,
                float(self.retargeting_config.ground_height),
            ],
            device=self._wp_device,
        )
        wp.copy(self._T_world_mapped_wp, self._T_world_target_wp)
        return self._T_world_target_wp

    def solve(
        self,
        T_world_human: wp.array,
        valid_lengths: Optional[wp.array] = None,
        human_heights: Optional[wp.array] = None,
        *,
        out: Optional[wp.array] = None,
        target_overrides: Optional[Dict[str, np.ndarray]] = None,
    ) -> wp.array:
        """Retarget one padded `[B, T, J]` Warp trajectory batch.

        Lifecycle:
            1. Validate the shape fixed by `warmup(...)`.
            2. Compute all frame targets and independent IK warm starts.
            3. Reset the cached trajectory state and valid-frame masks.
            4. Solve once and write the requested padded output buffer.

        Args:
            T_world_human: Human joint transforms with shape `(batch, frames, joints)`.
            valid_lengths: Optional valid frame count for each padded batch element.
            human_heights: Optional human height in meters for each batch element.
            out: Optional caller-owned output with shape `(batch, frames, 7 + dofs)`.
            target_overrides: Optional `(batch, frames, 7)` target tracks keyed by robot link; they move the IK warm start and `TARGET_POSITIONS`, while orientation templates read `TARGETS` (the mapped poses) unless they read `OVERRIDDEN_TARGETS`.

        Returns:
            `out` when provided; otherwise a reusable helper-owned buffer overwritten by the next call.
        """
        # --- validate inputs ---
        if self._shape is None:
            raise RuntimeError("call warmup() before solve()")
        batch, length = cast(Tuple[int, int], self._shape)
        expected_shape = (batch, length, len(self.human_joint_names))
        if T_world_human.shape != expected_shape:
            raise ValueError(f"expected T_world_human shape {expected_shape}, got {T_world_human.shape}")
        lengths = self._default_valid_lengths_wp if valid_lengths is None else valid_lengths
        heights = self._default_human_heights_wp if human_heights is None else human_heights
        if lengths.shape != (batch,):
            raise ValueError(f"expected valid_lengths shape ({batch},), got {lengths.shape}")
        if heights.shape != (batch,):
            raise ValueError(f"expected human_heights shape ({batch},), got {heights.shape}")

        # --- compute targets ---
        self._compute_targets(T_world_human, heights)
        if target_overrides:
            target_names = list(self.retargeting_config.link_mapping)
            targets = self._target_buf_wp.numpy()
            for name, track in target_overrides.items():
                if name not in target_names:
                    raise ValueError(f"unknown target link: {name}")
                if track.shape != (batch, length, 7):
                    raise ValueError(f"expected override {name!r} shape {(batch, length, 7)}, got {track.shape}")
                targets[:, :, target_names.index(name)] = track
            self._target_buf_wp.assign(targets)

        # --- compute IK warm starts ---
        ik_state = self._ik.solve(self._T_world_target_wp, init_state=self._ik_init_state)

        # --- reset trajectory state ---
        for task in self._tasks:
            task.set_valid_lengths(lengths)
        wp.copy(
            self._state.q,
            ik_state.q.reshape((batch, length, self.robot.spec.num_actuated_joints)),
        )
        wp.copy(
            self._state.T_world_base,
            self._T_world_base_target_wp.reshape((batch, length)),
        )
        self._state.invalidate()

        # --- solve and finalize ---
        optimized_var, _ = self._optimizer.solve(self._var)
        optimized_state = cast(RobotState, optimized_var.get(VarKey.ROBOT))
        output = self._output_wp if out is None else out
        wp.launch(
            set_trajectory_solution_kernel,
            dim=(batch, 7 + self.robot.spec.num_actuated_joints),
            inputs=[
                optimized_state.q,
                optimized_state.T_world_base,
                self._joint_limits_wp,
                self._max_dq_wp,
                output,
                1 if self.retargeting_config.velocity_clamp_scale > 0 else 0,
            ],
            device=self._wp_device,
        )
        return output

    def solve_numpy(
        self,
        T_world_human: np.ndarray,
        valid_lengths: Optional[np.ndarray] = None,
        human_heights: Optional[np.ndarray] = None,
        *,
        target_overrides: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Convert padded `[B, T, J, 7]` NumPy inputs, warm up for their shape, solve, and return a NumPy copy."""
        motions = np.ascontiguousarray(T_world_human, dtype=np.float32)
        if motions.ndim != 4 or motions.shape[-1] != 7:
            raise ValueError("T_world_human must have shape [B, T, J, 7]")
        batch, length, joints, _ = motions.shape
        if joints != len(self.human_joint_names):
            raise ValueError(f"expected J={len(self.human_joint_names)}, got {joints}")
        lengths = (
            np.full(batch, length, dtype=np.int32)
            if valid_lengths is None
            else np.asarray(valid_lengths, dtype=np.int32)
        )
        heights = (
            np.full(batch, self.retargeting_config.human_height_assumption, dtype=np.float32)
            if human_heights is None
            else np.asarray(human_heights, dtype=np.float32)
        )
        if lengths.shape != (batch,) or not np.all((1 <= lengths) & (lengths <= length)):
            raise ValueError(f"valid_lengths must contain {batch} values in [1, {length}]")
        if heights.shape != (batch,) or np.any(heights <= 0):
            raise ValueError(f"human_heights must contain {batch} positive values")
        self.warmup(batch, length)
        output = self.solve(
            wp.from_numpy(motions, dtype=wp_vec7, device=self._wp_device),
            wp.from_numpy(lengths, dtype=wp.int32, device=self._wp_device),
            wp.from_numpy(heights, dtype=wp.float32, device=self._wp_device),
            target_overrides=target_overrides,
        )
        return output.numpy().copy()


__all__ = ["HumanoidRetargeterOffline"]
