# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIndexIssue=false
"""Whole-trajectory hand retargeting from named point targets."""

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, cast

import numpy as np
import warp as wp

from robokit.helpers.hand_retargeting._kernels import (
    convert_root_quaternions_kernel,
    set_default_trajectory_state_kernel,
    set_trajectory_solution_kernel,
)
from robokit.helpers.hand_retargeting.config import HandContext, HandRetargeterOfflineConfig, HandSpec
from robokit.helpers.hand_retargeting.kinematics_utils import build_retargeting_pairs
from robokit.lie.se3 import se3_identity
from robokit.opt.lm_optimizer import LMOptimizer
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.task import ResidualTask
from robokit.utils.ee_coord_utils import ee_coord_conversion
from robokit.utils.warp_utils import resolve_device, wp_device_type, wp_vec7
from robokit.xform.numpy.rotation_conversions import matrix_to_quaternion


class HandRetargeterOffline:
    """Retarget trajectory batches in `HandSpec.target_names` order by solving `config.terms` over `HandContext` inputs.

    Args:
        robot: Loaded robot model.
        spec: Ordered semantic point topology.
        config: Trajectory, contact, and solver settings.
        scene: Optional caller-owned collision scene.
        device: Warp device. The first CUDA device is used when available.
        scene_indices: Optional static `(batch,)` or per-frame `(batch, frames)` scene mapping.
    """

    def __init__(
        self,
        config: HandRetargeterOfflineConfig,
        robot: Robot,
        spec: HandSpec,
        scene: Optional["WarpScene"] = None,  # noqa: F821
        device: Optional[wp_device_type] = None,
        scene_indices: Optional[wp.array] = None,
    ) -> None:
        self.robot = robot
        self.spec = spec
        self.config = config
        self.scene = scene
        self.device = resolve_device(device)
        self.scene_indices = scene_indices
        self._shape: Optional[Tuple[int, int]] = None
        self._num_dofs = self.robot.spec.num_actuated_joints

        target_index = {name: index for index, name in enumerate(spec.target_names)}
        link_index = {name: index for index, name in enumerate(self.robot.spec.link_names)}
        selected_names = [name for name in spec.target_names if name in spec.target_link_names]
        self._target_indices = [target_index[name] for name in selected_names]
        self._link_indices = [link_index[spec.target_link_names[name]] for name in selected_names]
        self._root_target_index = target_index[spec.root_target_name]
        self._root_link_index = link_index[spec.target_link_names[spec.root_target_name]]
        self._contact_target_indices = [target_index[name] for name in spec.contact_target_names]
        self._contact_link_indices = np.asarray(
            [link_index[spec.target_link_names[name]] for name in spec.contact_target_names], dtype=np.int32
        )
        self._contact_sphere_indices = np.asarray(spec.contact_sphere_indices, dtype=np.int32)
        self._pair_indices = build_retargeting_pairs(spec, selected_names, include_root=True)
        needs = set().union(*(term.needs() for term in config.terms if term.active))
        self._needs_root_pose = HandContext.ROOT_POSE in needs
        if (
            HandContext.CONTACT_SPHERES in needs
            and self._contact_sphere_indices.shape != self._contact_link_indices.shape
        ):
            raise ValueError("contact_sphere_indices must contain one sphere per contact target.")

        joint_index = {name: index for index, name in enumerate(self.robot.spec.actuated_joint_names)}
        self._active_dofs = (
            None
            if config.active_joint_names is None
            else np.asarray([joint_index[name] for name in config.active_joint_names], dtype=np.int32)
        )

        root_ancestors = self.robot.spec.link_ancestor_joints_mask[self._root_link_index]
        hand_joints = self.robot.spec.link_ancestor_joints_mask[self._link_indices].any(axis=0) & ~root_ancestors
        hand_mapping = self.robot.spec.joints_to_actuated_mapping[hand_joints]
        hand_dofs = np.flatnonzero(np.any(hand_mapping != 0.0, axis=0)).astype(np.int32)
        # rest pulls the hand's own joints on a fixed-base arm, every joint otherwise
        self._rest_mask_np = np.ones(self._num_dofs, dtype=np.float32)
        if not spec.floating_base and len(hand_dofs) != self._num_dofs:
            self._rest_mask_np.fill(0.0)
            self._rest_mask_np[hand_dofs] = 1.0

        limits = self.robot.spec.actuated_joint_limits.astype(np.float32, copy=False)
        self._q_rest_np = np.clip(self.robot.spec.zero_q.astype(np.float32, copy=True), limits[:, 0], limits[:, 1])
        for name, value in spec.rest_q_by_name.items():
            self._q_rest_np[joint_index[name]] = value
        self._q_init_np = self._q_rest_np.copy()
        if hand_dofs.size:
            self._q_init_np[hand_dofs] = np.mean(limits[hand_dofs], axis=1)
        for name, value in spec.init_q_by_name.items():
            self._q_init_np[joint_index[name]] = value
        self._q_init_np = np.clip(self._q_init_np, limits[:, 0], limits[:, 1])
        self._joint_limits_wp = self.robot.spec.get_tensors(str(self.device)).actuated_joint_limits
        self._q_init_wp = wp.from_numpy(self._q_init_np, dtype=wp.float32, device=self.device)

        R_root_target = ee_coord_conversion(spec.target_coord_spec, spec.root_link_coord_spec)
        self._q_wxyz_target_root = wp.vec4(*matrix_to_quaternion(R_root_target).astype(np.float32))
        self._T_base_root_rest_wp = wp_vec7(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        if spec.floating_base:
            rest_state = self.robot.state(q=wp.from_numpy(self._q_rest_np[None], dtype=wp.float32, device=self.device))
            self.robot.forward_kinematics(rest_state)
            T_base_root_rest = cast(wp.array, rest_state.T_world_link[:, self._root_link_index]).numpy()[0]
            self._T_base_root_rest_wp = wp_vec7(*T_base_root_rest.astype(np.float32))

        # collision guards the contact fingers' branches
        self._collision_sphere_indices = np.empty(0, dtype=np.int32)
        if len(self._contact_link_indices) and scene is not None:
            contact_links = np.unique(self._contact_link_indices)
            contact_branches = self.robot.spec.link_ancestor_links_mask[contact_links].any(axis=0)
            contact_branches &= ~self.robot.spec.link_ancestor_links_mask[contact_links].all(axis=0)
            contact_branches[contact_links] = True
            self._collision_sphere_indices = np.flatnonzero(
                contact_branches[self.robot.spec.collision_spheres_link_indices]
            )

    def warmup(self, batch_size: int, num_frames: int) -> None:
        """Build buffers and the config's tasks, and capture the solver for one trajectory shape.

        Args:
            batch_size: Number of independent trajectories.
            num_frames: Frames in each fixed-shape trajectory.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if num_frames < 3:
            raise ValueError("num_frames must be at least 3.")
        if self._shape == (batch_size, num_frames):
            return
        self._shape = None
        config, device, num_contacts = self.config, self.device, len(self._contact_link_indices)
        flat = batch_size * num_frames

        self._target_points_wp = wp.empty(
            (batch_size, num_frames, len(self.spec.target_names), 3), dtype=wp.float32, device=device
        )
        self._valid_frame_mask_wp = wp.ones((batch_size, num_frames), dtype=wp.uint8, device=device)
        self._valid_frames_wp = wp.ones((batch_size, num_frames, 1), dtype=wp.bool, device=device)
        self._target_scales_wp = wp.full(batch_size, config.target_scale, dtype=wp.float32, device=device)
        # These views alias one xyz-wxyz root target buffer.
        T_world_root = np.zeros((batch_size, num_frames, 7), dtype=np.float32)
        T_world_root[:, :, 3] = 1.0
        self._T_world_root_wp = wp.from_numpy(T_world_root, dtype=wp.float32, device=device)
        self._q_wxyz_world_root_wp = cast(wp.array, self._T_world_root_wp[:, :, 3:7])
        root_pose = wp.array(ptr=self._T_world_root_wp.ptr, shape=(flat, 1), dtype=wp_vec7, device=device, copy=False)
        q = wp.from_numpy(np.tile(self._q_init_np, (batch_size, num_frames, 1)), dtype=wp.float32, device=device)
        T_world_base = se3_identity((batch_size, num_frames), device) if self.spec.floating_base else None
        self._state = self.robot.state(q=q, T_world_base=T_world_base)
        self._var = VarValues({VarKey.ROBOT: self._state})
        self._output_wp = wp.empty((batch_size, num_frames, 7 + self._num_dofs), dtype=wp.float32, device=device)
        self._reference_q_wp = wp.zeros((batch_size, num_frames, self._num_dofs), dtype=wp.float32, device=device)
        self._contact_points_wp = wp.zeros((batch_size, num_frames, num_contacts, 3), dtype=wp.float32, device=device)
        self._local_contact_points_wp = wp.zeros_like(self._contact_points_wp)
        self._contact_mask_wp = wp.zeros((batch_size, num_frames, num_contacts), dtype=wp.bool, device=device)
        self._surface_contact_mask_wp = wp.zeros_like(self._contact_mask_wp)
        self._anchor_q_wp = wp.zeros((flat, self._num_dofs), dtype=wp.float32, device=device)
        self._anchor_base_wp = se3_identity((flat,), device)

        ctx = HandContext
        context = {
            ctx.POINTS: self._target_points_wp,
            ctx.REST_Q: self._q_rest_np,
            ctx.REST_MASK: self._rest_mask_np,
            ctx.TARGET_LINKS: self._link_indices,
            ctx.TARGET_POINTS: self._target_indices,
            ctx.PAIRS: self._pair_indices,
            ctx.TARGET_SCALES: self._target_scales_wp,
            ctx.CONTACT_LINKS: self._contact_link_indices,
            ctx.CONTACT_TARGETS: self._contact_target_indices,
            ctx.ROOT_LINK: self._root_link_index,
            ctx.ROOT_TARGET: self._root_target_index,
            ctx.ROOT_POSE: root_pose,
            ctx.REFERENCE_Q: self._reference_q_wp,
            ctx.CONTACT_POINTS: self._contact_points_wp,
            ctx.CONTACT_MASK: self._contact_mask_wp,
            ctx.LOCAL_CONTACT_POINTS: self._local_contact_points_wp,
            ctx.SURFACE_CONTACT_MASK: self._surface_contact_mask_wp,
            ctx.CONTACT_SPHERES: self._contact_sphere_indices,
            ctx.COLLISION_SPHERES: self._collision_sphere_indices,
            ctx.SCENE: self.scene,
            ctx.SCENE_INDICES: self.scene_indices,
            ctx.VALID_FRAMES: self._valid_frames_wp,
            ctx.ANCHOR_Q: self._anchor_q_wp,
            ctx.ANCHOR_BASE: self._anchor_base_wp,
        }
        tasks: List[ResidualTask] = []
        for template in config.terms:
            if template.active:
                task = copy.copy(template)
                task.resolve(self._var, context)
                if task.active:  # building can still empty it, e.g. contacts on a spec without contact targets
                    tasks.append(task)
        if not tasks:
            raise ValueError("The config has no active task.")
        for task in tasks:  # padded frames add no cost
            task.frame_mask, task.use_mask = self._valid_frame_mask_wp, True

        # the config's base, joint, and prefix settings choose the optimized tangent entries
        base_dim = 6 if self.spec.floating_base else 0
        mask = np.ones((num_frames, base_dim + self._num_dofs), dtype=np.float32)
        mask[:, :base_dim] *= float(config.optimize_base)
        if self._active_dofs is not None:
            joint_mask = np.zeros(self._num_dofs, dtype=np.float32)
            joint_mask[self._active_dofs] = 1.0
            mask[:, base_dim:] *= joint_mask
        mask[: config.locked_prefix_frames, base_dim:] = 0.0
        active_mask = None if np.all(mask == 1.0) else wp.from_numpy(mask.reshape(-1), dtype=wp.float32, device=device)
        self._optimizer = LMOptimizer(tasks, replace(config.solver, early_stopping_interval=0), device, active_mask)
        self._shape = (batch_size, num_frames)
        self.set_valid_lengths(np.full(batch_size, num_frames, dtype=np.int32))

        self._target_points_wp.zero_()
        self._optimizer.solve(self._var)
        wp.synchronize_device(device)

    def set_valid_lengths(self, lengths: np.ndarray) -> None:
        """Mask padded frames and temporal edges in every task."""
        if self._shape is None:
            raise RuntimeError("call warmup(batch_size, num_frames) first.")
        batch_size, num_frames = self._shape
        lengths = np.asarray(lengths, dtype=np.int32)
        if lengths.shape != (batch_size,) or np.any((lengths < 3) | (lengths > num_frames)):
            raise ValueError(f"valid lengths must have shape ({batch_size},) and lie in [3, {num_frames}].")
        frame_mask = np.arange(num_frames)[None, :] < lengths[:, None]
        self._valid_frame_mask_wp.assign(frame_mask.astype(np.uint8))
        self._valid_frames_wp.assign(frame_mask[:, :, None])

    def set_target_scales(self, scales: np.ndarray) -> None:
        """Set one positive pairwise target scale per warmed trajectory without rebuilding the solver.

        Example:
            >>> # xdoctest: +SKIP
            >>> from robokit.helpers.full_body_retargeting import FullBodyRetargeter
            >>> from robokit.helpers.hand_retargeting.presets.inspire import offline, spec
            >>> solver = HandRetargeterOffline(offline, FullBodyRetargeter().hand_robot("right"), spec)
            >>> solver.warmup(2, 3)
            >>> solver.set_target_scales(np.array([0.9, 1.2]))
        """
        if self._shape is None:
            raise RuntimeError("call warmup(batch_size, num_frames) first.")
        scales = np.asarray(scales, dtype=np.float32)
        if scales.shape != (self._shape[0],) or not np.all(np.isfinite(scales) & (scales > 0)):
            raise ValueError(f"target scales must be positive and have shape ({self._shape[0]},).")
        wp.copy(self._target_scales_wp, wp.from_numpy(scales, dtype=wp.float32, device=self.device))

    def solve(
        self,
        target_points: wp.array,
        root_quat_wxyz: Optional[wp.array] = None,
        init_state: Optional[RobotState] = None,
        out: Optional[wp.array] = None,
    ) -> wp.array:
        """Retarget one Warp trajectory batch.

        Args:
            target_points: Float points shaped (batch, frames, targets, 3) in spec.target_names order.
            root_quat_wxyz: Optional root rotations shaped (batch, frames, 4).
            init_state: Optional joint and base warm start.
            out: Optional packed output shaped (batch, frames, 7 + dofs).

        Returns:
            Packed base transforms and joint positions.
        """
        if self._needs_root_pose and root_quat_wxyz is None:
            raise ValueError("root_quat_wxyz is required when a task reads the root pose.")
        self._contact_mask_wp.zero_()
        self._surface_contact_mask_wp.zero_()
        return self._solve(target_points, root_quat_wxyz, init_state, out, self._T_base_root_rest_wp)

    def solve_numpy(
        self,
        target_points: np.ndarray,
        root_quat_wxyz: Optional[np.ndarray] = None,
        init_state: Optional[RobotState] = None,
    ) -> np.ndarray:
        """NumPy wrapper for `solve()`."""
        points = np.asarray(target_points, dtype=np.float32)
        squeeze_batch = points.ndim == 3
        if squeeze_batch:
            points = points[None]
        batch_size, num_frames = points.shape[:2]
        quaternions = None if root_quat_wxyz is None else np.asarray(root_quat_wxyz, dtype=np.float32)
        if quaternions is not None and squeeze_batch and quaternions.ndim == 2:
            quaternions = quaternions[None]
        self.warmup(batch_size, num_frames)
        result = (
            self.solve(
                wp.from_numpy(np.ascontiguousarray(points), dtype=wp.float32, device=self.device),
                (
                    wp.from_numpy(np.ascontiguousarray(quaternions), dtype=wp.float32, device=self.device)
                    if quaternions is not None
                    else None
                ),
                init_state,
            )
            .numpy()
            .copy()
        )
        return result[0] if squeeze_batch else result

    def solve_with_contact(
        self,
        target_points: wp.array,
        contact_points: wp.array,
        contact_mask: wp.array,
        init_state: Optional[RobotState] = None,
        local_contact_points: Optional[wp.array] = None,
        out: Optional[wp.array] = None,
        surface_contact_mask: Optional[wp.array] = None,
    ) -> wp.array:
        """Retarget with world-frame contacts; an `init_state` anchors the solution and its motion to that trajectory.

        Args:
            target_points: Named points shaped (batch, frames, targets, 3).
            contact_points: World contacts shaped (batch, frames, contacts, 3).
            contact_mask: Active contacts shaped (batch, frames, contacts).
            init_state: Optional state that anchors the solution around that trajectory.
            local_contact_points: Optional contacts in each link frame.
            out: Optional caller-owned packed output.
            surface_contact_mask: Optional mask for sphere-surface contacts; uses `contact_mask` when omitted.

        Returns:
            Packed base transforms and joint positions.
        """
        expected = self._contact_points_wp.shape
        if contact_points.shape != expected or contact_mask.shape != expected[:-1]:
            raise ValueError(f"contact_points must have shape {expected} and contact_mask {expected[:-1]}.")
        if local_contact_points is not None and local_contact_points.shape != expected:
            raise ValueError(f"local_contact_points must have shape {expected}.")
        wp.copy(self._contact_points_wp, contact_points)
        wp.copy(self._contact_mask_wp, contact_mask)
        wp.copy(self._surface_contact_mask_wp, contact_mask if surface_contact_mask is None else surface_contact_mask)
        if local_contact_points is None:
            self._local_contact_points_wp.zero_()
        else:
            wp.copy(self._local_contact_points_wp, local_contact_points)
        identity = wp_vec7(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        return self._solve(target_points, None, init_state, out, identity, follow_init=init_state is not None)

    def _solve(
        self,
        target_points: wp.array,
        root_quat_wxyz: Optional[wp.array],
        init_state: Optional[RobotState],
        out: Optional[wp.array],
        T_base_root_rest: Any,
        follow_init: bool = False,
    ) -> wp.array:
        """Fill the inputs, start from `init_state` or the default trajectory, anchor to that start, and solve."""
        batch_size, num_frames = self._validate_inputs(target_points, init_state, out)
        if root_quat_wxyz is not None:
            expected_quaternions = (batch_size, num_frames, 4)
            if (root_quat_wxyz.shape, root_quat_wxyz.dtype, root_quat_wxyz.device) != (
                expected_quaternions,
                wp.float32,
                self.device,
            ):
                raise ValueError(f"root_quat_wxyz must be float32 on {self.device} with shape {expected_quaternions}.")
            wp.launch(
                convert_root_quaternions_kernel,
                dim=(batch_size, num_frames),
                inputs=[root_quat_wxyz, self._q_wxyz_target_root, self._q_wxyz_world_root_wp],
                device=self.device,
            )
        wp.copy(self._target_points_wp, target_points)
        if init_state is None:
            wp.launch(
                set_default_trajectory_state_kernel,
                dim=(batch_size, num_frames),
                inputs=[
                    self._target_points_wp,
                    self._root_target_index,
                    self._q_wxyz_world_root_wp,
                    int(root_quat_wxyz is not None),
                    T_base_root_rest,
                    int(self.spec.floating_base),
                    self._q_init_wp,
                    self._state.q,
                    self._state.T_world_base,
                ],
                device=self.device,
            )
        else:
            wp.copy(self._state.q, init_state.q)
            wp.copy(self._state.T_world_base, init_state.T_world_base)
        self._state.invalidate()
        # smoothness follows an anchored start's motion; anchors hold the start
        if follow_init:
            wp.copy(self._reference_q_wp, self._state.q)
        else:
            self._reference_q_wp.zero_()
        wp.copy(self._anchor_q_wp, self._state.q.reshape(self._anchor_q_wp.shape))
        wp.copy(self._anchor_base_wp, self._state.T_world_base.flatten())

        optimized_var, _ = self._optimizer.solve(self._var)
        return self._write_output(cast(RobotState, optimized_var.get(VarKey.ROBOT)), out)

    def _validate_inputs(
        self,
        target_points: wp.array,
        init_state: Optional[RobotState],
        out: Optional[wp.array],
    ) -> Tuple[int, int]:
        if self._shape is None:
            raise RuntimeError("call warmup(batch_size, num_frames) before solving.")
        batch_size, num_frames = self._shape
        expected_points = (batch_size, num_frames, len(self.spec.target_names), 3)
        if (target_points.shape, target_points.dtype, target_points.device) != (
            expected_points,
            wp.float32,
            self.device,
        ):
            raise ValueError(f"target_points must be float32 on {self.device} with shape {expected_points}.")
        if init_state is not None:
            expected_state = (batch_size, num_frames, self._num_dofs), (batch_size, num_frames), self.device
            if (init_state.q.shape, init_state.T_world_base.shape, init_state.device) != expected_state:
                raise ValueError("init_state must match the warmed trajectory shape, robot DOFs, and device.")
        expected_output = (batch_size, num_frames, 7 + self._num_dofs)
        if out is not None and (out.shape, out.dtype, out.device) != (expected_output, wp.float32, self.device):
            raise ValueError(f"out must be float32 on {self.device} with shape {expected_output}.")
        return batch_size, num_frames

    def _write_output(self, state: RobotState, out: Optional[wp.array]) -> wp.array:
        output = self._output_wp if out is None else out
        wp.launch(
            set_trajectory_solution_kernel,
            dim=output.shape,
            inputs=[state.q, state.T_world_base, self._joint_limits_wp, output],
            device=self.device,
        )
        return output

    def solve_with_contact_numpy(
        self,
        target_points: np.ndarray,
        contact_points: np.ndarray,
        contact_mask: np.ndarray,
        init_state: Optional[RobotState] = None,
        local_contact_points: Optional[np.ndarray] = None,
        surface_contact_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """NumPy wrapper for `solve_with_contact()`."""
        points = np.asarray(target_points, dtype=np.float32)
        squeeze_batch = points.ndim == 3
        if squeeze_batch:
            points = points[None]
        batch_size, num_frames = points.shape[:2]
        contacts = np.asarray(contact_points, dtype=np.float32)
        masks = np.asarray(contact_mask, dtype=np.bool_)
        local_contacts = None if local_contact_points is None else np.asarray(local_contact_points, dtype=np.float32)
        surface_masks = None if surface_contact_mask is None else np.asarray(surface_contact_mask, dtype=np.bool_)
        if squeeze_batch:
            if contacts.ndim == 3:
                contacts = contacts[None]
            if masks.ndim == 2:
                masks = masks[None]
            if local_contacts is not None and local_contacts.ndim == 3:
                local_contacts = local_contacts[None]
            if surface_masks is not None and surface_masks.ndim == 2:
                surface_masks = surface_masks[None]
        self.warmup(batch_size, num_frames)
        result = (
            self.solve_with_contact(
                wp.from_numpy(np.ascontiguousarray(points), dtype=wp.float32, device=self.device),
                wp.from_numpy(np.ascontiguousarray(contacts), dtype=wp.float32, device=self.device),
                wp.from_numpy(np.ascontiguousarray(masks), dtype=wp.bool, device=self.device),
                init_state,
                (
                    wp.from_numpy(np.ascontiguousarray(local_contacts), dtype=wp.float32, device=self.device)
                    if local_contacts is not None
                    else None
                ),
                surface_contact_mask=(
                    wp.from_numpy(np.ascontiguousarray(surface_masks), dtype=wp.bool, device=self.device)
                    if surface_masks is not None
                    else None
                ),
            )
            .numpy()
            .copy()
        )
        return result[0] if squeeze_batch else result


if TYPE_CHECKING:
    from robokit.geom import WarpScene


__all__ = ["HandRetargeterOffline"]
