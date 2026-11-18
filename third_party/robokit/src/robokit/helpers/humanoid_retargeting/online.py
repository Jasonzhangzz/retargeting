"""Dense-array online humanoid retargeting."""

import copy
from typing import Any, Dict, List, Optional, cast

import numpy as np
import warp as wp

from robokit.geom import MeshGeom, WarpScene
from robokit.helpers.humanoid_retargeting._kernels import (
    compute_init_states_kernel,
    compute_targets_kernel,
    set_solution_state_kernel,
)
from robokit.helpers.humanoid_retargeting.config import HumanoidContext, HumanoidRetargeterOnlineConfig
from robokit.lie.se3 import se3_identity
from robokit.opt.population_solver import PopulationSolver
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.dense.interaction_mesh_task import InteractionMeshTask
from robokit.terms.task import Context, ResidualTask
from robokit.utils.warp_utils import resolve_device, wp_device_type, wp_vec7


class HumanoidRetargeterOnline:
    """Online humanoid retargeting with input joints ordered by `human_joint_names`.

    Lifecycle:
        1. `__init__` stores the robot, configuration, and caller-owned scene.
        2. `warmup(batch_size)` builds batch-shaped buffers and solvers.
        3. `solve(T_world_human)` advances one batched frame from Warp inputs.
    """

    def __init__(
        self,
        config: HumanoidRetargeterOnlineConfig,
        robot: Optional[Robot] = None,
        scene: Optional[WarpScene] = None,
        device: Optional[wp_device_type] = None,
        seed: int = 0,
    ):
        self.config = config
        self.robot = robot if robot is not None else config.load_robot()
        self.device = resolve_device(device)
        self._scene = scene
        if scene is not None and scene.device != self.device:
            raise ValueError(f"scene must be on {self.device}, got {scene.device}")
        needs = set().union(*(term.needs() for term in [*config.terms, *config.interaction_terms] if term.active))
        if HumanoidContext.SCENE in needs and scene is None:
            raise ValueError("a weighted scene collision task requires scene")
        self._seed = seed
        self._num_candidates0 = config.solver.num_candidates[0]
        self._num_dofs = self.robot.spec.num_actuated_joints
        self.link_name_to_index = {name: i for i, name in enumerate(self.robot.spec.link_names)}
        assert config.solver.num_candidates[-1] == 1, "the schedule must end with num_candidates=1"
        if config.interaction_terms:
            assert list(config.solver.num_candidates) == [1], (
                "interaction retargeting currently requires one stage with one candidate"
            )

        self.human_joint_names = config.human_joint_names
        joint_index = {name: i for i, name in enumerate(self.human_joint_names)}
        self._targets = config.target_arrays(self.human_joint_names)
        self._num_targets = len(self._targets.joint_indices)

        edges = config.correspondence_edges or []
        self._corr_origin_indices_np = np.array([joint_index[e.origin_joint] for e in edges], dtype=np.int32)
        self._corr_task_indices_np = np.array([joint_index[e.task_joint] for e in edges], dtype=np.int32)
        self._corr_origin_scales_np = np.array(
            [config.scale_table.get(e.origin_joint, 1.0) for e in edges], dtype=np.float32
        )
        self._corr_task_scales_np = np.array(
            [config.scale_table.get(e.task_joint, 1.0) for e in edges], dtype=np.float32
        )
        self._num_vector_targets = len(edges)
        self._batch_size: Optional[int] = None

        self.link_indices = [self.link_name_to_index[name] for name in config.link_mapping]
        self._interaction_tasks: List[ResidualTask] = []
        if config.interaction_terms:
            if scene is None:
                raise ValueError("interaction retargeting requires scene")
            mesh_geoms = [geom for geom in scene.geoms if isinstance(geom, MeshGeom)]
            if len(mesh_geoms) != 1 or mesh_geoms[0].count != 1:
                raise ValueError("interaction retargeting requires one scene mesh")
            # built now: the caller sets the first frame before warmup
            state = self.robot.state(q=wp.zeros((1, self._num_dofs), dtype=wp.float32, device=self.device))
            for template in config.interaction_terms:
                task = copy.copy(template)
                task.resolve(VarValues({VarKey.ROBOT: state}), {HumanoidContext.TARGET_LINKS: self.link_indices})
                self._interaction_tasks.append(task)

    @property
    def interaction_task(self) -> InteractionMeshTask:
        """Return the interaction task; set its first frame before warmup and update it before each solve."""
        if not self._interaction_tasks:
            raise RuntimeError("interaction is not enabled")
        return cast(InteractionMeshTask, self._interaction_tasks[0])

    def _build_solver(self, batch_size: int):
        """Build one copy of every template per solver stage on the shared input buffers."""
        device = self.device
        self._targets_wp = se3_identity(shape=(batch_size, len(self.link_indices)), device=device)
        self._target_vectors_wp = wp.zeros(
            (batch_size, max(self._num_vector_targets, 1), 3), dtype=wp.float32, device=device
        )
        limits = self.robot.spec.actuated_joint_limits
        # zero_q may sit outside asymmetric joint limits; clamp so frame-0 candidates start legal
        self._zero_q_wp = wp.from_numpy(
            np.tile(np.clip(self.robot.spec.zero_q, limits[:, 0], limits[:, 1]).astype(np.float32), (batch_size, 1)),
            dtype=wp.float32,
            device=device,
        )
        reset_base_np = np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), (batch_size, 1))
        self._reset_base_wp = wp.from_numpy(reset_base_np, dtype=wp_vec7, device=device)
        self._prev_q_wp = wp.clone(self._zero_q_wp)
        self._prev_state_base = wp.clone(self._reset_base_wp)
        self._prev_state = self.robot.state(q=self._prev_q_wp, T_world_base=self._prev_state_base)
        self._frame_wp = wp.zeros(1, dtype=wp.int32, device=device)
        self._velocity_limits_np = (
            np.array(self.config.velocity_limit_override, dtype=np.float32)
            if self.config.velocity_limit_override is not None
            else self.robot.spec.actuated_joint_velocity_limits.astype(np.float32, copy=False)
        )

        first_stage_candidates = self.config.solver.num_candidates[0]
        self._initial_state = self.robot.state(
            q=wp.zeros((batch_size * first_stage_candidates, self._num_dofs), dtype=wp.float32, device=device),
            T_world_base=se3_identity(shape=(batch_size * first_stage_candidates,), device=device),
        )
        self._initial_var = VarValues({VarKey.ROBOT: self._initial_state})
        edges = self.config.correspondence_edges or []
        ctx = HumanoidContext
        context: Dict[Context, Any] = {
            ctx.TARGET_LINKS: self.link_indices,
            ctx.TARGETS: self._targets_wp,
            ctx.VELOCITY_LIMITS: self._velocity_limits_np,
            ctx.STEP: self.config.velocity_limit_dt,
            ctx.PREVIOUS_STATE: self._prev_state,
            ctx.FRAME: self._frame_wp,
            ctx.CORRESPONDENCE_ORIGINS: [self.link_name_to_index[edge.origin_link] for edge in edges],
            ctx.CORRESPONDENCE_LINKS: [self.link_name_to_index[edge.task_link] for edge in edges],
            ctx.VECTORS: self._target_vectors_wp,
            ctx.SCENE: self._scene,
        }
        stage_terms: List[List[ResidualTask]] = []
        for _ in self.config.solver.num_candidates:
            terms: List[ResidualTask] = []
            for template in self.config.terms:
                if template.active:
                    task = copy.copy(template)
                    task.resolve(self._initial_var, context)
                    if task.active:  # building can still empty it, e.g. correspondences without edges
                        terms.append(task)
            stage_terms.append([*terms, *self._interaction_tasks])
        self._solver = PopulationSolver(terms=stage_terms, config=self.config.solver, device=device, score_terms=None)
        self._solver.setup(self._initial_var)

    def warmup(self, batch_size: int):
        """Fix the online batch size and build its reusable solver state."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self._batch_size is not None:
            if batch_size != self._batch_size:
                raise ValueError(f"batch size mismatch: expected {self._batch_size}, got {batch_size}")
            return

        if self._interaction_tasks and batch_size != 1:
            raise ValueError("interaction retargeting supports batch_size=1")
        self._build_solver(batch_size)
        limits = self.robot.spec.actuated_joint_limits.astype(np.float32, copy=False)
        self._target_joint_indices_wp = wp.from_numpy(self._targets.joint_indices, dtype=wp.int32, device=self.device)
        self._base_scale_wp = wp.from_numpy(self._targets.scales, dtype=wp.float32, device=self.device)
        self._rot_off_wp = wp.from_numpy(self._targets.rotation_offsets, dtype=wp.vec4, device=self.device)
        self._pos_off_wp = wp.from_numpy(self._targets.position_offsets, dtype=wp.vec3, device=self.device)
        self._joint_limits_wp = wp.from_numpy(limits, dtype=wp.float32, device=self.device)
        self._midrange_q_wp = wp.from_numpy(
            self.robot.spec.midrange_q.astype(np.float32, copy=False), dtype=wp.float32, device=self.device
        )
        self._max_dq_wp = wp.from_numpy(
            self._velocity_limits_np * self.config.velocity_limit_dt * self.config.velocity_clamp_scale,
            dtype=wp.float32,
            device=self.device,
        )

        perturbations = np.zeros((batch_size, self._num_candidates0, self._num_dofs), dtype=np.float32)
        joint_range = limits[:, 1] - limits[:, 0]
        for b in range(batch_size):
            rng = np.random.default_rng(self._seed if b == 0 else (self._seed, b))
            perturbations[b, 1:] = (
                rng.uniform(-0.1, 0.1, (self._num_candidates0 - 1, self._num_dofs)).astype(np.float32) * joint_range
            )
        # the uniform table is the candidate noise; it lands on the joint block of the state tangent
        tangent = np.zeros((batch_size * self._num_candidates0, self._initial_state.tangent_dim), dtype=np.float32)
        tangent[:, -self._num_dofs :] = perturbations.reshape(-1, self._num_dofs)
        self._candidate_tangent = wp.from_numpy(tangent, dtype=wp.float32, device=self.device)
        self._n_local = (self._num_candidates0 - 1) // 2

        self._default_human_heights_wp = wp.from_numpy(
            np.full(batch_size, self.config.human_height_assumption, dtype=np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        self._output_wp = wp.zeros((batch_size, 7 + self._num_dofs), dtype=wp.float32, device=self.device)

        self._corr_origin_indices_wp = wp.from_numpy(self._corr_origin_indices_np, dtype=wp.int32, device=self.device)
        self._corr_task_indices_wp = wp.from_numpy(self._corr_task_indices_np, dtype=wp.int32, device=self.device)
        self._corr_origin_scales_wp = wp.from_numpy(self._corr_origin_scales_np, dtype=wp.float32, device=self.device)
        self._corr_task_scales_wp = wp.from_numpy(self._corr_task_scales_np, dtype=wp.float32, device=self.device)
        self._base_target_wp = wp.from_numpy(
            np.empty((batch_size, 7), dtype=np.float32), dtype=wp_vec7, device=self.device
        )
        self._candidate_base_wp = self._prev_state_base if self._interaction_tasks else self._base_target_wp

        self._batch_size = batch_size
        self._frame_count = 0
        self.reset()

    def solve(
        self,
        T_world_human: wp.array,
        human_heights: Optional[wp.array] = None,
        *,
        out: Optional[wp.array] = None,
    ) -> wp.array:
        """Retarget one batched human frame from Warp inputs.

        Lifecycle:
            1. Validate the shape fixed by `warmup(...)`.
            2. Update temporal terms from the previous robot state.
            3. Compute targets and candidate states.
            4. Solve once, retain the state, and write the requested output buffer.

        Args:
            T_world_human: Human joint transforms with shape `(batch, joints)`.
            human_heights: Optional human height in meters for each batch element.
            out: Optional caller-owned output with shape `(batch, 7 + dofs)`.

        Returns:
            `out` when provided; otherwise a reusable helper-owned buffer overwritten by the next call.
        """
        # --- validate inputs ---
        if self._batch_size is None:
            raise RuntimeError("call warmup() before solve()")
        expected_shape = (self._batch_size, len(self.human_joint_names))
        if T_world_human.shape != expected_shape:
            raise ValueError(f"expected T_world_human shape {expected_shape}, got {T_world_human.shape}")
        heights = self._default_human_heights_wp if human_heights is None else human_heights
        if heights.shape != (self._batch_size,):
            raise ValueError(f"expected human_heights shape ({self._batch_size},), got {heights.shape}")

        # --- compute targets; the tasks read these buffers and the previous state ---
        self._frame_wp.fill_(self._frame_count)
        wp.launch(
            compute_targets_kernel,
            dim=(self._batch_size, self._num_targets + self._num_vector_targets + 1),
            inputs=[
                T_world_human,
                heights,
                self._target_joint_indices_wp,
                self._base_scale_wp,
                self._rot_off_wp,
                self._pos_off_wp,
                self._corr_origin_indices_wp,
                self._corr_task_indices_wp,
                self._corr_origin_scales_wp,
                self._corr_task_scales_wp,
                self._targets_wp,
                self._base_target_wp,
                self._target_vectors_wp,
                self._num_targets,
                self._num_vector_targets,
                self._targets.root_joint_index,
                self._targets.root_target_index,
                self._targets.root_scale,
                float(self.config.human_height_assumption),
                1,
                float(self.config.ground_height),
            ],
            device=self.device,
        )

        # --- compute candidate states ---
        wp.launch(
            compute_init_states_kernel,
            dim=(self._batch_size, self._num_candidates0, self._num_dofs + 1),
            inputs=[
                self._prev_q_wp,
                self._midrange_q_wp,
                self._joint_limits_wp,
                self._candidate_tangent,
                self._candidate_base_wp,
                self._initial_state.q,
                self._initial_state.T_world_base,
                self._n_local,
                self._num_candidates0,
            ],
            device=self.device,
        )
        if self._interaction_tasks:
            # Resume interaction from the previous solution, replacing the clamped single candidate.
            wp.copy(self._initial_state.q, self._prev_q_wp)

        # --- solve and finalize ---
        self._initial_state.invalidate()
        best_var, _ = self._solver.solve(self._initial_var)
        best_state = cast(RobotState, best_var.get(VarKey.ROBOT))
        output = self._output_wp if out is None else out

        wp.launch(
            set_solution_state_kernel,
            dim=(self._batch_size, 7 + self._num_dofs),
            inputs=[
                best_state.q,
                best_state.T_world_base,
                self._joint_limits_wp,
                self._max_dq_wp,
                self._prev_q_wp,
                self._prev_state_base,
                output,
                1 if (self.config.velocity_clamp_scale > 0.0 and self._frame_count > 0) else 0,
            ],
            device=self.device,
        )
        self._frame_count += 1
        return output

    def solve_numpy(
        self,
        T_world_human: np.ndarray,
        human_heights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convert NumPy inputs, call `solve(...)`, and return a NumPy copy."""
        frames = np.ascontiguousarray(T_world_human, dtype=np.float32)
        if frames.ndim != 3 or frames.shape[-1] != 7:
            raise ValueError("T_world_human must have shape [B, J, 7]")
        batch_size, num_joints, _ = frames.shape
        if num_joints != len(self.human_joint_names):
            raise ValueError(f"expected J={len(self.human_joint_names)} in human_joint_names order, got {num_joints}")
        heights_wp = None
        if human_heights is not None:
            heights = np.asarray(human_heights, dtype=np.float32)
            if heights.shape != (batch_size,) or np.any(heights <= 0):
                raise ValueError(f"human_heights must contain {batch_size} positive values")
            heights_wp = wp.from_numpy(heights, dtype=wp.float32, device=self.device)
        return self.solve(wp.from_numpy(frames, dtype=wp_vec7, device=self.device), heights_wp).numpy().copy()

    def reset(self, T_world_base: Optional[wp.array] = None):
        """Reset temporal state, optionally from an initial floating-base transform."""
        self._frame_count = 0
        if self._batch_size is None:
            return
        wp.copy(self._prev_q_wp, self._zero_q_wp)
        wp.copy(self._prev_state_base, self._reset_base_wp if T_world_base is None else T_world_base)
        self._output_wp.zero_()


__all__ = ["HumanoidRetargeterOnline"]
