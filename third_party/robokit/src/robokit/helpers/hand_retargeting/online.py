# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportInvalidTypeForm=false
"""Stateful online hand retargeting from named point targets."""

import copy
from dataclasses import replace
from typing import Any, Dict, List, Optional, cast

import numpy as np
import warp as wp

from robokit.helpers.hand_retargeting._kernels import (
    center_base_kernel,
    center_points_kernel,
    compute_direction_targets_kernel,
    compute_vector_targets_kernel,
    restore_translation_kernel,
    set_online_solution_kernel,
)
from robokit.helpers.hand_retargeting.config import HandContext, HandRetargeterOnlineConfig, HandSpec
from robokit.lie.se3 import se3_identity
from robokit.opt.population_solver import PopulationSolver
from robokit.opt.sampler import RobertsSampler
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot
from robokit.robo.robot_state import RobotState
from robokit.terms.task import Context, ResidualTask
from robokit.utils.warp_utils import gather, resolve_device, wp_device_type


class HandRetargeterOnline:
    """Retarget point streams; `reference_origin="wrist"` centers floating hands internally, but I/O stays world-frame.

    Temporal cost:
    - First frame after warmup/reset: none.
    - Later frames: smooth against the previous solution, independently of the initial guess.

    Args:
        robot: Loaded robot model; self-collision skip pairs come from its spec (`Robot.load(self_collision_ignore_path=...)`).
        spec: Ordered semantic point topology.
        config: Online solver, objective, filtering, and sampling settings.
        device: Warp device. The first CUDA device is used when available.

    Example:
        >>> from robokit.robo.robot_spec import RobotSpec
        >>> from robokit.utils.ee_coord_utils import MANOPTH_HAND_COORD_SPEC as coords
        >>> from robokit.opt.population_solver import PopulationSolverConfig, Stage
        >>> from robokit.opt.lm_optimizer import LMOptimizerConfig
        >>> from robokit.utils.warp_utils import wp_vec7
        >>> urdf = '''<robot name="finger"><link name="palm"/><link name="tip"/>
        ... <joint name="finger" type="revolute"><parent link="palm"/><child link="tip"/>
        ... <origin xyz="0 0.03 0"/><axis xyz="0 0 1"/>
        ... <limit lower="0" upper="1" effort="1" velocity="1"/></joint></robot>'''
        >>> robot = Robot(RobotSpec.parse(urdf, load_meshes=False, mesh_dir=None))
        >>> spec = HandSpec(True, ("tip", "wrist"), {"tip": "tip", "wrist": "palm"},
        ...                 (("tip",),), "wrist", ("tip",), coords)
        >>> config = HandRetargeterOnlineConfig(reference_origin="wrist",
        ...     solver=PopulationSolverConfig(stages=[Stage(2, optimizer=LMOptimizerConfig())],
        ...         cuda_graph_mode="none"))
        >>> config.task("positions").weight = {"tip": 24.0, "wrist": 0.1}
        >>> solver = HandRetargeterOnline(config, robot, spec, device="cpu")
        >>> solver.warmup(1)
        >>> points = np.array([[[1, 2.03, 3], [1, 2, 3]]], dtype=np.float32)
        >>> packed = solver.solve_numpy(points)
        >>> packed.shape
        (1, 8)
        >>> output = wp.empty((1, 8), dtype=wp.float32, device="cpu")
        >>> solver.solve(wp.from_numpy(points, dtype=wp.float32, device="cpu"), out=output) is output
        True
        >>> state = robot.state(q=wp.from_numpy(packed[:, 7:].copy(), dtype=wp.float32, device="cpu"),
        ...     T_world_base=wp.from_numpy(packed[:, :7].copy(), dtype=wp_vec7, device="cpu"))
        >>> all(np.isfinite(cost.numpy()).all() for cost in solver.compute_costs(state).values())
        True
        >>> solver.reset()
    """

    def __init__(
        self,
        config: HandRetargeterOnlineConfig,
        robot: Robot,
        spec: HandSpec,
        device: Optional[wp_device_type] = None,
    ) -> None:
        self.robot = robot
        self.spec = spec
        self.config = config
        if config.reference_origin not in ("world", "wrist"):
            raise ValueError("reference_origin must be 'world' or 'wrist'.")
        self._wrist_origin = spec.floating_base and config.reference_origin == "wrist"
        self._root_index = spec.target_names.index(spec.root_target_name)
        self.device = resolve_device(device)
        self._num_dofs = self.robot.spec.num_actuated_joints
        self._batch_size: Optional[int] = None
        self._history_valid = False
        if not (0.0 < config.output_ema_alpha <= 1.0 and 0.0 < config.target_ema_alpha <= 1.0):
            raise ValueError("output_ema_alpha and target_ema_alpha must lie in (0, 1].")

        names = spec.target_names
        needs = set().union(*(term.needs() for term in config.terms if term.active))
        ctx = HandContext
        if not needs & {ctx.POINTS, ctx.VECTORS, ctx.DIRECTIONS}:
            raise ValueError("At least one position, vector, or direction target is required.")
        self._uses_vectors, self._uses_directions = ctx.VECTORS in needs, ctx.DIRECTIONS in needs

        # vector and direction targets cover every ordered target pair; tasks read the rows they weight
        count = len(names)
        origins, tasks = np.repeat(np.arange(count), count), np.tile(np.arange(count), count)
        self._num_pairs = count * count
        pinch_pairs = {(pair[2], pair[3]) for pair in config.pinch_correspondences}
        self._pair_np = {
            "origins": origins.astype(np.int32),
            "tasks": tasks.astype(np.int32),
            "pinch": np.asarray([(names[o], names[t]) in pinch_pairs for o, t in zip(origins, tasks)], dtype=np.bool_),
        }
        self._contact_targets = [names.index(name) for name in spec.contact_target_names]

        joint_index = {name: index for index, name in enumerate(self.robot.spec.actuated_joint_names)}
        self._reset_q_np = self.robot.spec.zero_q.astype(np.float32, copy=True)
        for name, value in spec.init_q_by_name.items():
            self._reset_q_np[joint_index[name]] = value
        self._rest_q_np = self.robot.spec.zero_q.astype(np.float32, copy=True)
        for name, value in spec.rest_q_by_name.items():
            self._rest_q_np[joint_index[name]] = value

    def warmup(self, batch_size: int) -> None:
        """Build all batch-shaped terms, buffers, and solver state.

        Args:
            batch_size: Number of independent streams advanced by each solve.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self._batch_size == batch_size:
            return
        self._batch_size = None

        device, pairs = self.device, self._num_pairs
        self._target_points_wp = wp.zeros((batch_size, len(self.spec.target_names), 3), dtype=wp.float32, device=device)
        if self._wrist_origin:
            self._input_points_wp = wp.zeros_like(self._target_points_wp)
            self._local_state = self.robot.state(
                q=wp.zeros((batch_size, self._num_dofs), dtype=wp.float32, device=device),
                T_world_base=se3_identity(shape=(batch_size,), device=device),
            )
        self._vector_targets_wp = wp.zeros((batch_size, pairs, 3), dtype=wp.float32, device=device)
        self._direction_targets_wp = wp.zeros((batch_size, pairs, 3), dtype=wp.float32, device=device)
        self._frame_wp = wp.zeros(1, dtype=wp.int32, device=device)
        reset_q = np.clip(
            self._reset_q_np,
            self.robot.spec.actuated_joint_limits[:, 0],
            self.robot.spec.actuated_joint_limits[:, 1],
        )
        self._reset_q_wp = wp.from_numpy(np.tile(reset_q, (batch_size, 1)), dtype=wp.float32, device=self.device)
        self._reset_base_wp = se3_identity(shape=(batch_size,), device=self.device)
        self._previous_state = self.robot.state(
            q=wp.clone(self._reset_q_wp),
            T_world_base=wp.clone(self._reset_base_wp) if self.spec.floating_base else None,
        )

        # smoothness reads the frame before the one being solved; the previous solution moves on after it
        self._smoothness_reference = self.robot.state(
            q=wp.clone(self._reset_q_wp),
            T_world_base=wp.clone(self._reset_base_wp) if self.spec.floating_base else None,
        )
        max_candidates = self.config.solver.num_candidates[0]
        expanded_batch_size = batch_size * max_candidates
        self._initial_state = self.robot.state(
            q=wp.empty((expanded_batch_size, self._num_dofs), dtype=wp.float32, device=self.device),
            T_world_base=se3_identity(shape=(expanded_batch_size,), device=self.device)
            if self.spec.floating_base
            else None,
        )
        self._initial_var = VarValues({VarKey.ROBOT: self._initial_state})
        ctx = HandContext
        context: Dict[Context, Any] = {
            ctx.POINTS: self._target_points_wp,
            ctx.VECTORS: self._vector_targets_wp,
            ctx.DIRECTIONS: self._direction_targets_wp,
            ctx.TARGET_NAMES: self.spec.target_names,
            ctx.TARGET_LINK_NAMES: self.spec.target_link_names,
            ctx.CONTACT_TARGETS: self._contact_targets,
            ctx.PREVIOUS_STATE: self._smoothness_reference,
            ctx.REST_Q: self._rest_q_np,
            ctx.FRAME: self._frame_wp,
        }
        terms: List[ResidualTask] = []
        for template in self.config.terms:
            if template.active:
                task = copy.copy(template)
                task.resolve(self._initial_var, context)
                if task.active:  # building can still empty it, e.g. base smoothness on a fixed base
                    terms.append(task)
        self._pair_wp = {key: wp.from_numpy(value, device=device) for key, value in self._pair_np.items()}
        self._pinch_latched_wp = wp.zeros((batch_size, pairs), dtype=wp.bool, device=device)

        tangent_dim = (6 if self.spec.floating_base else 0) + self._num_dofs
        self._sampler = RobertsSampler(
            max_candidates, [0.0] * tangent_dim, self.config.sampler.seed, self.config.sampler.keep_first, self.device
        )
        self._candidate_tangent = wp.zeros((expanded_batch_size, tangent_dim), dtype=wp.float32, device=self.device)
        self._repeat_idx = wp.from_numpy(
            np.repeat(np.arange(batch_size), max_candidates).astype(np.int32), device=self.device
        )
        self._zero_idx = wp.zeros(expanded_batch_size, dtype=wp.int32, device=self.device)
        limits = self.robot.spec.actuated_joint_limits.astype(np.float32)
        self._joint_range_np = limits[:, 1] - limits[:, 0]
        self._candidate_scale_np = np.zeros(0, dtype=np.float32)
        self._mid_state = self.robot.state(
            q=wp.from_numpy(self.robot.spec.midrange_q.astype(np.float32)[None], dtype=wp.float32, device=self.device)
        )
        self._init_candidates(self._previous_state if self.spec.init_q_by_name else None)
        self._solver = PopulationSolver(
            terms=terms,
            config=replace(self.config.solver, use_early_stopping=False),
            device=self.device,
            score_terms=None,
        )
        self._solver.setup(self._initial_var)

        self._joint_limits_wp = self.robot.spec.get_tensors(str(self.device)).actuated_joint_limits
        self._output_q_history_wp = wp.zeros((batch_size, self._num_dofs), dtype=wp.float32, device=self.device)
        self._output_wp = wp.zeros((batch_size, 7 + self._num_dofs), dtype=wp.float32, device=self.device)
        self._batch_size = batch_size
        self.reset()
        self.solve(wp.zeros((batch_size, len(self.spec.target_names), 3), dtype=wp.float32, device=self.device))
        wp.synchronize_device(self.device)
        self.reset()

    def _init_candidates(self, center: Optional[RobotState]):
        """Roberts noise around `center`, or around the joint midrange spanning the full limits."""
        if center is not None and self.config.solver.num_candidates[0] == 1:  # one candidate sits on its centre
            wp.copy(self._initial_state.q, center.q)
            if self.spec.floating_base:
                wp.copy(self._initial_state.T_world_base, center.T_world_base)
        else:
            src, idx = (center, self._repeat_idx) if center is not None else (self._mid_state, self._zero_idx)
            gather(src.q, idx, self._initial_state.q)
            scale = (self.config.sampler.scale if center is not None else 1.0) * self._joint_range_np
            if self.spec.floating_base:
                gather(src.T_world_base, idx, self._initial_state.T_world_base)
                base_scale = np.zeros(6, dtype=np.float32)
                if center is not None and center.has_floating_base:
                    base_scale[:3] = self.config.base_init_sample_range
                scale = np.concatenate([base_scale, scale])
            scale = scale.astype(np.float32)
            if not np.array_equal(scale, self._candidate_scale_np):  # the spread rarely changes; the noise only then
                self._sampler.scale.assign(scale)
                self._candidate_scale_np = scale
                self._sampler.fill(self._candidate_tangent)
            self._initial_state.integrate(self._candidate_tangent, out=self._initial_state)
        self._initial_state.clamp_q()

    def _center_state(self, state: RobotState) -> RobotState:
        wp.copy(self._local_state.q, state.q)
        wp.launch(
            center_base_kernel,
            dim=self._batch_size,
            inputs=[state.T_world_base, self._input_points_wp, self._root_index, self._local_state.T_world_base],
            device=self.device,
        )
        self._local_state.invalidate()
        return self._local_state

    def solve(
        self,
        target_points: wp.array,
        init_state: Optional[RobotState] = None,
        out: Optional[wp.array] = None,
    ) -> wp.array:
        """Retarget one batched frame.

        Args:
            target_points: Point targets with shape `(batch, targets, 3)` in `spec.target_names` order.
            init_state: Optional state overriding the streaming warm start.
            out: Optional packed output buffer with shape `(batch, 7 + dofs)`.

        Returns:
            Packed base transform and joint positions; a helper-owned buffer is reused when `out` is omitted.
        """
        if self._batch_size is None:
            raise RuntimeError("call warmup(batch_size) before solve().")
        expected_points = (self._batch_size, len(self.spec.target_names), 3)
        if target_points.shape != expected_points or target_points.dtype != wp.float32:
            raise ValueError(f"target_points must be float32 with shape {expected_points}.")
        if target_points.device != self.device:
            raise ValueError(f"target_points must be on {self.device}, got {target_points.device}.")
        output = self._output_wp if out is None else out
        expected_output = (self._batch_size, 7 + self._num_dofs)
        if out is not None and (out.shape, out.dtype, out.device) != (expected_output, wp.float32, self.device):
            raise ValueError(f"out must be float32 on {self.device} with shape {expected_output}.")
        expected_state = (self._batch_size, self._num_dofs), (self._batch_size,), self.device
        if init_state is not None:
            if (init_state.q.shape, init_state.T_world_base.shape, init_state.device) != expected_state:
                raise ValueError("init_state must match the warmed batch, robot DOFs, and device.")

        if self._wrist_origin:
            wp.copy(self._input_points_wp, target_points)
            wp.launch(
                center_points_kernel,
                dim=expected_points,
                inputs=[self._input_points_wp, self._root_index, self._target_points_wp],
                device=self.device,
            )
            if init_state is not None:
                init_state = self._center_state(init_state)
        else:
            wp.copy(self._target_points_wp, target_points)
        pair = self._pair_wp
        if self._uses_vectors:
            wp.launch(
                compute_vector_targets_kernel,
                dim=(self._batch_size, self._num_pairs),
                inputs=[
                    self._target_points_wp,
                    pair["origins"],
                    pair["tasks"],
                    pair["pinch"],
                    self.config.target_scale,
                    self.config.pinch_threshold,
                    self.config.pinch_release_threshold,
                    self.config.target_ema_alpha,
                    1 if self._history_valid else 0,
                    self._pinch_latched_wp,
                    self._vector_targets_wp,
                ],
                device=self.device,
            )
        if self._uses_directions:
            wp.launch(
                compute_direction_targets_kernel,
                dim=(self._batch_size, self._num_pairs),
                inputs=[
                    self._target_points_wp,
                    pair["origins"],
                    pair["tasks"],
                    self.config.target_ema_alpha,
                    1 if self._history_valid else 0,
                    self._direction_targets_wp,
                ],
                device=self.device,
            )

        if init_state is None and (self._history_valid or self.spec.init_q_by_name):
            init_state = self._previous_state
        self._init_candidates(init_state)
        self._frame_wp.fill_(int(self._history_valid))
        if self._history_valid:
            self._copy_previous_state(self._smoothness_reference)

        self._initial_state.invalidate()
        best_var, _ = self._solver.solve(self._initial_var)
        best_state = cast(RobotState, best_var.get(VarKey.ROBOT))
        wp.launch(
            set_online_solution_kernel,
            dim=(self._batch_size, 7 + self._num_dofs),
            inputs=[
                best_state.q,
                best_state.T_world_base,
                self._joint_limits_wp,
                self.config.output_ema_alpha,
                1 if self._history_valid else 0,
                self._previous_state.q,
                self._previous_state.T_world_base,
                self._output_q_history_wp,
                output,
            ],
            device=self.device,
        )
        if self._wrist_origin:
            wp.launch(
                restore_translation_kernel,
                dim=(self._batch_size, 3),
                inputs=[self._input_points_wp, self._root_index, output],
                device=self.device,
            )
        self._history_valid = True
        return output

    def solve_numpy(
        self,
        target_points: np.ndarray,
        init_state: Optional[RobotState] = None,
    ) -> np.ndarray:
        """NumPy wrapper for `solve()`."""
        points = np.asarray(target_points, dtype=np.float32)
        if points.ndim == 2:
            points = points[None]
        points_wp = wp.from_numpy(np.ascontiguousarray(points), dtype=wp.float32, device=self.device)
        return self.solve(points_wp, init_state=init_state).numpy().copy()

    def reset(self) -> None:
        """Clear all streaming history and disable smoothness until a previous frame exists."""
        self._history_valid = False
        if self._batch_size is None:
            return
        wp.copy(self._previous_state.q, self._reset_q_wp)
        wp.copy(self._previous_state.T_world_base, self._reset_base_wp)
        self._target_points_wp.zero_()
        if self._wrist_origin:
            self._input_points_wp.zero_()
        self._vector_targets_wp.zero_()
        self._direction_targets_wp.zero_()
        self._pinch_latched_wp.zero_()
        self._output_q_history_wp.zero_()
        self._output_wp.zero_()
        self._frame_wp.zero_()
        self._copy_previous_state(self._smoothness_reference)

    def _copy_previous_state(self, out: RobotState):
        wp.copy(out.q, self._previous_state.q)
        if self.spec.floating_base:
            wp.copy(out.T_world_base, self._previous_state.T_world_base)

    def compute_costs(self, state: RobotState) -> Dict[str, wp.array]:
        """Evaluate the current targets for a caller-provided robot state.

        Args:
            state: Robot state matching the warmed batch and device.

        Returns:
            Per-term costs keyed by task cost name.
        """
        if self._batch_size is None:
            raise RuntimeError("call warmup(batch_size) before compute_costs().")
        expected_state = (self._batch_size, self._num_dofs), (self._batch_size,), self.device
        if (state.q.shape, state.T_world_base.shape, state.device) != expected_state:
            raise ValueError("state must match the warmed batch, robot DOFs, and device.")
        if self._wrist_origin:
            state = self._center_state(state)
        return self._solver.compute_costs(VarValues({VarKey.ROBOT: state}))


__all__ = ["HandRetargeterOnline"]
