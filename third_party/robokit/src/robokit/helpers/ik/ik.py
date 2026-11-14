# pyright: reportInvalidTypeForm=false, reportIndexIssue=false

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union, cast

import numpy as np
import warp as wp

from robokit.geom import WarpScene
from robokit.helpers.ik.config import IKContext, IKSolverConfig
from robokit.lie.se3 import se3_from_matrix, se3_identity, se3_to_matrix
from robokit.lie.se3_kernels import se3_compose_func, se3_inverse_func
from robokit.opt.optimizer import aggregate_residuals_to_costs_kernel
from robokit.opt.population_solver import PopulationSolver
from robokit.opt.sampler import RobertsSampler
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo.robot import Robot, RobotState
from robokit.terms.composite_task import CompositeTask
from robokit.terms.task import Context, ResidualTask
from robokit.utils.warp_utils import gather, resolve_device, wp_device_type, wp_vec7


if TYPE_CHECKING:
    import torch


@wp.kernel
def _pin_inactive_q_kernel(
    inactive_q: wp.array2d(dtype=wp.float32),
    joint_mask: wp.array1d(dtype=wp.float32),
    num_candidates: wp.int32,
    out_q: wp.array2d(dtype=wp.float32),
):
    """Locked joints take the rest value instead of the candidate value; the batch is instance-major."""
    tid = wp.tid()
    batch_idx, joint = tid // out_q.shape[1], tid % out_q.shape[1]
    if joint_mask[joint] == 0.0:
        out_q[batch_idx, joint] = inactive_q[batch_idx // num_candidates, joint]


@wp.kernel
def select_better_score_kernel(
    prev_q: wp.array2d(dtype=wp.float32),
    prev_base: wp.array1d(dtype=wp_vec7),
    result_q: wp.array2d(dtype=wp.float32),
    result_base: wp.array1d(dtype=wp_vec7),
    prev_score: wp.array1d(dtype=wp.float32),
    result_score: wp.array1d(dtype=wp.float32),
    num_joints: int,
):
    bid = wp.tid()
    if prev_score[bid] <= result_score[bid]:
        for j in range(num_joints):
            result_q[bid, j] = prev_q[bid, j]
        result_base[bid] = prev_base[bid]


@wp.kernel
def _ik_metrics_kernel(
    T_world_link: wp.array2d(dtype=wp_vec7),
    T_world_target: wp.array2d(dtype=wp_vec7),
    target_link_indices: wp.array1d(dtype=wp.int32),
    position_threshold_m: float,
    orientation_threshold_rad: float,
    position_error_m: wp.array2d(dtype=wp.float32),
    orientation_error_rad: wp.array2d(dtype=wp.float32),
    success: wp.array1d(dtype=wp.bool),
):
    """Per-target FK-vs-target error in meters/radians; success is every target under both thresholds."""
    batch = wp.tid()
    reached = int(1)
    for i in range(target_link_indices.shape[0]):
        link = T_world_link[batch, target_link_indices[i]]
        error = se3_compose_func(se3_inverse_func(link), T_world_target[batch, i])
        position = wp.sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2])  # pyright: ignore[reportOperatorIssue]
        angle = 2.0 * wp.asin(wp.min(wp.sqrt(error[4] * error[4] + error[5] * error[5] + error[6] * error[6]), 1.0))  # pyright: ignore[reportOperatorIssue]
        position_error_m[batch, i] = position
        orientation_error_rad[batch, i] = angle
        if position >= position_threshold_m or angle >= orientation_threshold_rad:
            reached = int(0)
    success[batch] = reached == 1


class IKMetrics(NamedTuple):
    """Weight-independent task-space accuracy: errors per target, success per batch element."""

    position_error_m: Union[wp.array, np.ndarray]  # `[batch, num_targets]` meters
    orientation_error_rad: Union[wp.array, np.ndarray]  # `[batch, num_targets]` radians
    success: Union[wp.array, np.ndarray]  # `[batch]` bool, every target under both thresholds


class IKResultTorch(NamedTuple):
    q: "torch.Tensor"
    T_world_base: Optional["torch.Tensor"] = None


class IKSolver:
    """IK solver accepting one target link or a sequence, using names or indices."""

    def __init__(
        self,
        config: IKSolverConfig,
        robot: Robot,
        target_links: Union[str, int, Sequence[Union[str, int]]],
        device: Optional[wp_device_type] = None,
        scene: Optional[WarpScene] = None,
    ):
        self.config = config
        self.robot: Robot = robot
        self.scene = scene
        self._wp_device = resolve_device(device)
        self.device = str(self._wp_device)

        if isinstance(target_links, (str, int)):
            self._link_names: List[Union[str, int]] = [target_links]
        else:
            self._link_names = list(target_links)

        self.target_link_indices: List[int] = []
        for frame in self._link_names:
            if isinstance(frame, str):
                if frame not in self.robot.link_names:
                    raise ValueError(f"link '{frame}' not in robot.link_names ({len(self.robot.link_names)} links)")
                self.target_link_indices.append(self.robot.link_names.index(frame))
            else:
                if frame < 0 or frame >= len(self.robot.link_names):
                    raise ValueError(f"link index {frame} out of range [0, {len(self.robot.link_names)})")
                self.target_link_indices.append(int(frame))
        self.num_joints = self.robot.spec.num_actuated_joints

        n_links = len(self._link_names)
        for term in [*config.terms, *config.score_terms]:
            w = term.weight  # a term on the target links may weigh each link
            if isinstance(w, (list, tuple)) and IKContext.TARGET_LINKS in term.needs() and len(w) != n_links:
                raise ValueError(
                    f"weight sequence length {len(w)} != {n_links} (term '{term.name}' targets {n_links} link(s))"
                )

        self._base_offset = 6 if config.enable_T_world_base else 0
        tangent_dim = self._base_offset + self.num_joints
        self._sampler = RobertsSampler(
            config.solver.num_candidates[0],
            [0.0] * tangent_dim,
            config.sampler.seed,
            config.sampler.keep_first,
            self._wp_device,
        )
        limits = self.robot.spec.actuated_joint_limits.astype(np.float32)
        self._joint_range_np = limits[:, 1] - limits[:, 0]
        self._active_dof_mask = wp.ones(tangent_dim, dtype=wp.float32, device=self._wp_device)
        # These views share _active_dof_mask; Warp <1.9 rejects the empty base slice on fixed-base robots.
        self._active_base_mask = (
            cast(wp.array, self._active_dof_mask[: self._base_offset])
            if self._base_offset
            else wp.zeros(0, dtype=wp.float32, device=self._wp_device)
        )
        self._active_joint_mask = cast(wp.array, self._active_dof_mask[self._base_offset :])
        self._joint_mask_staging_np = np.ones(self.num_joints, dtype=np.float32)
        self._base_mask_staging_np = np.asarray(config.base_active_mask, dtype=np.float32).copy()
        if config.enable_T_world_base:
            wp.copy(self._active_base_mask, wp.from_numpy(self._base_mask_staging_np, dtype=wp.float32, device="cpu"))

        self._term_by_name: Dict[str, ResidualTask] = {t.name: t for t in [*config.terms, *config.score_terms]}
        self._built: Dict[ResidualTask, List[ResidualTask]] = {t: [] for t in [*config.terms, *config.score_terms]}

        self._initialized = False
        self._target_link_indices_wp: Optional[wp.array] = None

    # --- public mutation API ------------------------------------------------------
    @property
    def num_targets(self) -> int:
        return len(self.target_link_indices)

    @property
    def num_solutions(self) -> int:
        """Number of solutions returned per target instance."""
        return self.config.solver.num_candidates[-1]

    def set_weight(self, name: str, value: Union[float, Sequence[float]]):
        """Update a term's weight in-place: scalar for uniform, per-link sequence for terms on the target links."""
        if name not in self._term_by_name:
            raise KeyError(f"No term named '{name}'. Available: {list(self._term_by_name)}")
        term = self._term_by_name[name]

        if isinstance(value, (list, tuple, np.ndarray)):
            if IKContext.TARGET_LINKS not in term.needs():
                raise TypeError(f"per-link weights need a term on IKContext.TARGET_LINKS; '{name}' is not one")
            n_links = len(self._link_names)
            if len(value) != n_links:
                raise ValueError(
                    f"weight sequence length {len(value)} != {n_links} (term '{name}' targets {n_links} link(s))"
                )

        term.weight = value

        for w in self._built[term]:
            w.set_weight(value)

    def set_active_joint_mask(self, joint_mask: Sequence[float]):
        """Set which actuated joints are active (1.0) or locked (0.0)."""
        np.copyto(self._joint_mask_staging_np, joint_mask)
        mask_wp = wp.from_numpy(self._joint_mask_staging_np, dtype=wp.float32, device="cpu")
        wp.copy(self._active_joint_mask, mask_wp)
        self._global_candidates = None  # locked joints get no candidate noise, so redraw

    def set_active_base_mask(self, base_mask: Sequence[float]):
        """Set active (1.0)/locked (0.0) base DOFs `[x,y,z,roll,pitch,yaw]`; needs `enable_T_world_base=True`."""
        if self._base_offset != 6:
            raise ValueError("set_active_base_mask requires enable_T_world_base=True")
        if len(base_mask) != 6:
            raise ValueError(f"base_mask length {len(base_mask)} != 6")
        np.copyto(self._base_mask_staging_np, base_mask)
        mask_wp = wp.from_numpy(self._base_mask_staging_np, dtype=wp.float32, device="cpu")
        wp.copy(self._active_base_mask, mask_wp)
        self._global_candidates = None

    def warmup(self, batch_size: int):
        """Eagerly build the multi-candidate solver for a given batch size."""
        self._build_solver(batch_size)

    # --- solve --------------------------------------------------------------------
    def solve(
        self,
        T_world_target: wp.array,
        prev_state: Optional[RobotState] = None,
        init_state: Optional[RobotState] = None,
        rest_state: Optional[RobotState] = None,
        out_state: Optional[RobotState] = None,
        init_sample_range: Optional[float] = None,
        base_init_sample_range: Optional[Sequence[float]] = None,
        scene_indices: Optional[wp.array] = None,
    ) -> RobotState:
        """Solve IK and return instance-major final-stage solutions shaped `(batch * num_solutions, dofs)`; `solver.rank_final=True` sorts them best first within each instance."""
        batch_size = self._validate_target(T_world_target)

        if not self._initialized:
            self._build_solver(batch_size)
        elif batch_size != self._batch_size:
            raise ValueError(f"Batch size mismatch: expected {self._batch_size}, got {batch_size}")
        if prev_state is not None and self.num_solutions != 1:
            raise ValueError("prev_state requires an IK final stage with num_candidates=1.")

        q_sample_range = init_sample_range if init_sample_range is not None else self.config.sampler.scale
        base_sample_range = (
            base_init_sample_range if base_init_sample_range is not None else self.config.base_init_sample_range
        )

        self._init_candidates(init_state, q_sample_range, base_sample_range, rest_state)

        if prev_state is not None and prev_state.batch_size != self._batch_size:
            raise ValueError(
                f"prev_state batch size mismatch: expected {self._batch_size}, got {prev_state.batch_size}"
            )
        self._update_targets(T_world_target, prev_state, rest_state)
        self._update_scene_indices(scene_indices, batch_size)

        # Score prev with the solver's ranking rule; keep it on a tie to avoid solution jitter.
        if prev_state is not None:
            self._prev_ruler.compute_weighted_residual(
                VarValues({VarKey.ROBOT: prev_state}), out_residual=self._prev_rbuf
            )
            wp.launch(
                aggregate_residuals_to_costs_kernel,
                dim=self._batch_size,
                inputs=[self._prev_rbuf, self._prev_score],
                device=self._solver.device,
            )
            wp.copy(self._prev_q_save, prev_state.q)  # type: ignore[reportArgumentType]
            wp.copy(self._prev_base_save, prev_state.T_world_base)  # type: ignore[reportArgumentType]

        best_var, best_costs = self._solver.solve(self._initial_var)
        best_state = cast(RobotState, best_var.get(VarKey.ROBOT))

        if prev_state is not None:
            wp.launch(
                select_better_score_kernel,
                dim=self._batch_size,
                inputs=[
                    self._prev_q_save,
                    self._prev_base_save,
                    best_state.q,
                    best_state.T_world_base,
                    self._prev_score,
                    best_costs,
                    self.robot.num_actuated_joints,
                ],
                device=self._solver.device,
            )

        final_var = self._solver.final_var
        if self.config.solver.rank_final and self.num_solutions > 1:
            final_var = final_var.gather(
                self._solver.ranked_indices.reshape((final_var.batch_size,)), out=self._ranked_var
            )
        state = cast(RobotState, final_var.get(VarKey.ROBOT))

        if out_state is not None:
            wp.copy(out_state.q, state.q)
            if self.config.enable_T_world_base and out_state.T_world_base is not None:
                wp.copy(out_state.T_world_base, state.T_world_base)
            return out_state

        return state

    def solve_torch(
        self,
        T_world_target: "torch.Tensor",
        prev_state: Optional[RobotState] = None,
        init_q: Optional["torch.Tensor"] = None,
        init_T_world_base: Optional["torch.Tensor"] = None,
        rest_q: Optional["torch.Tensor"] = None,
        rest_T_world_base: Optional["torch.Tensor"] = None,
        init_sample_range: Optional[float] = None,
        base_init_sample_range: Optional[Sequence[float]] = None,
    ) -> IKResultTorch:
        if not 2 <= T_world_target.ndim <= 4 or T_world_target.shape[-2:] != (4, 4):
            raise ValueError(f"T_world_target must have shape (..., 4, 4), got {tuple(T_world_target.shape)}")
        T_world_target = T_world_target.reshape(-1, self.num_targets, 4, 4)
        T_world_target_wp = se3_from_matrix(wp.from_torch(T_world_target.contiguous(), dtype=wp.mat44))

        def to_state(q: Optional["torch.Tensor"], T_world_base: Optional["torch.Tensor"]) -> Optional[RobotState]:
            if q is None:
                return None
            base = (
                se3_from_matrix(wp.from_torch(T_world_base.contiguous(), dtype=wp.mat44))
                if T_world_base is not None and self.config.enable_T_world_base
                else None
            )
            return self.robot.state(q=wp.from_torch(q.contiguous(), dtype=wp.float32), T_world_base=base)

        init_state = to_state(init_q, init_T_world_base)
        rest_state = to_state(rest_q, rest_T_world_base)

        state = self.solve(
            T_world_target_wp,
            prev_state=prev_state,
            init_state=init_state,
            rest_state=rest_state,
            init_sample_range=init_sample_range,
            base_init_sample_range=base_init_sample_range,
        )
        q_torch = wp.to_torch(state.q).clone()
        T_world_base_torch = None
        if self.config.enable_T_world_base:
            T_world_base_torch = wp.to_torch(se3_to_matrix(state.T_world_base)).clone()
        return IKResultTorch(q=q_torch, T_world_base=T_world_base_torch)

    def solve_numpy(
        self,
        T_world_target: np.ndarray,
        prev_state: Optional[RobotState] = None,
        init_state: Optional[RobotState] = None,
        rest_state: Optional[RobotState] = None,
        out_state: Optional[RobotState] = None,
        init_sample_range: Optional[float] = None,
        base_init_sample_range: Optional[Sequence[float]] = None,
    ) -> RobotState:
        T_world_target = np.asarray(T_world_target, dtype=np.float32)
        if not 1 <= T_world_target.ndim <= 3 or T_world_target.shape[-1] != 7:
            raise ValueError(f"T_world_target must have shape (..., 7), got {T_world_target.shape}")
        T_world_target = T_world_target.reshape(-1, self.num_targets, 7)
        return self.solve(
            wp.from_numpy(T_world_target, dtype=wp_vec7, device=self.device),
            prev_state=prev_state,
            init_state=init_state,
            rest_state=rest_state,
            out_state=out_state,
            init_sample_range=init_sample_range,
            base_init_sample_range=base_init_sample_range,
        )

    def compute_costs(
        self,
        state: RobotState,
        T_world_target: wp.array,
        prev_state: Optional[RobotState] = None,
        scene_indices: Optional[wp.array] = None,
    ) -> Dict[str, wp.array]:
        batch_size = self._validate_target(T_world_target)
        if batch_size != state.batch_size:
            raise ValueError(f"Target batch size ({batch_size}) must match state batch size ({state.batch_size})")
        self._update_targets(T_world_target, prev_state, None)
        self._update_scene_indices(scene_indices, batch_size)
        return self._solver.compute_costs(VarValues({VarKey.ROBOT: state}))

    def compute_metrics(
        self,
        state: RobotState,
        T_world_target: wp.array,
        position_threshold_m: float = 0.005,
        orientation_threshold_rad: float = 0.05,
        out: Optional[IKMetrics] = None,
    ) -> IKMetrics:
        """Task-space accuracy of `state` against `T_world_target`, independent of term weights.

        Args:
            state: Robot state to score; any solver's output works, not just this one's.
            T_world_target: `[batch, num_targets]` wp_vec7 targets on this solver's device.
            position_threshold_m: Success threshold on translation error.
            orientation_threshold_rad: Success threshold on rotation error.
            out: Buffers to write in place, from an earlier call; allocated when omitted.

        Returns:
            Errors per target and per-batch success.
        """
        batch_size = self._validate_target(T_world_target)
        if batch_size != state.batch_size:
            raise ValueError(f"Target batch size ({batch_size}) must match state batch size ({state.batch_size})")
        if self._target_link_indices_wp is None:
            self._target_link_indices_wp = wp.array(
                np.asarray(self.target_link_indices, dtype=np.int32), dtype=wp.int32, device=self._wp_device
            )
        if out is None:
            shape = (batch_size, self.num_targets)
            out = IKMetrics(
                position_error_m=wp.empty(shape, dtype=wp.float32, device=self._wp_device),  # type: ignore[reportArgumentType]
                orientation_error_rad=wp.empty(shape, dtype=wp.float32, device=self._wp_device),  # type: ignore[reportArgumentType]
                success=wp.empty(batch_size, dtype=wp.bool, device=self._wp_device),  # type: ignore[reportArgumentType]
            )
        state = self.robot.forward_kinematics(state)
        wp.launch(
            _ik_metrics_kernel,
            dim=batch_size,
            inputs=[
                state.T_world_link.reshape((state.num_elements, self.robot.spec.num_links)),
                T_world_target,
                self._target_link_indices_wp,
                position_threshold_m,
                orientation_threshold_rad,
                out.position_error_m,
                out.orientation_error_rad,
                out.success,
            ],
            device=self._wp_device,
        )
        return out

    def compute_metrics_numpy(
        self,
        state: RobotState,
        T_world_target: np.ndarray,
        position_threshold_m: float = 0.005,
        orientation_threshold_rad: float = 0.05,
    ) -> IKMetrics:
        """`compute_metrics` for numpy targets shaped `(7,)`, `(batch, 7)` or `(batch, num_targets, 7)`.

        Args:
            state: Robot state to score; any solver's output works, not just this one's.
            T_world_target: Targets as `[x, y, z, q_w, q_x, q_y, q_z]`.
            position_threshold_m: Success threshold on translation error.
            orientation_threshold_rad: Success threshold on rotation error.

        Returns:
            Errors per target and per-batch success, as numpy arrays.
        """
        T_world_target = np.asarray(T_world_target, dtype=np.float32)
        if not 1 <= T_world_target.ndim <= 3 or T_world_target.shape[-1] != 7:
            raise ValueError(f"T_world_target must have shape (..., 7), got {T_world_target.shape}")
        T_world_target = T_world_target.reshape(-1, self.num_targets, 7)
        metrics = self.compute_metrics(
            state,
            wp.from_numpy(T_world_target, dtype=wp_vec7, device=self.device),
            position_threshold_m,
            orientation_threshold_rad,
        )
        return IKMetrics(
            position_error_m=metrics.position_error_m.numpy(),
            orientation_error_rad=metrics.orientation_error_rad.numpy(),
            success=metrics.success.numpy(),
        )

    def _validate_target(self, T_world_target: wp.array) -> int:
        if T_world_target.dtype != wp_vec7:
            raise TypeError(f"T_world_target must have dtype wp_vec7, got {T_world_target.dtype}")
        if T_world_target.device != self._wp_device:
            raise ValueError(f"T_world_target must be on {self._wp_device}, got {T_world_target.device}")
        if T_world_target.ndim != 2:
            raise ValueError(f"T_world_target must have logical shape [batch, targets], got {T_world_target.shape}")
        if T_world_target.shape[1] != self.num_targets:
            raise ValueError(
                f"Target count ({T_world_target.shape[1]}) must match IK target count ({self.num_targets})"
            )
        return T_world_target.shape[0]

    # --- internal: solver build (lazy) -------------------------------------------
    def _init_candidates(
        self,
        init_state: Optional[RobotState],
        q_range: Union[float, Sequence[float]],
        base_range: Sequence[float],
        rest_state: Optional[RobotState],
    ):
        """Seed joints with Roberts noise.

        Seed centers:
        - `init_state`: perturb the supplied joints.
        - Joint midrange: span the full limits using candidates drawn once and copied on subsequent solves.
        """
        initial_state = cast(RobotState, self._initial_var.get(VarKey.ROBOT))
        pin = rest_state is not None and np.any(self._joint_mask_staging_np == 0.0)
        if init_state is None and self._global_candidates is not None and not pin:
            wp.copy(initial_state.q, self._global_candidates[0])
            if self.config.enable_T_world_base:
                wp.copy(initial_state.T_world_base, cast(wp.array, self._global_candidates[1]))
            return
        if init_state is not None and self.config.solver.num_candidates[0] == 1:  # one candidate sits on its centre
            wp.copy(initial_state.q, init_state.q)
            if self.config.enable_T_world_base:
                wp.copy(initial_state.T_world_base, init_state.T_world_base)
        else:
            src, idx = (init_state, self._repeat_idx) if init_state is not None else (self._mid_state, self._zero_idx)
            gather(src.q, idx, initial_state.q)
            scale = (q_range if init_state is not None else 1.0) * self._joint_range_np * self._joint_mask_staging_np
            if self.config.enable_T_world_base:
                gather(src.T_world_base, idx, initial_state.T_world_base)
                base_scale = np.zeros(6, dtype=np.float32)
                if init_state is not None:
                    base_scale[:3] = np.asarray(base_range, dtype=np.float32) * self._base_mask_staging_np[:3]
                scale = np.concatenate([base_scale, scale])
            scale = scale.astype(np.float32)
            if not np.array_equal(scale, self._candidate_scale_np):  # the spread rarely changes; the noise only then
                self._sampler.scale.assign(scale)
                self._candidate_scale_np = scale
                self._sampler.fill(self._candidate_tangent)
            initial_state.integrate(self._candidate_tangent, out=initial_state)
        initial_state.clamp_q()
        if init_state is None and not pin:
            self._global_candidates = (
                wp.clone(initial_state.q),
                wp.clone(initial_state.T_world_base) if self.config.enable_T_world_base else None,
            )
        if pin:
            wp.launch(
                _pin_inactive_q_kernel,
                dim=initial_state.q.size,
                inputs=[rest_state.q, self._active_joint_mask, self.config.solver.num_candidates[0], initial_state.q],
                device=self._wp_device,
            )

    def _build_solver(self, batch_size: int):
        if self._initialized:
            return

        self._batch_size = batch_size
        num_candidates = self.config.solver.num_candidates[0]
        self._candidate_tangent = wp.zeros(
            (batch_size * num_candidates, self._sampler.tangent_dim), dtype=wp.float32, device=self._wp_device
        )
        self._repeat_idx = wp.from_numpy(
            np.repeat(np.arange(batch_size), num_candidates).astype(np.int32), device=self._wp_device
        )
        self._zero_idx = wp.zeros(batch_size * num_candidates, dtype=wp.int32, device=self._wp_device)
        self._candidate_scale_np = np.zeros(0, dtype=np.float32)
        self._global_candidates: Optional[Tuple[wp.array, Optional[wp.array]]] = None
        self._mid_state = self.robot.state(
            q=wp.from_numpy(
                self.robot.spec.midrange_q.astype(np.float32)[None], dtype=wp.float32, device=self._wp_device
            )
        )

        placeholder_q = wp.empty((batch_size, self.num_joints), dtype=wp.float32, device=self._wp_device)  # type: ignore[reportArgumentType]
        if self.config.enable_T_world_base:
            T_world_base = se3_identity(shape=(batch_size,), device=self._wp_device)
            placeholder_state = self.robot.state(q=placeholder_q, T_world_base=T_world_base)
        else:
            placeholder_state = self.robot.state(q=placeholder_q)

        # the tasks read these buffers, which each solve refills; targets broadcast over the candidates
        device, floating = self._wp_device, self.config.enable_T_world_base
        self._frame_placeholders = se3_identity(shape=(batch_size, self.num_targets), device=device)
        prev_q = wp.zeros((batch_size, self.num_joints), dtype=wp.float32, device=device)
        prev_base = se3_identity(shape=(batch_size,), device=device) if floating else None
        self._prev_placeholder = self.robot.state(q=prev_q, T_world_base=prev_base)
        zero_q = np.tile(self.robot.spec.zero_q.astype(np.float32), (batch_size, 1))
        self._rest_q = wp.from_numpy(zero_q, dtype=wp.float32, device=device)
        self._rest_base = se3_identity(shape=(batch_size,), device=device) if floating else None
        self._scene_indices = wp.zeros(batch_size, dtype=wp.int32, device=device)
        ctx = IKContext
        self._context: Dict[Context, Any] = {
            ctx.TARGET_LINKS: self.target_link_indices,
            ctx.TARGET_LINK: self.target_link_indices[0],
            ctx.TARGET: self._frame_placeholders,
            ctx.PREVIOUS_STATE: self._prev_placeholder,
            ctx.REST_Q: self._rest_q,
            ctx.REST_BASE: self._rest_base,
            ctx.NUM_SOLUTIONS: self.num_solutions,
            ctx.SCENE: self.scene,
            ctx.SCENE_INDICES: self._scene_indices,
        }
        var = VarValues({VarKey.ROBOT: placeholder_state})
        terms = [task for t in self.config.terms for task in self._build(t, var)]
        self._optimization_terms = terms
        score_term: Optional[ResidualTask] = None
        self._context[ctx.OPTIMIZATION_TERMS] = terms
        if self.config.score_terms:
            scores = [task for t in self.config.score_terms for task in self._build(t, var)]
            score_term = scores[0] if len(scores) == 1 else CompositeTask(scores)
        # Score prev_state with score terms or total cost; ties retain it and suppress null-space jitter.
        self._prev_ruler: ResidualTask = score_term if score_term is not None else CompositeTask(list(terms))
        self._prev_rbuf = wp.empty(
            (batch_size, self._prev_ruler.residual_dim),
            dtype=wp.float32,  # type: ignore[reportArgumentType]
            device=self._wp_device,
        )

        supports_cuda_graph = all(t.SUPPORTS_CUDA_GRAPH for t in terms) and (
            score_term is None or score_term.SUPPORTS_CUDA_GRAPH
        )
        solver_config = replace(
            self.config.solver,
            cuda_graph_mode=self.config.solver.cuda_graph_mode if supports_cuda_graph else "none",
        )
        self._solver = PopulationSolver(
            terms=[terms],
            config=solver_config,
            device=self._wp_device,
            score_terms=score_term,
            active_dof_mask=self._active_dof_mask,
        )
        first_stage_batch = batch_size * self.config.solver.num_candidates[0]
        indices = wp.zeros((first_stage_batch,), dtype=wp.int32, device=self._wp_device)
        self._initial_var = VarValues({VarKey.ROBOT: placeholder_state.gather(indices)})
        self._solver.setup(self._initial_var)
        indices = wp.zeros((batch_size * self.num_solutions,), dtype=wp.int32, device=self._wp_device)
        self._ranked_var = VarValues({VarKey.ROBOT: placeholder_state.gather(indices)})

        self._prev_score = wp.empty((batch_size,), dtype=wp.float32, device=self._wp_device)  # type: ignore[reportArgumentType]
        self._prev_q_save = wp.empty(
            (batch_size, self.robot.num_actuated_joints),
            dtype=wp.float32,  # type: ignore[reportArgumentType]
            device=self._wp_device,
        )
        self._prev_base_save = wp.empty((batch_size,), dtype=wp_vec7, device=self._wp_device)  # type: ignore[reportArgumentType]

        self._initialized = True

    # --- internal: term building --------------------------------------------------
    def _build(self, term: ResidualTask, var: VarValues) -> List[ResidualTask]:
        """Resolve a template copy against `IKContext`, dropping it if emptied; tasks built with a robot pass as is."""
        built = term
        if term.robot is None:
            built = copy.copy(term)
            built.resolve(var, self._context)
            if term.active and not built.active:
                return []
        self._built[term].append(built)
        return [built]

    # --- internal: per-solve target updates --------------------------------------
    def _update_targets(
        self,
        T_world_target: wp.array,
        prev_state: Optional[RobotState],
        rest_state: Optional[RobotState],
    ):
        """Refill the buffers the built tasks read."""
        wp.copy(self._frame_placeholders, T_world_target)
        if prev_state is not None:
            wp.copy(self._prev_placeholder.q, prev_state.q)
            if self._prev_placeholder.has_floating_base and prev_state.has_floating_base:
                wp.copy(self._prev_placeholder.T_world_base, prev_state.T_world_base)
        if rest_state is not None:
            wp.copy(self._rest_q, rest_state.q)
            if self._rest_base is not None and rest_state.T_world_base is not None:
                wp.copy(self._rest_base, rest_state.T_world_base)

    def _update_scene_indices(self, scene_indices: Optional[wp.array], batch_size: int):
        """Refill the scene-index buffer the built tasks read."""
        if scene_indices is not None:
            if scene_indices.dtype != wp.int32 or scene_indices.ndim != 1:
                raise TypeError("scene_indices must be a 1D wp.int32 array.")
            if scene_indices.shape != (batch_size,):
                raise ValueError("scene_indices must have shape (batch_size,).")
            if scene_indices.device != self._wp_device:
                raise ValueError(f"scene_indices must be on {self._wp_device}, got {scene_indices.device}.")
            wp.copy(self._scene_indices, scene_indices)


__all__ = ["IKSolver", "IKResultTorch", "IKMetrics"]
