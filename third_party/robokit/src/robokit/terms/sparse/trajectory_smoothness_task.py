# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportUndefinedVariable=false
"""Joint trajectory costs, candidate paths and sparse finite differences."""

from math import comb
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np
import warp as wp

from robokit.lie.se3_kernels import (
    se3_adjoint_func,
    se3_compose_func,
    se3_inverse_func,
    se3_jlog_func,
    se3_log_map_func,
)
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo.robot_spec import JointType
from robokit.terms.task import Context, GradientTask, SparseTask, SparsityPattern, sparse_cost_and_gradient
from robokit.utils import warp_utils
from robokit.utils.warp_utils import wp_device_type, wp_vec7


if TYPE_CHECKING:
    import torch


def _temporal_differences(times: "torch.Tensor", velocity_weight: float, acceleration_weight: float) -> list:
    import torch

    if not velocity_weight and not acceleration_weight:
        return []
    dt = torch.diff(times).clamp(min=1e-3)
    velocity = torch.diff(torch.eye(len(times), device=times.device, dtype=times.dtype), dim=0) / dt[:, None]
    differences = []
    if velocity_weight and len(times) > 1:
        differences.append(velocity * (velocity_weight * dt).sqrt()[:, None])
    if acceleration_weight and len(times) > 2:
        # Integrate squared acceleration of the natural cubic through the samples exactly.
        mass = torch.diag((dt[1:] + dt[:-1]) / 3)
        if len(times) > 3:
            mass += torch.diag(dt[1:-1] / 6, 1) + torch.diag(dt[1:-1] / 6, -1)
        differences.append(
            torch.linalg.solve_triangular(torch.linalg.cholesky(mass), torch.diff(velocity, dim=0), upper=False)
            * acceleration_weight**0.5
        )
    return differences


_HISTORY_FRAMES = 3  # executed-state ring; index 2 aligns with trajectory frame 0


# --- task -------------------------------------------------------------------
class TrajectorySmoothnessTask(SparseTask, GradientTask):
    """Penalize joint differences and optional floating-base motion.

    Configuration options:
    - robot/num_frames omitted: `resolve` takes them from the trajectory it reads.
    - smooth_boundary=False: disable the first and last finite-difference rows.
    - history: an executed `(q_history, history_valid)` prefix, e.g. a helper's shared buffers.
    """

    compute_weighted_cost_and_gradient = sparse_cost_and_gradient

    def __init__(
        self,
        robot: Optional["Robot"] = None,  # noqa: F821
        num_frames: Optional[int] = None,
        weight: Union[float, Sequence[float], np.ndarray] = 1.0,
        order: int = 1,
        dt: Union[float, Context] = 1.0,
        reference_q: Optional[Union[np.ndarray, Context]] = None,
        base_weight: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        smooth_boundary: bool = True,
        virtual_hold_frames: Union[int, Context] = 0,
        history: Optional[Union[Tuple[wp.array, wp.array], Context]] = None,
        name: str = "",
    ):
        """`virtual_hold_frames` > 0 treats that many frames before the start and after the end as copies of them."""
        self.name = name
        if order < 1 or (num_frames is not None and order >= num_frames):
            raise ValueError(f"order must be positive and smaller than num_frames={num_frames}, got {order}")
        if base_weight is not None and order != 1:
            raise ValueError("base_weight is only supported for order=1.")
        self.robot = robot
        if num_frames is not None:
            self.num_frames = num_frames
        self.weight = weight
        self.order = int(order)
        self.dt = dt
        self.smooth_boundary = smooth_boundary
        self.reference_q_input = reference_q
        self.base_weight = base_weight
        self.virtual_hold_frames = virtual_hold_frames
        self.history = history
        self.device: Optional[wp_device_type] = None
        self.residual_weight: Optional[wp.array] = None
        self.base_residual_weight: Optional[wp.array] = None
        self.boundary_weight: Optional[wp.array] = None
        self.coefficients: Optional[wp.array] = None
        self.q_history: Optional[wp.array] = None
        self.history_valid: Optional[wp.array] = None
        self.reference_q: Optional[wp.array] = None

    @property
    def _periodic_joints(self) -> Optional[np.ndarray]:
        """Select full-turn revolute coordinates that do not drive mimic joints."""
        if self.robot is None:
            return None
        spec = self.robot.spec
        periodic = np.array(
            [spec.joint_types[spec.joint_names.index(n)] == JointType.REVOLUTE for n in spec.actuated_joint_names],
            dtype=bool,
        )
        periodic &= np.ptp(spec.actuated_joint_limits, axis=1) >= 2 * np.pi - 1e-6
        periodic[spec.mimic_actuated_joint_indices[spec.mimic_actuated_joint_indices >= 0]] = False
        return periodic

    def compute_paths(
        self,
        q: np.ndarray,
        costs: np.ndarray,
        times: np.ndarray,
        weight: float = 0.08,
        count: int = 1,
    ) -> np.ndarray:
        """Return [K,T,J] paths, preserving alternative initial pose hypotheses; each initial hypothesis retains a distinct branch, and the output contains at most `count` paths; periodic joints use angular distances and return continuous equivalent angles within supplied limits; if no continuous lift fits the limits, the original bounded path is retained.

        >>> q = np.array([[[0.], [3.]], [[3.], [0.1]], [[0.2], [3.]]])
        >>> TrajectorySmoothnessTask().compute_paths(q, np.array([[0., .1], [.1, 0.], [0., .1]]), np.arange(3.)).ravel().tolist()
        [0.0, 0.1, 0.2]
        """
        periodic_joints = self._periodic_joints
        joint_limits = None if self.robot is None else self.robot.spec.actuated_joint_limits
        if q.ndim != 3 or costs.shape != q.shape[:2] or len(times) != len(q):
            raise ValueError("Expected hypotheses [frames, candidates, joints] and matching costs/times.")
        if np.any(np.diff(times) <= 0):
            raise ValueError("Timestamps must increase.")
        if count < 1:
            raise ValueError("path count must be positive")
        # --- forward path costs ---
        value = np.full((q.shape[1], q.shape[1]), np.inf)
        np.fill_diagonal(value, costs[0])
        parents = []
        for i in range(1, len(q)):
            difference = q[i - 1, :, None] - q[i, None]
            if periodic_joints is not None:
                difference[..., periodic_joints] = (difference[..., periodic_joints] + np.pi) % (2 * np.pi) - np.pi
            transition = weight * np.mean(difference**2, axis=-1) / (times[i] - times[i - 1])
            total = value[:, :, None] + transition[None]
            parent = total.argmin(axis=1)
            parents.append(parent)
            value = costs[i][None] + np.take_along_axis(total, parent[:, None], axis=1)[:, 0]
        # --- rank and backtrack branches ---
        endings = value.argmin(axis=1)
        starts = np.argsort(value[np.arange(len(value)), endings], kind="stable")[:count]
        paths = []
        for start in starts:
            indices = [int(endings[start])]
            for parent in parents[::-1]:
                indices.append(int(parent[start, indices[-1]]))
            path = q[np.arange(len(q)), indices[::-1]].copy()
            if periodic_joints is not None and periodic_joints.any():
                angles = np.unwrap(path[:, periodic_joints], axis=0)
                center = 0 if joint_limits is None else joint_limits[periodic_joints].mean(-1)
                shift = np.rint((center - angles.mean(0)) / (2 * np.pi))
                if joint_limits is None:
                    path[:, periodic_joints] = angles + 2 * np.pi * shift
                else:
                    lower = np.ceil((joint_limits[periodic_joints, 0] - angles.min(0)) / (2 * np.pi))
                    upper = np.floor((joint_limits[periodic_joints, 1] - angles.max(0)) / (2 * np.pi))
                    path[:, periodic_joints] = np.where(
                        lower <= upper,
                        angles + 2 * np.pi * np.clip(shift, lower, upper),
                        path[:, periodic_joints],
                    )
            paths.append(path)
        return np.stack(paths)

    def compute_offset_candidates(self, q: np.ndarray, samples: int = 16) -> np.ndarray:
        """Sample constant joint shifts that preserve every frame's joint limits; full-turn joints need at most one revolution, and extra cubic samples retain small shifts."""
        if self.robot is None:
            raise ValueError("joint offset candidates require a robot")
        periodic = cast(np.ndarray, self._periodic_joints)
        candidates = [q]
        for joint, (lower, upper) in enumerate(self.robot.spec.actuated_joint_limits):
            low, high = (lower - q[:, joint]).max(), (upper - q[:, joint]).min()
            if not np.isfinite([low, high]).all():
                continue
            angular = periodic[joint]
            if angular:
                low, high = max(low, -np.pi), min(high, np.pi)
            fine = np.linspace(np.cbrt(low), np.cbrt(high), samples, endpoint=not angular) ** 3
            offsets = np.r_[
                np.linspace(low, high, samples, endpoint=not angular), fine[np.argsort(abs(fine))[: samples // 4]]
            ]
            for offset in np.unique(offsets):
                shifted = q.copy()
                shifted[:, joint] += offset
                candidates.append(shifted)
        return np.stack(candidates)

    def init_buffers(self, device: wp_device_type):
        """Build finite-difference weights and history buffers.

        Lifecycle:
            1. Build joint and optional base weights.
            2. Build the requested difference stencil.
            3. Allocate history and reference trajectories.
        """
        self.device = device
        num_dofs = self.robot.spec.num_actuated_joints
        num_interior = self.num_frames - self.order
        if self.virtual_hold_frames < 0:
            raise ValueError(f"virtual_hold_frames must be >= 0, got {self.virtual_hold_frames}")
        dt_scale = np.float32(1.0 / self.dt**self.order)

        weight = self.weight if np.isscalar(self.weight) else np.asarray(self.weight, dtype=np.float32)
        if not self.smooth_boundary:
            if isinstance(weight, np.ndarray) and weight.size == num_interior * num_dofs:
                weight = weight.reshape(num_interior, num_dofs)
            weight = np.broadcast_to(np.asarray(weight), (num_interior, num_dofs)).copy().reshape(-1)
            weight[:num_dofs] = weight[-num_dofs:] = 0.0

        if isinstance(weight, np.ndarray):
            flat = weight.astype(np.float32).flatten()
            if flat.size == num_interior * num_dofs:
                residual_weight = flat
            else:
                residual_weight = np.tile(weight.astype(np.float32), num_interior)
            dof_weight = weight.astype(np.float32)
        else:
            residual_weight = np.full(num_interior * num_dofs, self.weight, dtype=np.float32)
            dof_weight = np.full(num_dofs, self.weight, dtype=np.float32)

        self.residual_weight = wp.from_numpy(residual_weight * dt_scale, dtype=wp.float32, device=device)  # pyright: ignore[reportIncompatibleVariableOverride]
        self.boundary_weight = wp.from_numpy(dof_weight * dt_scale, dtype=wp.float32, device=device)
        coefficients = np.array(
            [((-1) ** (self.order - j)) * comb(self.order, j) for j in range(self.order + 1)], dtype=np.float32
        )
        self.coefficients = wp.from_numpy(coefficients, dtype=wp.float32, device=device)
        if self.history is not None:
            self.q_history, self.history_valid = self.history
        if self.q_history is None:
            self.q_history = wp.zeros((1, _HISTORY_FRAMES, num_dofs), dtype=wp.float32, device=device)
            self.history_valid = wp.zeros(1, dtype=wp.float32, device=device)

        if self.base_weight is not None:
            base_weight = np.broadcast_to(np.asarray(self.base_weight, dtype=np.float32), (6,))
            self.base_residual_weight = wp.from_numpy(
                np.tile(base_weight, self.num_frames - 1) / self.dt,
                dtype=wp.float32,
                device=device,
            )

        if isinstance(self.reference_q_input, wp.array):  # the helper's buffer, read directly
            self.reference_q = self.reference_q_input
        if self.reference_q is None:
            reference_q = (
                np.zeros((1, self.num_frames, num_dofs), dtype=np.float32)
                if self.reference_q_input is None
                else self.reference_q_input.astype(np.float32)
            )
            if reference_q.shape == (self.num_frames, num_dofs):
                reference_q = reference_q[None]
            if reference_q.shape[1:] != (self.num_frames, num_dofs):
                raise ValueError(f"reference_q must have shape [T, D] or [B, {self.num_frames}, {num_dofs}].")
            self.reference_q = wp.from_numpy(reference_q, dtype=wp.float32, device=device)

    @property
    def active(self) -> bool:
        return bool(np.any(self.weight) or (self.base_weight is not None and np.any(self.base_weight)))

    def resolve(self, var_values: VarValues, context: Mapping[Context, Any]):
        """Build from the robot and horizon of the trajectory `var_values` holds, and the helper's values."""
        super().resolve(var_values, context)
        state = cast("RobotState", var_values.get(self.var_key))
        self.robot, self.num_frames = state.robot, state.q.shape[1]
        if not state.has_floating_base:  # base rows follow a floating base only
            self.base_weight = None
        if self.order >= self.num_frames:
            raise ValueError(f"order must be smaller than num_frames={self.num_frames}, got {self.order}")
        self.init_buffers(state.q.device)

    def set_history(self, q_history: Optional[wp.array], num_candidates: int, history_valid: bool):
        """Set the fixed executed prefix. No-op at `order=1`, which never reads it."""
        if self.device is None:
            if q_history is None:
                return
            self.init_buffers(q_history.device)
        if q_history is not None:
            batch_size = q_history.shape[0] * num_candidates
            if self.q_history.shape[0] != batch_size:
                self.q_history = wp.zeros(
                    (batch_size,) + tuple(self.q_history.shape[1:]), dtype=wp.float32, device=self.device
                )
                self.history_valid = wp.zeros(batch_size, dtype=wp.float32, device=self.device)
            warp_utils.repeat(q_history, num_candidates, out=self.q_history)
        self.history_valid.fill_(1.0 if history_valid else 0.0)

    @property
    def _joint_rows(self) -> int:
        """Difference rows per joint: held end frames add the windows that reach past the last frame."""
        return self.num_frames - 1 + min(self.virtual_hold_frames, self.order - 1)

    @property
    def residual_dim(self) -> int:
        base_rows = (self.num_frames - 1) * 6 if self.base_weight is not None else 0
        return self._joint_rows * self.robot.spec.num_actuated_joints + base_rows

    def _nnz(self, num_dofs: int) -> int:
        joint_nnz = (self.order + 1) * self._joint_rows * num_dofs
        base_nnz = (self.num_frames - 1) * 6 * 12 if self.base_weight is not None else 0
        return joint_nnz + base_nnz

    def compute_weighted_residual(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        robot_state = var_values.get(VarKey.ROBOT)
        if self.device is None:
            self.init_buffers(robot_state.q.device)

        kernel_device = out_residual.device if out_residual is not None else self.device
        num_dofs = self.robot.spec.num_actuated_joints
        batch = robot_state.q.shape[0]

        if out_residual is None:
            out_residual = wp.empty((batch, self.residual_dim), dtype=wp.float32, device=kernel_device)
            row_offset = 0

        wp.launch(
            kernel=compute_trajectory_smoothness_residual_kernel,
            dim=(batch, self._joint_rows, num_dofs),
            inputs=[
                robot_state.q,
                self.q_history,
                self.history_valid,
                self.reference_q,
                self.residual_weight,
                self.boundary_weight,
                self.coefficients,
                self.order,
                self.virtual_hold_frames,
                row_offset,
                num_dofs,
                self.use_mask,
                self._get_frame_mask(),
            ],
            outputs=[out_residual],
            device=kernel_device,
        )
        if self.base_weight is not None:
            if not robot_state.has_floating_base:
                raise ValueError("base_weight requires a floating-base robot state.")
            wp.launch(
                kernel=_compute_base_smoothness_residual_kernel,
                dim=(batch, self.num_frames - 1),
                inputs=[
                    robot_state.T_world_base,
                    self.base_residual_weight,
                    row_offset + self._joint_rows * num_dofs,
                    wp.float32(1e-4),
                    self.use_mask,
                    self._get_frame_mask(),
                ],
                outputs=[out_residual],
                device=kernel_device,
            )
        return out_residual

    def compute_sparse_jacobian_pattern(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        offset: int = 0,
        out_row_indices: Optional[wp.array] = None,
        out_col_indices: Optional[wp.array] = None,
        nnz_offset: int = 0,
        **kwargs: object,
    ) -> SparsityPattern:  # pyright: ignore[reportGeneralTypeIssues]
        robot_state = var_values.get(VarKey.ROBOT)
        if self.device is None:
            self.init_buffers(robot_state.q.device)

        kernel_device = robot_state.q.device
        num_dofs = self.robot.spec.num_actuated_joints
        base_dim = 6 if bool(robot_state.has_floating_base) else 0
        nnz = self._nnz(num_dofs)
        joint_residual_dim = self._joint_rows * num_dofs
        joint_nnz = (self.order + 1) * joint_residual_dim

        row_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_row_indices is None else out_row_indices
        )
        col_indices = (
            wp.empty(nnz, dtype=wp.int32, device=kernel_device) if out_col_indices is None else out_col_indices
        )

        wp.launch(
            kernel=compute_trajectory_smoothness_jacobian_sparse_pattern_kernel,
            dim=(self._joint_rows, num_dofs),
            inputs=[num_dofs, base_dim + num_dofs, base_dim, self.order, self.num_frames, offset, nnz_offset],
            outputs=[row_indices, col_indices],
            device=kernel_device,
        )
        if self.base_weight is not None:
            if base_dim == 0:
                raise ValueError("base_weight requires a floating-base robot state.")
            wp.launch(
                kernel=_compute_base_smoothness_jacobian_sparse_pattern_kernel,
                dim=(self.num_frames - 1, 6),
                inputs=[base_dim + num_dofs, offset + joint_residual_dim, nnz_offset + joint_nnz],
                outputs=[row_indices, col_indices],
                device=kernel_device,
            )

        pattern = SparsityPattern()
        pattern.row_indices = row_indices
        pattern.col_indices = col_indices
        return pattern

    def compute_weighted_sparse_jacobian_values(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        out_jacobian_values: Optional[wp.array] = None,
        offset: int = 0,
        nnz_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        robot_state = var_values.get(VarKey.ROBOT)
        if self.device is None:
            self.init_buffers(robot_state.q.device)

        kernel_device = robot_state.q.device
        num_dofs = self.robot.spec.num_actuated_joints
        batch = robot_state.q.shape[0]
        joint_nnz = (self.order + 1) * self._joint_rows * num_dofs

        if out_jacobian_values is None:
            out_jacobian_values = wp.empty((batch, self._nnz(num_dofs)), dtype=wp.float32, device=kernel_device)

        wp.launch(
            kernel=compute_trajectory_smoothness_jacobian_sparse_values_kernel,
            dim=(batch, self._joint_rows, num_dofs),
            inputs=[
                self.residual_weight,
                self.boundary_weight,
                self.history_valid,
                self.coefficients,
                self.order,
                self.virtual_hold_frames,
                self.num_frames,
                num_dofs,
                nnz_offset,
                max(1, batch // self.history_valid.shape[0]),
                self.use_mask,
                self._get_frame_mask(),
            ],
            outputs=[out_jacobian_values],
            device=kernel_device,
        )
        if self.base_weight is not None:
            if not robot_state.has_floating_base:
                raise ValueError("base_weight requires a floating-base robot state.")
            wp.launch(
                kernel=_compute_base_smoothness_jacobian_sparse_values_kernel,
                dim=(batch, self.num_frames - 1),
                inputs=[
                    robot_state.T_world_base,
                    self.base_residual_weight,
                    nnz_offset + joint_nnz,
                    wp.float32(1e-4),
                    self.use_mask,
                    self._get_frame_mask(),
                ],
                outputs=[out_jacobian_values],
                device=kernel_device,
            )
        return out_jacobian_values


# --- device code ------------------------------------------------------------
@wp.func
def _compute_window_mask_func(
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    batch_idx: int,
    first_frame: int,
    last_frame: int,
) -> float:
    if not use_mask:
        return float(1.0)
    return float(frame_mask[batch_idx, first_frame]) * float(frame_mask[batch_idx, last_frame])


@wp.kernel
def compute_trajectory_smoothness_residual_kernel(
    q_traj: wp.array3d(dtype=wp.float32),
    q_history: wp.array3d(dtype=wp.float32),
    history_valid: wp.array1d(dtype=wp.float32),
    reference_q: wp.array3d(dtype=wp.float32),
    residual_weight: wp.array1d(dtype=wp.float32),
    boundary_weight: wp.array1d(dtype=wp.float32),
    coefficients: wp.array1d(dtype=wp.float32),
    order: int,
    hold_frames: int,
    row_offset: int,
    num_dofs: int,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx, row_idx, dof_idx = wp.tid()  # type: ignore
    residual_idx = row_idx * num_dofs + dof_idx
    last = q_traj.shape[1] - 1
    interior_start = row_idx - (order - 1)
    is_interior = interior_start >= 0 and interior_start + order <= last
    # per-batch buffers broadcast over the launch batch, each by its own row count
    ref_idx = batch_idx // wp.max(1, q_traj.shape[0] // reference_q.shape[0])
    hist_idx = batch_idx // wp.max(1, q_traj.shape[0] // q_history.shape[0])

    diff = wp.float32(0.0)
    for j in range(order + 1):
        frame = interior_start + j
        if frame < 0 and hold_frames == 0:
            # history index 2 aligns with trajectory frame 0
            value = q_history[hist_idx, 2 + frame, dof_idx]
        else:
            held = wp.clamp(frame, 0, last)  # virtual hold frames repeat the start / end
            ref = wp.float32(0.0)
            if is_interior:
                ref = reference_q[ref_idx, held, dof_idx]
            value = q_traj[batch_idx, held, dof_idx] - ref
        diff += coefficients[j] * value

    if is_interior:
        weight = residual_weight[interior_start * num_dofs + dof_idx]
        mask = _compute_window_mask_func(use_mask, frame_mask, batch_idx, interior_start, interior_start + order)
    else:
        weight = history_valid[hist_idx] * boundary_weight[dof_idx]
        if hold_frames > 0:
            weight = boundary_weight[dof_idx]
            if interior_start < -hold_frames:
                weight = wp.float32(0.0)  # window reaches past the virtual frames
        mask = wp.float32(1.0)

    out_residual[batch_idx, row_offset + residual_idx] = weight * diff * mask


@wp.kernel
def compute_trajectory_smoothness_jacobian_sparse_pattern_kernel(
    num_dofs: int,
    single_tangent_dim: int,
    base_dim: int,
    order: int,
    num_frames: int,
    row_offset: int,
    nnz_offset: int,
    row_indices: wp.array1d(dtype=wp.int32),
    col_indices: wp.array1d(dtype=wp.int32),
):
    row_idx, dof_idx = wp.tid()  # type: ignore
    residual_idx = row_idx * num_dofs + dof_idx
    base = nnz_offset + (order + 1) * residual_idx
    # boundary windows keep order + 1 in-range columns; history and held slots fold onto them
    first = wp.clamp(row_idx - (order - 1), 0, num_frames - 1 - order)
    for j in range(order + 1):
        row_indices[base + j] = row_offset + residual_idx
        col_indices[base + j] = (first + j) * single_tangent_dim + base_dim + dof_idx


@wp.kernel
def compute_trajectory_smoothness_jacobian_sparse_values_kernel(
    residual_weight: wp.array1d(dtype=wp.float32),
    boundary_weight: wp.array1d(dtype=wp.float32),
    history_valid: wp.array1d(dtype=wp.float32),
    coefficients: wp.array1d(dtype=wp.float32),
    order: int,
    hold_frames: int,
    num_frames: int,
    num_dofs: int,
    nnz_offset: int,
    # rows launched per history_valid row; this kernel reads no state array to derive it from
    batch_expansion: int,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    values: wp.array2d(dtype=wp.float32),
):
    batch_idx, row_idx, dof_idx = wp.tid()  # type: ignore
    residual_idx = row_idx * num_dofs + dof_idx
    base = nnz_offset + (order + 1) * residual_idx
    last = num_frames - 1
    interior_start = row_idx - (order - 1)

    if interior_start >= 0 and interior_start + order <= last:
        weight = residual_weight[interior_start * num_dofs + dof_idx]
        mask = _compute_window_mask_func(use_mask, frame_mask, batch_idx, interior_start, interior_start + order)
        for j in range(order + 1):
            values[batch_idx, base + j] = coefficients[j] * weight * mask
        return

    weight = history_valid[batch_idx // batch_expansion] * boundary_weight[dof_idx]
    if hold_frames > 0:
        weight = boundary_weight[dof_idx]
        if interior_start < -hold_frames:
            weight = wp.float32(0.0)
    first = wp.clamp(interior_start, 0, last - order)
    for j in range(order + 1):
        values[batch_idx, base + j] = wp.float32(0.0)
    for j in range(order + 1):
        frame = interior_start + j
        if frame >= 0 or hold_frames > 0:
            col = base + wp.clamp(frame, 0, last) - first  # held frames fold onto the start / end column
            values[batch_idx, col] = values[batch_idx, col] + coefficients[j] * weight


@wp.kernel
def _compute_base_smoothness_residual_kernel(
    T_world_base: wp.array2d(dtype=wp_vec7),
    residual_weight: wp.array1d(dtype=wp.float32),
    row_offset: int,
    eps: wp.float32,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    out_residual: wp.array2d(dtype=wp.float32),
):
    batch_idx, frame_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    T_rel = se3_compose_func(
        se3_inverse_func(T_world_base[batch_idx, frame_idx]),
        T_world_base[batch_idx, frame_idx + 1],
    )
    error = se3_log_map_func(T_rel, eps)
    mask = _compute_window_mask_func(use_mask, frame_mask, batch_idx, frame_idx, frame_idx + 1)
    for i in range(6):
        residual_idx = frame_idx * 6 + i
        out_residual[batch_idx, row_offset + residual_idx] = residual_weight[residual_idx] * error[i] * mask


@wp.kernel
def _compute_base_smoothness_jacobian_sparse_pattern_kernel(
    single_tangent_dim: int,
    row_offset: int,
    nnz_offset: int,
    row_indices: wp.array1d(dtype=wp.int32),
    col_indices: wp.array1d(dtype=wp.int32),
):
    frame_idx, residual_dof_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    residual_idx = frame_idx * 6 + residual_dof_idx
    nnz_base = nnz_offset + 12 * residual_idx
    for col_dof in range(6):
        row_indices[nnz_base + col_dof] = row_offset + residual_idx
        col_indices[nnz_base + col_dof] = frame_idx * single_tangent_dim + col_dof
        row_indices[nnz_base + 6 + col_dof] = row_offset + residual_idx
        col_indices[nnz_base + 6 + col_dof] = (frame_idx + 1) * single_tangent_dim + col_dof


@wp.kernel
def _compute_base_smoothness_jacobian_sparse_values_kernel(
    T_world_base: wp.array2d(dtype=wp_vec7),
    residual_weight: wp.array1d(dtype=wp.float32),
    nnz_offset: int,
    eps: wp.float32,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    values: wp.array2d(dtype=wp.float32),
):
    batch_idx, frame_idx = wp.tid()  # pyright: ignore[reportAssignmentType]
    T_rel = se3_compose_func(
        se3_inverse_func(T_world_base[batch_idx, frame_idx]),
        T_world_base[batch_idx, frame_idx + 1],
    )
    jlog_R = se3_jlog_func(T_rel, eps)
    jlog_L = jlog_R * se3_adjoint_func(se3_inverse_func(T_rel))
    mask = _compute_window_mask_func(use_mask, frame_mask, batch_idx, frame_idx, frame_idx + 1)
    for residual_dof_idx in range(6):
        residual_idx = frame_idx * 6 + residual_dof_idx
        weight = residual_weight[residual_idx] * mask
        nnz_base = nnz_offset + 12 * residual_idx
        for col_dof in range(6):
            values[batch_idx, nnz_base + col_dof] = -weight * jlog_L[residual_dof_idx, col_dof]
            values[batch_idx, nnz_base + 6 + col_dof] = weight * jlog_R[residual_dof_idx, col_dof]


if TYPE_CHECKING:
    from robokit.robo import Robot
    from robokit.robo.robot_state import RobotState
