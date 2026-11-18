# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportCallIssue=false
"""Sparse joint position and derivative bounds along a trajectory."""

from math import comb
from typing import TYPE_CHECKING, Any, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import warp as wp

from robokit.opt.var_values import VarKey
from robokit.terms.task import Context, GradientTask, SparseTask, SparsityPattern, sparse_cost_and_gradient
from robokit.utils.warp_utils import wp_device_type


# 5-point central stencils over frames f-2..f+2, ends held as the retimer does, per unit time step
_STENCILS = {
    0: np.array([1.0]),
    1: np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0,
    2: np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0,
    3: np.array([-0.5, 1.0, 0.0, -1.0, 0.5]),
}


# --- task -------------------------------------------------------------------
class TrajectoryLimitTask(SparseTask, GradientTask):
    """Penalize bound violations with ``weight * max(lower - derivative, derivative - upper, 0)``.

    Derivative orders:
    - 0: position.
    - 1: velocity.
    - 2: acceleration.
    - 3: jerk.

    Difference schemes:
    - ``central``: retimer's five-point derivatives with held endpoints; multiple orders share one task and launch.
    - ``forward``: consecutive finite differences without padded endpoints; one order only.

    Broadcast layouts:
    - Bounds/weights: broadcast over joints, with a leading order axis when ``order`` is a sequence.
    - Scalar dt: shared by all plans.
    - Warp-vector dt: one per plan, repeated in instance-major order for candidate batches.

    Time scaling:
    - Default: divide each derivative by dt raised to its order.
    - Explicit ``dt_power``: override the exponent per order.
    - ``timing = (frame_segment, segments, time_weight)``: a bound row's step is its plan's ``exp(VarKey.SEGMENT_STEP)``
      for its frame's segment, a Jacobian column (zero-width bounds keep ``dt``); ``time_weight`` > 0 costs each second.

    Unbuilt templates (`resolve`):
    - ``upper`` may hold one entry per order; unset bounds default to the robot's position and velocity limits.
    - ``lower`` defaults to the robot's lower position limit and to ``-upper`` for derivatives.
    - ``activation`` keeps that fraction of each range away from both bounds, indexed by order.
    - ``normalize`` divides the weights by the bounds.

    Example:
        >>> task = TrajectoryLimitTask(robot, 33, order=3, lower=-500.0, upper=500.0)  # doctest: +SKIP
    """

    compute_weighted_cost_and_gradient = sparse_cost_and_gradient

    def __init__(
        self,
        robot: Optional["Robot"] = None,  # noqa: F821
        num_frames: Optional[int] = None,
        order: Union[int, Sequence[int]] = 1,
        lower: Optional[Union[float, np.ndarray]] = None,
        upper: Optional[Union[float, np.ndarray, Tuple[Any, ...], Context]] = None,
        weight: Union[float, Sequence[float], np.ndarray] = 1.0,
        dt: Union[float, wp.array, Context] = 1.0,
        stencil: Literal["central", "forward"] = "central",
        dt_power: Optional[Sequence[float]] = None,
        # fraction of each range kept away from the position, velocity, acceleration and jerk limits
        activation: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        normalize: bool = False,  # divide weights by the bounds: penalize derivatives jointly at goal-derived timing
        name: str = "",
    ):
        self.name = name
        self.robot = robot
        self.order = order
        self.lower = lower
        self.upper = upper
        self.weight = weight
        self.stencil = stencil
        self.dt_power_arg = dt_power
        self.activation = activation
        self.normalize = normalize
        self.dt = dt
        self.device: Optional[wp_device_type] = None
        self.timing: Optional[Tuple[wp.array, int, float]] = None
        if robot is None or num_frames is None:
            return  # runtime bounds, horizon and robot come from the helper
        orders = [int(o) for o in np.atleast_1d(order)]
        if any(o not in _STENCILS for o in orders) or num_frames < 1:
            raise ValueError("order must be 0, 1, 2, or 3; num_frames must be positive")
        if stencil not in ("central", "forward") or (
            stencil == "forward" and (len(orders) > 1 or num_frames <= orders[0])
        ):
            raise ValueError("stencil must be central or forward, a forward stencil one order with enough frames")
        self.num_frames = num_frames
        self.orders = orders
        self._num_rows = num_frames - (orders[0] if stencil == "forward" else 0)
        self._offset = -2 if stencil == "central" and max(orders) > 0 else 0
        if stencil == "forward":
            stencils = [np.array([(-1) ** (orders[0] - k) * comb(orders[0], k) for k in range(orders[0] + 1)])]
        else:  # order 0 among higher orders: the position at the center of the five frames
            stencils = [
                _STENCILS[o] if o > 0 or self._offset == 0 else np.array([0.0, 0.0, 1.0, 0.0, 0.0]) for o in orders
            ]
        self._coefficients_np = np.stack(stencils).astype(np.float32)
        shape = (len(orders), robot.num_actuated_joints)
        self._bounds_np = np.broadcast_to(np.stack(np.broadcast_arrays(lower, upper), axis=-1), (*shape, 2)).astype(
            np.float32
        )
        if np.any(np.isnan(self._bounds_np)) or np.any(self._bounds_np[..., 0] > self._bounds_np[..., 1]):
            raise ValueError("lower and upper must be ordered, non-NaN bounds")
        self._weight_np = np.broadcast_to(np.asarray(weight, dtype=np.float32), shape).copy()
        self._dt_power_np = np.asarray(orders if dt_power is None else dt_power, dtype=np.float32)
        self.device: Optional[wp_device_type] = None
        self.bounds = None
        self.residual_weight = None
        self.coefficients = None
        self.dt_power = None
        self.set_dt(dt)

    def resolve(self, var_values: "VarValues", context: Mapping[Context, Any]):  # noqa: F821
        """Build a template's robot, horizon, step and bounds from the trajectory `var_values` holds."""
        super().resolve(var_values, context)
        state = var_values.get(self.var_key)
        robot, spec = state.robot, state.robot.spec
        orders = np.atleast_1d(self.order)
        shape = (len(orders), robot.num_actuated_joints)
        upper = self.upper
        if upper is None:
            upper = tuple(
                {0: spec.actuated_joint_limits[:, 1], 1: spec.actuated_joint_velocity_limits}[o] for o in orders
            )
        if isinstance(upper, tuple):
            upper = np.stack([np.broadcast_to(np.asarray(row), shape[1:]) for row in upper]).astype(np.float32)
        lower = self.lower
        if lower is None:
            bounds = np.broadcast_to(upper, shape)
            lower = np.stack([spec.actuated_joint_limits[:, 0] if o == 0 else -b for o, b in zip(orders, bounds)])
        lower = np.broadcast_to(lower, shape).copy()
        upper = np.broadcast_to(upper, shape).copy()
        shrink = np.asarray(self.activation)[orders, None] * (upper - lower)
        lower += shrink
        upper -= shrink
        weight = np.asarray(self.weight, dtype=np.float32)
        if len(orders) > 1 and weight.ndim == 1:
            weight = weight[:, None]
        if self.normalize:
            weight = weight / np.maximum(np.maximum(np.abs(lower), np.abs(upper)), 1e-8)
        TrajectoryLimitTask.__init__(
            self,
            robot,
            state.q.shape[1],
            self.order,
            lower,
            upper,
            weight=weight,
            dt=self.dt,
            stencil=self.stencil,
            dt_power=self.dt_power_arg,
            activation=self.activation,
            normalize=self.normalize,
            name=self.name,
        )

    @staticmethod
    def fuse(tasks: Sequence[Any], dt: Any, keep: Sequence[Any] = ()) -> List[Any]:
        """Fuse central bounds at `dt` and all-frame `abs` position limits into one launch, skipping tasks in `keep`."""
        from robokit.terms.dense.position_limit import PositionLimit
        from robokit.terms.trajectory_task import TrajectoryTask

        tasks, fusable = list(tasks), []
        for index, task in enumerate(tasks):
            if dt is None or any(task is kept for kept in keep):
                continue
            dense = task.dense_task
            if (
                isinstance(task, TrajectoryTask)
                and isinstance(dense, PositionLimit)
                and dense.residual_mode == "abs"
                and dense.base_axis is None
                and dense.mask is None
                and len(task.frames) == task.num_frames
            ):  # as a zeroth-order bound it joins the fused launch
                limits = dense._joint_limits_np
                if limits is None:
                    limits = dense.robot.spec.actuated_joint_limits.astype(np.float32)
                task = tasks[index] = TrajectoryLimitTask(
                    dense.robot,
                    task.num_frames,
                    0,
                    limits[:, 0],
                    limits[:, 1],
                    weight=1.0 if dense.weight is None else dense.weight,  # PositionLimit weighs None as 1
                    dt=dt,
                    dt_power=(0.0,),
                    name=task.name,
                )
            if (
                isinstance(task, TrajectoryLimitTask)
                and task.stencil == "central"
                and task.dt is dt
                and not task.normalize
            ):
                fusable.append(task)
        if len(fusable) < 2:
            return tasks
        fused = TrajectoryLimitTask(
            fusable[0].robot,
            fusable[0].num_frames,
            [order for task in fusable for order in task.orders],
            np.concatenate([task._bounds_np[..., 0] for task in fusable]),
            np.concatenate([task._bounds_np[..., 1] for task in fusable]),
            weight=np.concatenate([task._weight_np for task in fusable]),
            dt=dt,
            dt_power=np.concatenate([task._dt_power_np for task in fusable]),
            name="limits",
        )
        first = tasks.index(fusable[0])
        tasks = [task for task in tasks if all(task is not other for other in fusable)]
        tasks.insert(first, fused)
        return tasks

    def init_buffers(self, device: wp_device_type):
        self.device = device
        self.bounds = wp.from_numpy(self._bounds_np, dtype=wp.float32, device=device)
        self.residual_weight = wp.from_numpy(self._weight_np, dtype=wp.float32, device=device)
        self.coefficients = wp.from_numpy(self._coefficients_np, dtype=wp.float32, device=device)
        self.dt_power = wp.from_numpy(self._dt_power_np, dtype=wp.float32, device=device)
        self.set_dt(self.dt)

    def set_dt(self, dt: Union[float, wp.array]):
        """Set a positive scalar time step or a float32 Warp vector with one entry per plan."""
        if isinstance(dt, wp.array):
            if dt.ndim != 1 or dt.shape[0] == 0 or dt.dtype != wp.float32:
                raise ValueError("dt must be a nonempty float32 vector")
            if self.device is not None and dt.device != wp.get_device(self.device):
                raise ValueError("dt must be on the task's device")
            self.dt = dt
        else:
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError("dt must be positive and finite")
            self.dt = (
                wp.full(1, float(dt), dtype=wp.float32, device=self.device) if self.device is not None else float(dt)
            )

    def _segment_steps(self, var_values: "VarValues") -> Tuple[Optional[wp.array], Optional[wp.array], float, float]:
        """Each plan's segment log steps, each frame interval's segment, their tangent unit and time cost weight."""
        frame_segment, _, time_weight = self.timing or (None, 0, 0.0)
        steps = None if frame_segment is None else var_values.get(VarKey.SEGMENT_STEP)
        return (None, None, 1.0, 0.0) if steps is None else (steps.x, frame_segment, steps.unit, time_weight)

    @property
    def residual_dim(self) -> int:
        cost_rows = self.timing[1] if self.timing is not None and self.timing[2] > 0.0 else 0
        return len(self.orders) * self._num_rows * self.robot.spec.num_actuated_joints + cost_rows

    def compute_weighted_residual(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
        **kwargs: object,
    ) -> wp.array:
        q = var_values.get(VarKey.ROBOT).q
        if self.device is None:
            self.init_buffers(q.device)
        if q.shape[0] % self.dt.shape[0]:
            raise ValueError("the plan batch must be a multiple of the dt batch")
        if out_residual is None:
            out_residual = wp.empty((q.shape[0], self.residual_dim), dtype=wp.float32, device=q.device)
            row_offset = 0
        wp.launch(
            _limit_kernel,
            dim=(q.shape[0], len(self.orders) * self._num_rows, q.shape[2]),
            inputs=[
                q,
                self.coefficients,
                self.bounds,
                self.dt,
                self.dt_power,
                self._num_rows,
                self.residual_weight,
                self._offset,
                self.use_mask,
                self._get_frame_mask(),
                *self._segment_steps(var_values),
                row_offset,
                0,
                0,
            ],
            outputs=[out_residual, out_residual],
            device=q.device,
        )
        return out_residual

    def fill_residual_and_jacobian(
        self,
        var_values: "VarValues",  # noqa: F821
        out_residual: wp.array,
        row_offset: int,
        out_jacobian: Optional[wp.array] = None,
        nnz_offset: int = 0,
    ):
        """Write the residual and, when asked, the Jacobian values in one launch."""
        if out_jacobian is None:
            self.compute_weighted_residual(var_values, out_residual=out_residual, row_offset=row_offset)
            return
        if self.use_mask:
            self._build_frame_mask(var_values.device, var_values.batch_size)
        q = var_values.get(VarKey.ROBOT).q
        if self.device is None:
            self.init_buffers(q.device)
        if q.shape[0] % self.dt.shape[0]:
            raise ValueError("the plan batch must be a multiple of the dt batch")
        wp.launch(
            _limit_kernel,
            dim=(q.shape[0], len(self.orders) * self._num_rows, q.shape[2]),
            inputs=[
                q,
                self.coefficients,
                self.bounds,
                self.dt,
                self.dt_power,
                self._num_rows,
                self.residual_weight,
                self._offset,
                self.use_mask,
                self._get_frame_mask(),
                *self._segment_steps(var_values),
                row_offset,
                nnz_offset,
                2,
            ],
            outputs=[out_residual, out_jacobian],
            device=q.device,
        )

    def compute_sparse_jacobian_pattern(
        self,
        var_values: "VarValues",  # noqa: F821
        *args: object,
        offset: int = 0,
        out_row_indices: Optional[wp.array] = None,
        out_col_indices: Optional[wp.array] = None,
        nnz_offset: int = 0,
        **kwargs: object,
    ) -> SparsityPattern:
        state = var_values.get(VarKey.ROBOT)
        if self.device is None:
            self.init_buffers(state.q.device)
        num_dofs = self.robot.spec.num_actuated_joints
        base_dim = 6 if bool(state.has_floating_base) else 0
        segments = 0 if self.timing is None else self.timing[1]
        width, limit_rows = self._coefficients_np.shape[1] + segments, len(self.orders) * self._num_rows * num_dofs
        nnz = self.residual_dim * width
        rows = wp.empty(nnz, dtype=wp.int32, device=state.q.device) if out_row_indices is None else out_row_indices
        cols = wp.empty(nnz, dtype=wp.int32, device=state.q.device) if out_col_indices is None else out_col_indices
        wp.launch(
            _limit_pattern_kernel,
            dim=(len(self.orders) * self._num_rows, num_dofs),
            inputs=[
                self.num_frames,
                self._num_rows,
                num_dofs,
                base_dim + num_dofs,
                base_dim,
                self._coefficients_np.shape[1],
                self._offset,
                segments,
                var_values.tangent_offset(VarKey.SEGMENT_STEP) if segments else 0,  # the step columns
                limit_rows,
                self.residual_dim - limit_rows,
                offset,
                nnz_offset,
            ],
            outputs=[rows, cols],
            device=state.q.device,
        )
        pattern = SparsityPattern()
        pattern.row_indices = rows
        pattern.col_indices = cols
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
        q = var_values.get(VarKey.ROBOT).q
        if self.device is None:
            self.init_buffers(q.device)
        if q.shape[0] % self.dt.shape[0]:
            raise ValueError("the plan batch must be a multiple of the dt batch")
        if out_jacobian_values is None:
            width = self._coefficients_np.shape[1] + (0 if self.timing is None else self.timing[1])
            out_jacobian_values = wp.empty((q.shape[0], self.residual_dim * width), dtype=wp.float32, device=q.device)
        wp.launch(
            _limit_kernel,
            dim=(q.shape[0], len(self.orders) * self._num_rows, q.shape[2]),
            inputs=[
                q,
                self.coefficients,
                self.bounds,
                self.dt,
                self.dt_power,
                self._num_rows,
                self.residual_weight,
                self._offset,
                self.use_mask,
                self._get_frame_mask(),
                *self._segment_steps(var_values),
                0,
                nnz_offset,
                1,
            ],
            outputs=[out_jacobian_values, out_jacobian_values],
            device=q.device,
        )
        return out_jacobian_values


# --- kernels ----------------------------------------------------------------
@wp.kernel
def _limit_kernel(
    q: wp.array3d(dtype=wp.float32),
    coefficients: wp.array2d(dtype=wp.float32),  # (orders, stencil)
    bounds: wp.array3d(dtype=wp.float32),  # (orders, dofs, 2)
    dt: wp.array1d(dtype=wp.float32),
    dt_power: wp.array1d(dtype=wp.float32),  # (orders,)
    num_rows: int,  # frames per order
    weight: wp.array2d(dtype=wp.float32),  # (orders, dofs)
    frame_offset: int,
    use_mask: wp.bool,
    frame_mask: wp.array2d(dtype=wp.uint8),
    log_step: wp.array2d(dtype=wp.float32),  # (plans, segments) log steps, or empty at `dt`
    frame_segment: wp.array1d(dtype=wp.int32),  # each frame interval's segment
    unit: float,  # the log steps' tangent unit
    time_weight: float,  # >0: the segments' time cost per second
    row_offset: int,
    nnz_offset: int,
    jacobian: int,  # 0: the residual, 1: the Jacobian values, 2: both
    out_residual: wp.array2d(dtype=wp.float32),
    out_jacobian: wp.array2d(dtype=wp.float32),
):
    b, r, d = wp.tid()  # pyright: ignore[reportAssignmentType]
    t, f = r // num_rows, r % num_rows
    n = q.shape[1]
    scale = 1.0 / wp.pow(dt[b // (q.shape[0] // dt.shape[0])], dt_power[t])
    stencil, segments = coefficients.shape[1], log_step.shape[1]
    if time_weight > 0.0 and d == 0 and r < segments:  # segment r's cost row: half the square of sqrt(2 weight time)
        time = float(0.0)
        for interval in range(n - 1):
            time += wp.where(frame_segment[interval] == r, wp.exp(log_step[b, r]), 0.0)
        cost_row = coefficients.shape[0] * num_rows * q.shape[2] + r
        if jacobian != 1:
            out_residual[b, row_offset + cost_row] = wp.sqrt(2.0 * time_weight * time)
        for k in range(wp.where(jacobian != 0, stencil + segments, 0)):
            entry = wp.where(k == stencil + r, wp.sqrt(0.5 * time_weight * time) * unit, 0.0)
            out_jacobian[b, nnz_offset + cost_row * (stencil + segments) + k] = entry
    segment = int(-1)  # a bound's segment; zero-width bounds (penalties) stay at `dt`
    if segments > 0 and bounds[t, d, 0] < bounds[t, d, 1]:
        segment = frame_segment[wp.min(f, n - 2)]
        scale = wp.exp(-dt_power[t] * log_step[b, segment])
    value = float(0.0)
    mask = float(1.0)
    for k in range(coefficients.shape[1]):
        frame = wp.clamp(f + k + frame_offset, 0, n - 1)
        value += coefficients[t, k] * q[b, frame, d]
        if use_mask and coefficients[t, k] != 0.0:
            mask *= float(frame_mask[b, frame])
    value *= scale
    below, above = bounds[t, d, 0] - value, value - bounds[t, d, 1]
    row = r * q.shape[2] + d
    if jacobian != 1:
        out_residual[b, row_offset + row] = weight[t, d] * wp.max(wp.max(below, above), 0.0) * mask
    if jacobian == 0:
        return
    sign = float(1.0) if above > 0.0 else float(-1.0)
    for k in range(stencil + segments):
        entry = float(0.0)
        if (below > 0.0 or above > 0.0) and k < stencil:
            entry = weight[t, d] * sign * coefficients[t, k] * scale * mask
        if (below > 0.0 or above > 0.0) and segment >= 0 and k == stencil + segment:  # d value / d log step
            entry = -weight[t, d] * sign * dt_power[t] * value * mask * unit
        out_jacobian[b, nnz_offset + row * (stencil + segments) + k] = entry


@wp.kernel
def _limit_pattern_kernel(
    num_frames: int,
    num_rows: int,  # frames per order
    num_dofs: int,
    single_tangent_dim: int,
    base_dim: int,
    stencil_size: int,
    frame_offset: int,
    segments: int,  # segment step columns, from `step_column` on
    step_column: int,
    limit_rows: int,  # rows before the segments' time cost rows
    cost_rows: int,
    row_offset: int,
    nnz_offset: int,
    row_indices: wp.array1d(dtype=wp.int32),
    col_indices: wp.array1d(dtype=wp.int32),
):
    r, d = wp.tid()  # pyright: ignore[reportAssignmentType]
    f = r % num_rows
    row = r * num_dofs + d
    width = stencil_size + segments
    for k in range(width):
        row_indices[nnz_offset + row * width + k] = row_offset + row
        col_indices[nnz_offset + row * width + k] = (
            wp.clamp(f + k + frame_offset, 0, num_frames - 1) * single_tangent_dim + base_dim + d
        )
        if k >= stencil_size:
            col_indices[nnz_offset + row * width + k] = step_column + k - stencil_size
        if d == 0 and r < cost_rows:  # segment r's time, in its step's column
            row_indices[nnz_offset + (limit_rows + r) * width + k] = row_offset + limit_rows + r
            col_indices[nnz_offset + (limit_rows + r) * width + k] = step_column + r


if TYPE_CHECKING:
    from robokit.opt.var_values import VarValues
    from robokit.robo import Robot
