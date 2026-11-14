# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
import abc
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import warp as wp

from robokit.opt.var_values import Var, VarValues
from robokit.terms.task import SparseTask, Task
from robokit.utils.warp_utils import warp_sum, wp_device_type


# Called with live buffers at iteration 0 and after each completed iteration.
IterationCallback = Callable[[int, VarValues, wp.array], None]


@wp.kernel
def aggregate_residuals_to_costs_kernel(
    residuals: wp.array2d(dtype=wp.float32),
    costs: wp.array(dtype=wp.float32),
):
    """Aggregate squared residuals to compute cost per batch element."""
    batch_idx = wp.tid()

    cost = wp.float32(0.0)
    for i in range(residuals.shape[1]):
        r = residuals[batch_idx, i]
        cost += r * r

    costs[batch_idx] = wp.float32(0.5) * cost


@wp.kernel
def apply_dof_mask_kernel(vec: wp.array2d(dtype=wp.float32), mask: wp.array1d(dtype=wp.float32)):
    """Zero masked tangent dims of a (B, D) vector (gradient or search direction)."""
    batch_idx, dof_idx = wp.tid()  # type: ignore[misc]
    vec[batch_idx, dof_idx] = vec[batch_idx, dof_idx] * mask[dof_idx]


@wp.kernel
def _sparse_jtr_split_kernel(
    residuals: wp.array2d(dtype=wp.float32),  # (B, R) shared residual buffer (all terms)
    jac_values: wp.array2d(dtype=wp.float32),  # (B, NNZ) shared sparse-Jacobian values (all terms)
    value_perm: wp.array1d(dtype=wp.int32),  # column-sorted nnz order
    row_perm: wp.array1d(dtype=wp.int32),  # residual row per sorted nnz
    split_cols: wp.array1d(dtype=wp.int32),  # gradient column per split
    split_offsets: wp.array1d(dtype=wp.int32),  # sorted-nnz range per split
    rows_per_split: int,  # > 0: also add 0.5 |r|^2 into costs, this many residual rows per split
    lanes: int,  # threads per warp: a CUDA warp's 32, or 1 on CPU
    costs: wp.array1d(dtype=wp.float32),  # (B,) accumulated with rows_per_split (caller zeroes)
    gradient: wp.array2d(dtype=wp.float32),  # (B, D) accumulated (caller zeroes)
):
    # Reduce merged Jᵀr once; split long CSC columns across atomics to avoid serial work.
    batch_idx, split_idx = wp.tid()  # type: ignore[misc]
    if rows_per_split > 0:  # launched over whole warps of splits, each warp adding its rows' cost once
        num_rows = residuals.shape[1]
        cost = float(0.0)
        for k in range(split_idx * rows_per_split, wp.min((split_idx + 1) * rows_per_split, num_rows)):
            cost += residuals[batch_idx, k] * residuals[batch_idx, k]
        cost = warp_sum(cost)
        if split_idx % lanes == 0:
            wp.atomic_add(costs, batch_idx, 0.5 * cost)
    if split_idx >= split_cols.shape[0]:
        return
    g = float(0.0)
    for i in range(split_offsets[split_idx], split_offsets[split_idx + 1]):
        g += jac_values[batch_idx, value_perm[i]] * residuals[batch_idx, row_perm[i]]
    if g != 0.0:
        wp.atomic_add(gradient, batch_idx, split_cols[split_idx], g)


TILE_THREADS = 32
_KERNEL_CACHE: Dict[tuple, object] = {}


@wp.kernel(enable_backward=False, module="unique")
def _strided_cost_kernel(
    residuals: wp.array2d(dtype=wp.float32),
    add: bool,  # add to `costs`, else assign: zeroing them first would cost a captured memset per launch
    costs: wp.array1d(dtype=wp.float32),
):
    """Per-row `0.5‖r‖²`; block threads stride over the rows, so R stays out of the kernel."""
    batch_idx, lane = wp.tid()  # type: ignore[misc]
    cost = float(0.0)
    for k in range(lane, residuals.shape[1], wp.block_dim()):
        v = residuals[batch_idx, k]
        cost += v * v
    cost = 0.5 * wp.tile_sum(wp.tile(cost))[0]
    if lane == 0:
        costs[batch_idx] = wp.where(add, costs[batch_idx] + cost, cost)


def _launch_strided_cost(residuals: wp.array, costs: wp.array, add: bool = False) -> None:
    wp.launch_tiled(
        _strided_cost_kernel,
        dim=[residuals.shape[0]],
        inputs=[residuals, add],
        outputs=[costs],
        block_dim=TILE_THREADS if residuals.shape[1] < 256 else 128,  # few rows leave most of a wide block idle
        device=residuals.device,
    )


def _jtr_kernel(R: int, D: int):
    """Tiled per-row `g = Jᵀ r` for a dense `(B, R, D)` Jacobian."""
    key = ("jtr", R, D)
    if key not in _KERNEL_CACHE:
        Rc, Dc = wp.constant(R), wp.constant(D)

        def jtr(
            jacobians: wp.array3d(dtype=wp.float32),
            residuals: wp.array2d(dtype=wp.float32),
            gradient: wp.array2d(dtype=wp.float32),
        ):
            batch_idx = wp.tid()
            J = wp.tile_load(jacobians[batch_idx], shape=(Rc, Dc))
            r = wp.tile_load(residuals[batch_idx], shape=(Rc,))
            Jt = wp.tile_transpose(J)
            r_2d = wp.tile_reshape(r, shape=(Rc, 1))
            # Warp's CPU matmul reads the accumulator even with beta=0; 0 * uninitialized NaN stays NaN.
            g_2d = wp.tile_zeros(shape=(Dc, 1), dtype=wp.float32, storage="shared")
            wp.tile_matmul(Jt, r_2d, g_2d)
            g = wp.tile_reshape(g_2d, shape=(Dc,))
            wp.tile_store(gradient[batch_idx], g)

        jtr.__name__ = jtr.__qualname__ = f"_jtr_{R}x{D}"
        _KERNEL_CACHE[key] = wp.kernel(enable_backward=False, module="unique")(jtr)
    return _KERNEL_CACHE[key]


def _jv_kernel(R: int, D: int):
    """Tiled per-row `J v` for a dense `(B, R, D)` Jacobian (LM's CG operator)."""
    key = ("jv", R, D)
    if key not in _KERNEL_CACHE:
        Rc, Dc = wp.constant(R), wp.constant(D)

        def jv(
            jacobians: wp.array3d(dtype=wp.float32),
            vec: wp.array2d(dtype=wp.float32),
            out: wp.array2d(dtype=wp.float32),
        ):
            batch_idx = wp.tid()
            J = wp.tile_load(jacobians[batch_idx], shape=(Rc, Dc))
            v = wp.tile_reshape(wp.tile_load(vec[batch_idx], shape=(Dc,)), shape=(Dc, 1))
            # Warp's CPU matmul reads the accumulator even with beta=0; 0 * uninitialized NaN stays NaN.
            out_2d = wp.tile_zeros(shape=(Rc, 1), dtype=wp.float32, storage="shared")
            wp.tile_matmul(J, v, out_2d)
            wp.tile_store(out[batch_idx], wp.tile_reshape(out_2d, shape=(Rc,)))

        jv.__name__ = jv.__qualname__ = f"_jv_{R}x{D}"
        _KERNEL_CACHE[key] = wp.kernel(enable_backward=False, module="unique")(jv)
    return _KERNEL_CACHE[key]


@dataclass(frozen=True)
class OptimizerConfig:
    max_iter: int = 100
    """Iterations per solve."""
    early_stopping_interval: int = 1
    """Check the stop rule every this many iterations; 0 never stops early (each check syncs the host)."""
    cost_tol: float = 1e-9
    """The solve stops once every batch element's cost is at most this."""
    cuda_graph_mode: Literal["none", "iter", "full"] = "none"
    """Capture per width: "iter" replays one step; "full" replays the solve without callbacks or early stopping."""

    def build(
        self, terms: Sequence[Task], device: wp_device_type, active_dof_mask: Optional[wp.array] = None
    ) -> "Optimizer":
        """The optimizer a `PopulationSolver` steps with this config."""
        raise ValueError(f"{type(self).__name__} is not supported by PopulationSolver; use its optimizer directly")


class Optimizer(abc.ABC):
    """Warp optimizer lifecycle with shared Jacobian machinery for gradient optimizers and LM's CG step.

    Solve sequence:
    1. `reset`: build once, slice to batch size, optionally capture a graph, then call `_start`.
    2. Repeat `step` `max_iter` times, replaying the iteration graph or running `_step_body`.

    Subclass hooks:
    - `_build(var)`: allocate all buffers at the first var's batch size (`max_batch_size`).
    - `_BATCH_BUFFERS`: identify batch-leading arrays and vars to slice per solve.
    - `_start(var)`: initialize per-solve damping/history/schedules and starting cost.
    - `_step_body(var)`: run one iteration in place using kernels only.
    """

    terms: Sequence[Task]
    config: OptimizerConfig
    _BATCH_BUFFERS: Sequence[str] = ()
    _config_type: type = OptimizerConfig

    def __init__(
        self,
        terms: Union[Task, Sequence[Task]],
        config: Optional[OptimizerConfig] = None,
        device: Optional[wp_device_type] = None,
        active_dof_mask: Optional[wp.array] = None,
    ):
        """`active_dof_mask` is float32 `[tangent_dim]`: 0 freezes a tangent dim, 1 keeps it active."""
        self.terms = list(terms) if isinstance(terms, Sequence) else [terms]
        self.config = config if config is not None else self._config_type()
        self.device = wp.get_device(device) if device is None or isinstance(device, str) else device
        self._active_dof_mask = active_dof_mask
        self._built = False
        self._graphs: Dict[tuple, object] = {}  # (batch size, mode) -> captured graph
        self._graph_vars: Dict[tuple, VarValues] = {}  # ... -> the var it was recorded on
        self._iter = 0
        self._task_stop = False  # a term asked to stop (see `EagerTask.on_step`)
        self._iteration_callback: Optional[IterationCallback] = None

    # ------------------------------------------------------------------ lifecycle
    def solve(
        self, var: VarValues, iteration_callback: Optional[IterationCallback] = None
    ) -> Tuple[VarValues, wp.array]:
        """Optimize `var` in place; returns it with the final cost per batch element."""
        cfg = self.config
        capturing = self.device.is_cuda and wp.get_stream(self.device).is_capturing
        graph_solve = cfg.cuda_graph_mode == "full" and self.device.is_cuda and not capturing
        if cfg.cuda_graph_mode == "full" and self.device.is_cuda and iteration_callback is not None:
            raise ValueError("iteration_callback is incompatible with full-solve CUDA graph capture")
        self._iteration_callback = iteration_callback
        self.reset(var)
        if graph_solve:
            wp.capture_launch(self._graphs[(self.batch_size, "full")])
            return var, self.costs
        if iteration_callback is not None:
            iteration_callback(0, var, self.costs)
        interval = cfg.early_stopping_interval if not capturing else 0
        for i in range(cfg.max_iter):
            self.step(var)
            if iteration_callback is not None:
                iteration_callback(i + 1, var, self.costs)
            if interval > 0 and (i + 1) % interval == 0:
                if self._task_stop or bool(np.all(self.costs.numpy() <= cfg.cost_tol)):
                    break
        return var, self.costs

    def reset(self, var: VarValues, num_steps: Optional[int] = None) -> None:
        """Prepare to iterate from `var`; `num_steps` (default `max_iter`) steps follow and L-BFGS finishes the last."""
        wp.init()
        self._num_steps = self.config.max_iter if num_steps is None else num_steps
        if not self._built:
            self._build(var)
            self._built = True
        self._slice_buffers_to_batch_size(var.batch_size)
        mode = self.config.cuda_graph_mode
        capturing = self.device.is_cuda and wp.get_stream(self.device).is_capturing
        graph_solve = mode == "full" and self.device.is_cuda and not capturing
        key = (self.batch_size, mode)
        # a graph reads the arrays it was recorded on, so another var of the same width records its own
        if mode != "none" and self.device.is_cuda and not capturing and self._graph_vars.get(key) is not var:
            self._capture(var, mode)
            self._graph_vars[key] = var
        if not graph_solve:  # a whole-solve graph replays `_start` itself
            self._start(var)
        self._iter = 0
        self._task_stop = False

    def step(self, var: VarValues) -> None:
        """One iteration in place; an outer capture records kernels instead of replaying a graph."""
        graph = self._graphs.get((self.batch_size, "iter"))
        if graph is not None and not wp.get_stream(self.device).is_capturing:
            wp.capture_launch(graph)
        else:
            self._step_body(var)
        self._iter += 1

    def _capture(self, var: VarValues, mode: str) -> None:
        """Capture this width: one iteration ("iter") or the whole solve including `_start` ("full").

        Capture sequence:
        1. Warm up eagerly to allocate buffers and compile kernels.
        2. Restore the var and rewind per-solve state with `_start`.
        3. Record the warmed computation.
        """
        iters = 1 if mode == "iter" else self.config.max_iter
        initial_var = var.clone()
        var.invalidate()
        self._start(var)
        for _ in range(min(iters, 2)):
            self._step_body(var)
        wp.synchronize()
        var.accept(wp.ones(var.batch_size, dtype=wp.int32, device=self.device), initial_var)
        wp.synchronize()
        var.invalidate()
        self._start(var)
        self._iter = 0
        # Invalidate derived caches before recording so every graph replay recomputes FK, spheres, and active pairs.
        var.invalidate()
        with wp.ScopedCapture(device=self.device) as capture:
            if mode == "full":
                self._start(var)
            for _ in range(iters):
                self._step_body(var)
                self._iter += 1
        self._graphs[(self.batch_size, mode)] = capture.graph
        var.invalidate()

    def _build(self, var: VarValues) -> None:
        raise NotImplementedError

    def _start(self, var: VarValues) -> None:
        """Per-solve state from `var`; `costs` is valid afterwards unless the first step scores it (LM)."""
        raise NotImplementedError

    def _step_body(self, var: VarValues) -> None:
        raise NotImplementedError

    def _slice_buffers_to_batch_size(self, batch_size: int) -> None:
        """Slice max-capacity buffers into shared-storage batch views without allocating during CUDA capture.

        Buffer layouts:
        - Ordinary arrays/vars: first `batch_size` entries.
        - Line-search buffers: retain `k` entries per batch element.
        - Term inputs: allocate one target and pass `target[:batch_size]`.
        """
        if batch_size > self.max_batch_size:
            raise ValueError(f"batch size {batch_size} exceeds the {self.max_batch_size} this optimizer was built for")
        full = self.__dict__.setdefault("_max_size_buffers", {})
        for name in self._BATCH_BUFFERS:
            buf = full.setdefault(name, getattr(self, name, None))
            if buf is None:
                continue
            full_size = buf.batch_size if isinstance(buf, Var) else buf.shape[0]
            size = batch_size * (full_size // self.max_batch_size)
            if size < full_size:
                buf = buf.view(size) if isinstance(buf, Var) else buf[:size]
            setattr(self, name, buf)
        self.batch_size = batch_size

    @property
    def is_all_sparse(self) -> bool:
        return all(isinstance(t, SparseTask) for t in self.terms)

    def _build_terms(self, var: VarValues, residuals_require_grad: bool = False):
        terms, device = self.terms, self.device
        self.max_batch_size = self.batch_size = var.batch_size
        self.total_tangent_dim = var.tangent_dim
        self.residual_dims = [term.residual_dim for term in terms]
        self.total_residual_dim = sum(self.residual_dims)

        # residual offsets
        offset = 0
        self.residual_offsets = []
        for term in terms:
            self.residual_offsets.append(offset)
            offset += term.residual_dim

        # common buffers
        self.residuals = wp.zeros(
            (self.max_batch_size, self.total_residual_dim),
            dtype=wp.float32,
            device=device,
            requires_grad=residuals_require_grad,
        )
        # Jacobian storage (dense buffer or sparse machinery) is allocated by `_build_jacobian`.
        self.jacobians: Optional[wp.array] = None
        self.costs = wp.zeros((self.max_batch_size,), dtype=wp.float32, device=device)

        # CUDA streams for parallel term evaluation
        if self.device is not None and self.device.is_cuda:
            self.term_streams = [wp.Stream(self.device) for _ in terms]
            self.sync_events = [wp.Event(self.device) for _ in terms]
        else:
            self.term_streams = [None] * len(terms)
            self.sync_events = [None] * len(terms)

        # Build residual and jacobian functions; each term routes its own leaf and Jacobian column block via `var_key`.
        self._residual_fns = [
            lambda offset, var, buf, _fn=t.compute_weighted_residual: _fn(var, out_residual=buf, row_offset=offset)
            for t in terms
        ]
        self._jacobian_fns = [
            lambda offset, var, buf, _fn=t.compute_weighted_jacobian: _fn(var, out_jacobian=buf, row_offset=offset)
            for t in terms
        ]

        # analytic_jacobian storage; _build_jacobian flips this to sparse when the terms allow it
        self._sparse_jac = False

    def _parallel_for_objectives(self, fns, *extra):
        """Run prebuilt fns across terms on parallel CUDA streams."""
        use_streams = all(t.parallel_safe for t in self.terms) and self.device is not None and self.device.is_cuda
        if use_streams:
            main = wp.get_stream(self.device)
            init_evt = main.record_event()
            for offset, fn, term_stream, sync_event in zip(
                self.residual_offsets, fns, self.term_streams, self.sync_events
            ):
                term_stream.wait_event(init_evt)
                with wp.ScopedStream(term_stream):
                    fn(offset, *extra)
                term_stream.record_event(sync_event)
            for sync_event in self.sync_events:
                main.wait_event(sync_event)
        else:
            for offset, fn in zip(self.residual_offsets, fns):
                fn(offset, *extra)

    def _precompute_terms(self, var: VarValues, need_gradient: bool):
        for term in self.terms:
            term.precompute(var, need_gradient=need_gradient)

    def _build_jacobian(self, var: VarValues, jac_rows: int) -> None:
        """Allocate analytic_jacobian storage, building structural patterns once from the first-solve var.

        Storage paths:
        - All SparseTask terms: merged CSC, with each term writing residuals/values at shared row/nnz offsets.
        - Other terms: dense (B,R,D) buffer.

        Sparse reductions:
        - One cost launch across all terms.
        - One merged-CSC gradient launch across all terms.
        """
        self._sparse_jac = self.is_all_sparse
        device = self.device
        if not self._sparse_jac:
            self.jacobians = wp.zeros(
                (self.max_batch_size, self.total_residual_dim, self.total_tangent_dim),
                dtype=wp.float32,
                device=device,
            )
            self._dense_jtr_kernel = _jtr_kernel(self.total_residual_dim, self.total_tangent_dim)
            return
        rows_list: list = []
        cols_list: list = []
        nnz_offsets: list = []
        nnz_total = 0
        for t, row_off in zip(self.terms, self.residual_offsets):
            pattern = t.compute_sparse_jacobian_pattern(var, offset=row_off)
            rows_np = pattern.row_indices.numpy()
            cols_np = pattern.col_indices.numpy().astype(np.int64) + var.tangent_offset(t.var_key)
            nnz_offsets.append(nnz_total)
            nnz_total += rows_np.shape[0]
            rows_list.append(rows_np)
            cols_list.append(cols_np)
        merged_cols = np.concatenate(cols_list)
        merged_rows = np.concatenate(rows_list)
        # Row-major view for J·v (LM's CG), valid when every term emits its rows in order.
        self._sparse_cols = wp.from_numpy(merged_cols.astype(np.int32), dtype=wp.int32, device=device)
        row_starts = np.searchsorted(merged_rows, np.arange(self.total_residual_dim + 1)).astype(np.int32)
        self._sparse_row_starts = wp.from_numpy(row_starts, dtype=wp.int32, device=device)
        self._sparse_rows_sorted = bool(np.all(np.diff(merged_rows) >= 0))
        # Column-sorted splits of at most `split_size` nonzeros, so long columns don't serialize on one atomic.
        split_size = 8
        order = np.argsort(merged_cols, kind="stable")
        cols_sorted = merged_cols[order]
        col_run_starts = np.flatnonzero(np.r_[True, cols_sorted[1:] != cols_sorted[:-1]])
        split_starts = np.concatenate(
            [np.arange(s, e, split_size) for s, e in zip(col_run_starts, np.r_[col_run_starts[1:], nnz_total])]
        )
        self._sparse_value_perm = wp.from_numpy(order.astype(np.int32), dtype=wp.int32, device=device)
        self._sparse_row_perm = wp.from_numpy(merged_rows[order].astype(np.int32), dtype=wp.int32, device=device)
        self._sparse_split_cols = wp.from_numpy(
            cols_sorted[split_starts].astype(np.int32), dtype=wp.int32, device=device
        )
        self._sparse_split_offsets = wp.from_numpy(
            np.append(split_starts, nnz_total).astype(np.int32), dtype=wp.int32, device=device
        )
        self._sparse_num_splits = int(split_starts.shape[0])
        self._sparse_jac_values = wp.zeros((jac_rows, nnz_total), dtype=wp.float32, device=device)
        self._sparse_fill_fns = [
            lambda offset, var, res, jac, _t=t, _n=nnz_off: _t.fill_residual_and_jacobian(
                var, res, offset, out_jacobian=jac, nnz_offset=_n
            )
            for t, nnz_off in zip(self.terms, nnz_offsets)
        ]

    def _fill_jacobian(self, var: VarValues, residuals: wp.array, need_jacobian: bool) -> None:
        """Fill the shared residual buffer and (optionally) the Jacobian storage, term by term."""
        if self._sparse_jac:
            self._parallel_for_objectives(
                self._sparse_fill_fns, var, residuals, self._sparse_jac_values if need_jacobian else None
            )
            return
        self._parallel_for_objectives(self._residual_fns, var, residuals)
        if need_jacobian:
            self.jacobians.zero_()
            self._parallel_for_objectives(self._jacobian_fns, var, self.jacobians)

    def _reduce_cost_gradient(
        self, residuals: wp.array, costs: wp.array, gradient: Optional[wp.array], accumulate: bool = False
    ) -> None:
        """Reduce total cost and optional J^T r once; sparse `accumulate` adds both to caller-owned buffers."""
        batch_size = int(residuals.shape[0])
        if not self._sparse_jac:
            wp.launch(
                kernel=aggregate_residuals_to_costs_kernel,
                dim=batch_size,
                inputs=[residuals, costs],
                device=self.device,
            )
            if gradient is not None:
                wp.launch_tiled(
                    self._dense_jtr_kernel,
                    dim=[batch_size],
                    inputs=[self.jacobians, residuals],
                    outputs=[gradient],
                    block_dim=TILE_THREADS,
                    device=self.device,
                )
            return
        if not accumulate:
            _launch_strided_cost(residuals, costs)
            if gradient is None:
                return
            gradient.zero_()
        lanes = 32 if self.device.is_cuda else 1
        splits = -(-self._sparse_num_splits // lanes) * lanes
        wp.launch(
            kernel=_sparse_jtr_split_kernel,
            dim=(batch_size, splits),
            inputs=[
                residuals,
                self._sparse_jac_values,
                self._sparse_value_perm,
                self._sparse_row_perm,
                self._sparse_split_cols,
                self._sparse_split_offsets,
                -(-self.total_residual_dim // splits) if accumulate else 0,
                lanes,
            ],
            outputs=[costs, gradient],
            device=self.device,
        )

    def _mask_gradient(self, gradient: wp.array) -> None:
        # Zero locked tangent dims (reduced-DOF subset + lock_frames) so no term moves them.
        if self._active_dof_mask is None:
            return
        wp.launch(
            kernel=apply_dof_mask_kernel,
            dim=(int(gradient.shape[0]), self.total_tangent_dim),
            inputs=[gradient, self._active_dof_mask],
            device=self.device,
        )

    def _compute_cost_and_gradient_jacobian(self, var: VarValues) -> None:
        """analytic_jacobian mode: fill J (sparse or dense) -> reduce cost + J^T r -> mask."""
        self._precompute_terms(var, need_gradient=True)
        self._fill_jacobian(var, self.residuals, need_jacobian=True)
        self._reduce_cost_gradient(self.residuals, self.costs, self.gradient)
        self._mask_gradient(self.gradient)
