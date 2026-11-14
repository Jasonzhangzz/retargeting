# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportGeneralTypeIssues=false
import zlib
from dataclasses import dataclass, replace
from typing import Dict, Literal, Optional, Sequence

import numpy as np
import warp as wp

from robokit.opt.optimizer import (
    Optimizer,
    OptimizerConfig,
    aggregate_residuals_to_costs_kernel,
)
from robokit.opt.var_values import VarValues
from robokit.terms.task import GradientTask, Task
from robokit.utils.warp_utils import row_sum32, row_sync, warp_shuffle, wp_device_type


@dataclass(frozen=True)
class LBFGSOptimizerConfig(OptimizerConfig):
    early_stopping_interval: int = 0
    """Off by default: without host checks, a folded line search defers each step into the next direction launch."""
    history_len: int = 10
    """Curvature pairs kept for the two-loop recursion."""
    h0_scale: float = 1.0
    """Initial inverse-Hessian scale when there is no history or `normalize_direction` is on (else Barzilai-Borwein)."""
    line_search_alphas: tuple = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1.0)
    """Trial step sizes, evaluated in parallel."""
    wolfe_c1: float = 1e-4
    """Sufficient-decrease constant."""
    wolfe_c2: float = 0.9
    """Curvature constant (approx_wolfe)."""
    line_search: Literal["armijo", "approx_wolfe"] = "armijo"
    """armijo: largest passing trial, else no step; approx_wolfe: largest Armijo/curvature trial, else smallest, and needs folded gradients."""
    normalize_direction: bool = False
    """Normalize directions so trial steps are lengths; H0 then uses `h0_scale` instead of Barzilai-Borwein."""
    gradient_mode: Literal["analytic_jacobian", "autodiff", "analytic_gradient"] = "analytic_jacobian"
    """analytic_jacobian: terms fill a shared sparse/dense J, then one Jᵀr reduction; autodiff: tape gradients;
    analytic_gradient: term kernels accumulate cost and gradient directly."""
    fold_line_search_gradient: bool = False
    """Fold trial gradients to skip a recompute; needs analytic_gradient or all-sparse analytic_jacobian terms."""

    def build(
        self, terms: Sequence[Task], device: wp_device_type, active_dof_mask: Optional[wp.array] = None
    ) -> "LBFGSOptimizer":
        """Runs without a per-iteration graph: each step branches on the host iteration count."""
        return LBFGSOptimizer(terms, replace(self, cuda_graph_mode="none"), device, active_dof_mask)


_KERNEL_CACHE: Dict[tuple, object] = {}


class LBFGSOptimizer(Optimizer):
    @classmethod
    def _build_kernel(
        cls,
        tangent_dim: int,
        history_len: int,
        num_alphas: int,
        threads: int,
        fold_gradient: bool,
        approx_wolfe: bool,
        zero_trials: bool,
        wolfe_c1: float,
        wolfe_c2: float,
        h0_scale: float,
        normalize_direction: bool = False,
    ):
        """Build a vector-free L-BFGS kernel per tangent size, history length, and row threads (Chen et al. 2014).

        Execution layout:
        - CUDA: each thread keeps every threads-th element and its history in registers.
        - CPU: one thread holds the whole row.

        Dot-product sources:
        - Gradient/new-pair products: batched row sums.
        - Older-pair products: `sy_history` stores s_i . y_j for i older than j, avoiding reductions during the loops.
        """
        key = (
            tangent_dim,
            history_len,
            num_alphas,
            threads,
            fold_gradient,
            approx_wolfe,
            zero_trials,
            wolfe_c1,
            wolfe_c2,
            h0_scale,
            normalize_direction,
        )
        if key in _KERNEL_CACHE:
            return _KERNEL_CACHE[key]
        dims, pairs, threads = wp.constant(tangent_dim), wp.constant(history_len), wp.constant(threads)
        # fixed once the solver is built: the kernel and its stages read them as constants
        fold_gradient, approx_wolfe, zero_trials = (
            wp.constant(int(fold_gradient)),
            wp.constant(int(approx_wolfe)),
            wp.constant(int(zero_trials)),
        )
        wolfe_c1, wolfe_c2 = wp.constant(float(wolfe_c1)), wp.constant(float(wolfe_c2))
        h0_scale = wp.constant(float(h0_scale))
        normalize = wp.constant(int(normalize_direction))
        chunks = wp.constant((dims + threads - 1) // threads)
        lane_vector = wp.types.vector(length=chunks, dtype=wp.float32)
        history_vector = wp.types.vector(length=pairs, dtype=wp.float32)
        history_rows = wp.types.matrix(shape=(pairs, chunks), dtype=wp.float32)
        sums32 = wp.types.vector(length=32, dtype=wp.float32)
        trials = wp.constant(num_alphas)
        line_search_vector = wp.types.vector(length=trials, dtype=wp.float32)

        @wp.func
        def _select_best_step(
            batch_idx: int,
            thread: int,
            step_sizes: line_search_vector,
            mask: wp.array1d(dtype=wp.float32),
            expanded_costs: wp.array1d(dtype=wp.float32),
            expanded_gradient: wp.array2d(dtype=wp.float32),
            candidate_velocities: wp.array2d(dtype=wp.float32),
            slope_out: wp.array1d(dtype=wp.float32),
            last_step: wp.array2d(dtype=wp.float32),
            gradient: wp.array2d(dtype=wp.float32),
            gradient_prev: wp.array2d(dtype=wp.float32),
            costs: wp.array1d(dtype=wp.float32),
            winner_idx: wp.array1d(dtype=wp.int32),
            best_costs: wp.array1d(dtype=wp.float32),
            improved: wp.array1d(dtype=wp.int32),
        ):
            """All row threads select the pending trial, move its gradient, and compare its score with the best."""
            base = batch_idx * trials
            cost_curr = costs[batch_idx]
            trial_costs = line_search_vector()
            for trial_idx in range(trials):
                trial_costs[trial_idx] = expanded_costs[base + trial_idx]
            slopes = sums32()
            # loads outside branches, from indices kept in range, so each issues without waiting on the last
            if approx_wolfe != 0:  # gradient . direction at each trial
                for trial_idx in range(trials):
                    for chunk_idx in range(chunks):
                        dim_idx = wp.min(thread + threads * chunk_idx, dims - 1)
                        e = base + trial_idx
                        dot = expanded_gradient[e, dim_idx] * candidate_velocities[e, dim_idx]
                        slopes[trial_idx] += wp.where(thread + threads * chunk_idx < dims, dot, 0.0)
                slopes[0] = row_sum32(slopes)
            # Select the largest passing trial: approx Wolfe falls back to the smallest; Armijo falls back to no step.
            accept = wp.where(approx_wolfe != 0, 0, -1)
            accepted_cost = wp.where(approx_wolfe != 0, trial_costs[0], cost_curr)
            slope_0 = wp.where(approx_wolfe != 0, warp_shuffle(slopes[0], 0) / step_sizes[0], slope_out[batch_idx])
            reference = accepted_cost
            for trial_idx in range(trials):
                slope = (
                    wp.where(threads > 1, warp_shuffle(slopes[0], trial_idx), slopes[trial_idx]) / step_sizes[trial_idx]
                )
                sufficient = trial_costs[trial_idx] <= reference + wolfe_c1 * step_sizes[trial_idx] * slope_0
                curvature = wp.where(approx_wolfe != 0, slope >= wolfe_c2 * slope_0, slope_0 < 0.0)
                if sufficient and curvature:
                    accept = trial_idx
                    accepted_cost = trial_costs[trial_idx]
            winner = wp.where(accept >= 0, base + accept, -1)
            new_cost = accepted_cost
            row_sync()  # every thread read the current cost before it is replaced
            for chunk_idx in range(chunks):
                dim_idx = wp.min(thread + threads * chunk_idx, dims - 1)
                step_d = wp.where(winner >= 0, candidate_velocities[wp.max(winner, 0), dim_idx], 0.0)
                gradient_d = gradient[batch_idx, dim_idx]
                new_gradient = gradient_d
                if fold_gradient != 0:
                    new_gradient = wp.where(
                        winner >= 0, expanded_gradient[wp.max(winner, 0), dim_idx] * mask[dim_idx], gradient_d
                    )
                if thread + threads * chunk_idx < dims:
                    last_step[batch_idx, dim_idx] = step_d
                    if fold_gradient != 0:
                        gradient_prev[batch_idx, dim_idx] = gradient_d
                        gradient[batch_idx, dim_idx] = new_gradient
                    if zero_trials != 0:
                        for trial_idx in range(trials):
                            expanded_gradient[base + trial_idx, dim_idx] = 0.0
            if thread == 0:
                winner_idx[batch_idx] = winner
                costs[batch_idx] = new_cost
                improved[batch_idx] = wp.where(new_cost < best_costs[batch_idx], 1, 0)  # the host then keeps it
                best_costs[batch_idx] = wp.min(new_cost, best_costs[batch_idx])
                if zero_trials != 0:
                    for trial_idx in range(trials):
                        expanded_costs[base + trial_idx] = 0.0

        @wp.func
        def _load_history(
            batch_idx: int,
            thread: int,
            count: int,
            start: int,
            gradient: wp.array2d(dtype=wp.float32),
            gradient_prev: wp.array2d(dtype=wp.float32),
            last_step: wp.array2d(dtype=wp.float32),
            s_history: wp.array3d(dtype=wp.float32),
            y_history: wp.array3d(dtype=wp.float32),
        ):
            """Load each thread's every Lc-th history element, newest first, before reductions.

            Loaded state:
            - History rows and gradient.
            - Step reported by `integrate` (possibly shortened by variable limits) and its change.
            """
            s_rows = history_rows()
            y_rows = history_rows()
            g = lane_vector()
            step = lane_vector()
            change = lane_vector()
            for chunk_idx in range(chunks):
                dim_idx = wp.min(thread + threads * chunk_idx, dims - 1)
                valid = thread + threads * chunk_idx < dims
                for i in range(pairs):
                    slot = (start + count - 1 - i + pairs) % pairs
                    s_rows[i, chunk_idx] = wp.where(valid and i < count, s_history[batch_idx, slot, dim_idx], 0.0)
                    y_rows[i, chunk_idx] = wp.where(valid and i < count, y_history[batch_idx, slot, dim_idx], 0.0)
                g[chunk_idx] = wp.where(valid, gradient[batch_idx, dim_idx], 0.0)
                change[chunk_idx] = wp.where(
                    valid, gradient[batch_idx, dim_idx] - gradient_prev[batch_idx, dim_idx], 0.0
                )
                step[chunk_idx] = wp.where(valid, last_step[batch_idx, dim_idx], 0.0)
            return s_rows, y_rows, g, step, change

        @wp.func
        def _update_history(
            batch_idx: int,
            thread: int,
            count: int,
            start: int,
            y_dot_s: float,
            own_old_sy: float,
            step: lane_vector,
            change: lane_vector,
            s_rows: history_rows,
            y_rows: history_rows,
            s_history: wp.array3d(dtype=wp.float32),
            y_history: wp.array3d(dtype=wp.float32),
            sy_history: wp.array3d(dtype=wp.float32),
            rho_history: wp.array2d(dtype=wp.float32),
            history_count: wp.array1d(dtype=wp.int32),
            history_start: wp.array1d(dtype=wp.int32),
        ):
            """Store the previous step's pair as the newest, with its s_i . y column, and shift this thread's rows."""
            write_idx = (start + count) % pairs
            for chunk_idx in range(chunks):
                if thread + threads * chunk_idx < dims:
                    s_history[batch_idx, write_idx, thread + threads * chunk_idx] = step[chunk_idx]
                    y_history[batch_idx, write_idx, thread + threads * chunk_idx] = change[chunk_idx]
            if thread < count and thread < pairs - 1:  # the older pairs kept: s_i . y in the new pair's column
                sy_history[batch_idx, (start + count - 1 - thread) % pairs, write_idx] = own_old_sy
            if thread == 0:
                rho_history[batch_idx, write_idx] = 1.0 / y_dot_s
            for j in range(pairs - 1):
                i = pairs - 1 - j
                s_rows[i] = s_rows[i - 1]
                y_rows[i] = y_rows[i - 1]
            s_rows[0] = step
            y_rows[0] = change
            if count < pairs:
                count += 1
            else:
                start = (start + 1) % pairs
            if thread == 0:
                history_count[batch_idx] = count
                history_start[batch_idx] = start
            return count, start, s_rows, y_rows

        @wp.func
        def _compute_search_direction(
            batch_idx: int,
            pair_idx: int,
            count: int,
            start: int,
            new: bool,
            y_dot_s: float,
            s_g: float,
            own_new_sy: float,
            new_sy: history_vector,
            g: lane_vector,
            s_rows: history_rows,
            y_rows: history_rows,
            sy_history: wp.array3d(dtype=wp.float32),
            rho_history: wp.array2d(dtype=wp.float32),
        ):
            """Run the vector-free L-BFGS loops, then compute direction and slope.

            Thread ownership:
            - CUDA thread h: pair h's rho, s_h . y_j for newer j, and s_j . y_h for older j.
            - CPU thread: all pairs, following the textbook loops.
            """
            q = g
            if threads == 1:
                rho = history_vector()
                alpha = history_vector()
                for i in range(pairs):
                    rho[i] = wp.where(i < count, 1.0 / wp.dot(s_rows[i], y_rows[i]), 0.0)
                    alpha[i] = rho[i] * wp.dot(s_rows[i], q)
                    q = q - alpha[i] * y_rows[i]
                y_y = wp.dot(y_rows[0], y_rows[0])
                bb = normalize == 0 and count > 0 and y_y > 1.0e-12
                q = wp.where(bb, wp.dot(s_rows[0], y_rows[0]) / y_y, h0_scale) * q
                for j in range(pairs):
                    i = pairs - 1 - j
                    q = q + (alpha[i] - rho[i] * wp.dot(y_rows[i], q)) * s_rows[i]
            else:
                own = pair_idx < count
                slot_h = (start + count - 1 - wp.min(pair_idx, pairs - 1) + pairs) % pairs
                rho_h = wp.where(
                    own, wp.where(new and pair_idx == 0, 1.0 / y_dot_s, rho_history[batch_idx, slot_h]), 0.0
                )
                s_g = wp.where(own, s_g, 0.0)
                row = history_vector()
                column = history_vector()
                for j in range(pairs):
                    slot_j = (start + count - 1 - j + pairs) % pairs
                    row_j = wp.where(new and j == 0, own_new_sy, sy_history[batch_idx, slot_h, slot_j])
                    row[j] = wp.where(own and j < pair_idx, row_j, 0.0)
                    column_j = wp.where(new and pair_idx == 0, new_sy[j], sy_history[batch_idx, slot_j, slot_h])
                    column[j] = wp.where(own and j > pair_idx and j < count, column_j, 0.0)

                # First L-BFGS loop runs newest to oldest: alpha_i = rho_i s_i · (g - newer alpha_j y_j).
                alpha = history_vector()
                for i in range(pairs):
                    alpha[i] = warp_shuffle(rho_h * s_g, i)
                    s_g -= alpha[i] * row[i]
                for i in range(pairs):
                    q = q - alpha[i] * y_rows[i]

                # Use y_i · q and newest y · y / s · y for Barzilai-Borwein H0; normalized trial steps remain lengths.
                dots = sums32()
                for i in range(pairs):
                    dots[i] = wp.dot(y_rows[i], q)
                dots[pairs] = wp.dot(y_rows[0], y_rows[0])
                dots[pairs + 1] = wp.dot(s_rows[0], y_rows[0])
                total = row_sum32(dots)
                y_y = warp_shuffle(total, pairs)
                bb = normalize == 0 and count > 0 and y_y > 1.0e-12
                gamma = wp.where(bb, warp_shuffle(total, pairs + 1) / y_y, h0_scale)

                # second loop, oldest to newest: beta_i = rho_i y_i . (gamma q + sum over older j of (alpha_j - beta_j) s_j)
                coefficient = history_vector()  # alpha_i - beta_i
                s_r = float(0.0)
                for j in range(pairs):
                    i = pairs - 1 - j
                    coefficient[i] = alpha[i] - warp_shuffle(rho_h * (gamma * total + s_r), i)
                    s_r += coefficient[i] * column[i]
                q = gamma * q
                for i in range(pairs):
                    q = q + coefficient[i] * s_rows[i]

            q = -q
            dots = sums32()
            dots[0] = wp.dot(g, q)
            dots[1] = wp.dot(q, q)
            total = row_sum32(dots)
            if normalize != 0:
                norm = wp.sqrt(wp.where(threads > 1, warp_shuffle(total, 1), dots[1]))
                if norm > 1.0e-12:
                    q = q / norm
                    total = total / norm  # thread 0's g . q, of the unit direction
            return q, total

        @wp.func
        def _write_candidates(
            batch_idx: int,
            thread: int,
            q: lane_vector,
            slope: float,
            step_sizes: line_search_vector,
            mask: wp.array1d(dtype=wp.float32),
            search_direction: wp.array2d(dtype=wp.float32),
            candidate_velocities: wp.array2d(dtype=wp.float32),
            slope_out: wp.array1d(dtype=wp.float32),
        ):
            """The direction and each line-search trial's step."""
            for chunk_idx in range(chunks):
                dim_idx = wp.min(thread + threads * chunk_idx, dims - 1)
                masked = q[chunk_idx] * mask[dim_idx]
                if thread + threads * chunk_idx < dims:
                    search_direction[batch_idx, dim_idx] = q[chunk_idx]
                    for trial_idx in range(trials):
                        candidate_velocities[batch_idx * trials + trial_idx, dim_idx] = step_sizes[trial_idx] * masked
            if thread == 0:
                slope_out[batch_idx] = slope  # g . direction, for the Armijo line search

        def _lbfgs_iteration(
            last_step: wp.array2d(dtype=wp.float32),  # (B, Dc)
            gradient: wp.array2d(dtype=wp.float32),  # (B, Dc)
            gradient_prev: wp.array2d(dtype=wp.float32),  # (B, Dc)
            sy_history: wp.array3d(dtype=wp.float32),  # (B, Mc, Mc) s . y of the slots' pairs, row older than column
            s_history: wp.array3d(dtype=wp.float32),  # (B, Mc, Dc)
            y_history: wp.array3d(dtype=wp.float32),  # (B, Mc, Dc)
            rho_history: wp.array2d(dtype=wp.float32),  # (B, Mc)
            history_count: wp.array1d(dtype=wp.int32),  # (B,)
            history_start: wp.array1d(dtype=wp.int32),  # (B,)
            alphas: wp.array1d(dtype=wp.float32),  # (S,) line-search step sizes
            mask: wp.array1d(dtype=wp.float32),  # (Dc,) locked tangent dims don't move
            select_first: int,  # first take the pending line search's step
            direction: int,  # 0: the step alone
            expanded_costs: wp.array1d(dtype=wp.float32),  # (B * S,)
            expanded_gradient: wp.array2d(dtype=wp.float32),  # (B * S, Dc), approx_wolfe or folding only
            search_direction: wp.array2d(dtype=wp.float32),  # (B, Dc)
            slope_out: wp.array1d(dtype=wp.float32),  # (B,) the step's slope in (Armijo), the new one out
            candidate_velocities: wp.array2d(dtype=wp.float32),  # (B * S, Dc) each line-search trial's step
            costs: wp.array1d(dtype=wp.float32),  # (B,) the current cost in; the winner's out when folding
            winner_idx: wp.array1d(dtype=wp.int32),  # (B,) expanded-batch index of the winner, -1: no step
            best_costs: wp.array1d(dtype=wp.float32),  # (B,) the lowest cost seen
            improved: wp.array1d(dtype=wp.int32),  # (B,) 1 where this iterate is the new lowest
        ):
            tid = wp.tid()
            batch_idx, thread = tid // threads, tid % threads
            step_sizes = line_search_vector()
            for trial_idx in range(trials):
                step_sizes[trial_idx] = alphas[trial_idx]
            if select_first != 0:
                _select_best_step(
                    batch_idx,
                    thread,
                    step_sizes,
                    mask,
                    expanded_costs,
                    expanded_gradient,
                    candidate_velocities,
                    slope_out,
                    last_step,
                    gradient,
                    gradient_prev,
                    costs,
                    winner_idx,
                    best_costs,
                    improved,
                )
            if direction == 0:
                return

            count = history_count[batch_idx]
            start = history_start[batch_idx]
            s_rows, y_rows, g, step, change = _load_history(
                batch_idx,
                thread,
                count,
                start,
                gradient,
                gradient_prev,
                last_step,
                s_history,
                y_history,
            )

            # the previous step's pair (s, y): s . y, s . g, and each older s_i . y; each older s_i . g, in one row sum
            dots = sums32()
            dots[0] = wp.dot(step, change)
            dots[1] = wp.dot(step, g)
            for i in range(pairs):
                dots[2 + i] = wp.dot(s_rows[i], g)
                dots[2 + pairs + i] = wp.dot(s_rows[i], change)
            total = row_sum32(dots)
            y_dot_s = warp_shuffle(total, 0)
            new = y_dot_s > 1.0e-8 and wp.isfinite(y_dot_s)  # the pair becomes the newest
            pair_idx = thread % 32  # the history pair this thread carries through the scalar two-loop, newest first
            s_g = warp_shuffle(total, wp.where(new, wp.where(pair_idx == 0, 1, 1 + pair_idx), 2 + pair_idx))
            new_sy = history_vector()  # s_i . y of the new pair, the older ones shifted in behind it
            for i in range(pairs):
                new_sy[i] = warp_shuffle(total, 1 + pairs + i)
            own_new_sy = warp_shuffle(total, 1 + pairs + pair_idx)
            if new:
                count, start, s_rows, y_rows = _update_history(
                    batch_idx,
                    thread,
                    count,
                    start,
                    y_dot_s,
                    warp_shuffle(total, 2 + pairs + pair_idx),
                    step,
                    change,
                    s_rows,
                    y_rows,
                    s_history,
                    y_history,
                    sy_history,
                    rho_history,
                    history_count,
                    history_start,
                )

            q, slope = _compute_search_direction(
                batch_idx,
                pair_idx,
                count,
                start,
                new,
                y_dot_s,
                s_g,
                own_new_sy,
                new_sy,
                g,
                s_rows,
                y_rows,
                sy_history,
                rho_history,
            )
            _write_candidates(
                batch_idx,
                thread,
                q,
                slope,
                step_sizes,
                mask,
                search_direction,
                candidate_velocities,
                slope_out,
            )

        # Warp keys the kernel by qualname: name it by config, not build order, so every process hits its kernel cache
        _lbfgs_iteration.__qualname__ = f"_lbfgs_iteration_{zlib.crc32(repr(key).encode()):x}"
        _KERNEL_CACHE[key] = wp.kernel(enable_backward=False, module="unique")(_lbfgs_iteration)
        return _KERNEL_CACHE[key]

    def _tiled_search_direction(self, var: VarValues, select_first: bool = False, direction: bool = True):
        """Build curvature history, direction, and trials in one launch (`self._threads` threads/row; one on CPU).

        Stages:
        1. Optionally take the pending line-search step (`select_first`).
        2. Record the previous curvature pair when s·y > 0.
        3. Compute the direction, stopping here when `direction` is set.
        4. Build line-search trials.
        """
        wp.launch(
            self._search_direction_kernel,
            dim=self.batch_size * self._threads,
            inputs=[
                self.last_step_dq,
                self.gradient,
                self.gradient_prev,
                self.sy_history,
                self.s_history,
                self.y_history,
                self.rho_history,
                self.history_count,
                self.history_start,
                self.line_search_alphas,
                self._mask,
                int(select_first),
                int(direction),
                self._expanded_costs,
                self._expanded_gradient,
            ],
            outputs=[
                self.search_direction,
                self.initial_slope,
                self._candidate_velocities,
                self.costs,
                self._winner_expanded_idx,
                self.best_costs,
                self._best_improved,
            ],
            block_dim=self._threads,
            device=self.device,
        )

    def _tiled_select_best_step(self, var: VarValues):
        """Take the winning line-search trial: the direction kernel stopping at the step."""
        self._tiled_search_direction(var, select_first=True, direction=False)

    _config_type = LBFGSOptimizerConfig

    def _build(self, var: VarValues):
        config = self.config
        terms = self.terms
        D, M = var.tangent_dim, config.history_len
        approx_wolfe = config.line_search == "approx_wolfe"
        self._threads = min(-(-D // 32) * 32, 256) if self.device.is_cuda else 1
        if self.device.is_cuda and M > 15:
            raise ValueError("history_len takes at most 15 pairs on CUDA")
        device = self.device
        jac_mode = config.gradient_mode == "analytic_jacobian"
        all_sparse = self.is_all_sparse
        self._build_terms(var, residuals_require_grad=(config.gradient_mode == "autodiff"))

        batch_size = self.max_batch_size
        total_tangent_dim = self.total_tangent_dim
        history_len = config.history_len
        n_line_search = len(config.line_search_alphas)
        self.n_line_search = n_line_search
        self.gradient = wp.zeros((batch_size, total_tangent_dim), dtype=wp.float32, device=device)
        self.gradient_prev = wp.zeros((batch_size, total_tangent_dim), dtype=wp.float32, device=device)
        self.search_direction = wp.zeros((batch_size, total_tangent_dim), dtype=wp.float32, device=device)
        # locked tangent dims (reduced-DOF subset + lock_frames) never move; all ones without a mask
        self._mask = (
            self._active_dof_mask if self._active_dof_mask is not None else wp.ones(total_tangent_dim, device=device)
        )
        self.last_step_dq = wp.zeros((batch_size, total_tangent_dim), dtype=wp.float32, device=device)
        self._best_var = var.clone()  # the lowest-cost iterate, which the solve returns
        self.best_costs = wp.zeros((batch_size,), dtype=wp.float32, device=device)
        self._best_improved = wp.zeros((batch_size,), dtype=wp.int32, device=device)
        self.initial_slope = wp.zeros((batch_size,), dtype=wp.float32, device=device)

        # L-BFGS history
        self.s_history = wp.zeros((batch_size, history_len, total_tangent_dim), dtype=wp.float32, device=device)
        self.y_history = wp.zeros((batch_size, history_len, total_tangent_dim), dtype=wp.float32, device=device)
        self.rho_history = wp.zeros((batch_size, history_len), dtype=wp.float32, device=device)
        self.sy_history = wp.zeros((batch_size, history_len, history_len), dtype=wp.float32, device=device)
        self.history_count = wp.zeros((batch_size,), dtype=wp.int32, device=device)
        self.history_start = wp.zeros((batch_size,), dtype=wp.int32, device=device)

        # line search buffers (parallel expanded-batch evaluation)
        self.line_search_alphas = wp.from_numpy(
            np.array(config.line_search_alphas, dtype=np.float32), dtype=wp.float32, device=device
        )
        expanded_batch = batch_size * n_line_search
        # fold needs an expanded-batch gradient: analytic_gradient or sparse analytic_jacobian
        self._fold = config.fold_line_search_gradient and (
            config.gradient_mode == "analytic_gradient" or (jac_mode and all_sparse)
        )
        if approx_wolfe and not self._fold:
            raise ValueError("approx_wolfe line search needs the folded line-search gradient.")
        # Clear each sparse trial after reading so the next search accumulates its cost and gradient once.
        self._zero_trials = device.is_cuda and self._fold and jac_mode and all_sparse
        self._expanded_indices = wp.from_numpy(
            np.repeat(np.arange(batch_size), n_line_search).astype(np.int32), dtype=wp.int32, device=device
        )
        self._expanded_var = var.gather(self._expanded_indices)
        self._expanded_proposed_var = self._expanded_var.clone()
        self._search_direction_kernel = self._build_kernel(
            D,
            M,
            len(config.line_search_alphas),
            self._threads,
            self._fold,
            approx_wolfe,
            self._zero_trials,
            config.wolfe_c1,
            config.wolfe_c2,
            config.h0_scale,
            config.normalize_direction,
        )
        self._expanded_residuals = wp.zeros((expanded_batch, self.total_residual_dim), dtype=wp.float32, device=device)
        self._expanded_costs = wp.zeros((expanded_batch,), dtype=wp.float32, device=device)
        self._candidate_velocities = wp.zeros((expanded_batch, total_tangent_dim), dtype=wp.float32, device=device)
        # Winner index is always stored; allocate expanded gradients only for folded line search.
        self._winner_expanded_idx = wp.zeros((batch_size,), dtype=wp.int32, device=device)
        self._expanded_gradient = (
            wp.zeros((expanded_batch, total_tangent_dim), dtype=wp.float32, device=device) if self._fold else None
        )
        if config.gradient_mode == "autodiff":
            self._autodiff_tape = wp.Tape()
            self._autodiff_velocity = wp.zeros(
                (batch_size, total_tangent_dim), dtype=wp.float32, device=device, requires_grad=True
            )
            self._autodiff_proposed_var = var.integrate(self._autodiff_velocity, tangent_mask=self._active_dof_mask)
        elif config.gradient_mode == "analytic_jacobian":
            self._build_jacobian(var, expanded_batch if self._fold else batch_size)
        else:  # analytic_gradient; the closures capture buffers, not `self` (see _compute_cost_and_gradient)
            costs, gradient = self.costs, self.gradient
            expanded_costs, expanded_gradient = self._expanded_costs, self._expanded_gradient
            self._direct_cost_grad_fns = []
            self._direct_cost_only_fns = []
            self._direct_expanded_cost_grad_fns = []
            for t in terms:
                assert isinstance(t, GradientTask)
                fn = t.compute_weighted_cost_and_gradient
                self._direct_cost_grad_fns.append(
                    lambda offset, var, _fn=fn: _fn(var, out_cost=costs, out_gradient=gradient)
                )
                # out_gradient omitted → cost only
                self._direct_cost_only_fns.append(lambda offset, var, _fn=fn: _fn(var, out_cost=expanded_costs))
                self._direct_expanded_cost_grad_fns.append(
                    lambda offset, var, _fn=fn: _fn(var, out_cost=expanded_costs, out_gradient=expanded_gradient)
                )

    def _compute_cost_and_gradient(self, var: VarValues):
        """Dispatch on `gradient_mode` without method reference cycles that defer scene freeing until CUDA capture."""
        if self.config.gradient_mode == "autodiff":
            self._compute_cost_and_gradient_autodiff(var)
        elif self.config.gradient_mode == "analytic_jacobian":
            self._compute_cost_and_gradient_jacobian(var)
        else:
            self._compute_cost_and_gradient_direct(var)

    def _evaluate_line_search_costs(self, expanded_batch: int):
        """Costs (and, when folding, gradient) of the expanded line-search batch."""
        if self.config.gradient_mode == "analytic_gradient":
            self._evaluate_line_search_costs_direct(expanded_batch)
            return
        self._precompute_terms(self._expanded_proposed_var, need_gradient=self._fold)
        self._fill_jacobian(self._expanded_proposed_var, self._expanded_residuals, need_jacobian=self._fold)
        self._reduce_cost_gradient(
            self._expanded_residuals,
            self._expanded_costs,
            self._expanded_gradient if self._fold else None,
            accumulate=self._zero_trials,
        )

    def _evaluate_line_search_costs_direct(self, expanded_batch: int):
        self._precompute_terms(self._expanded_proposed_var, need_gradient=self._fold)
        if not self._fold:
            self._expanded_costs.zero_()
            self._parallel_for_objectives(self._direct_cost_only_fns, self._expanded_proposed_var)
            return
        # fold: also compute the gradient here so the winner's can be gathered (skips the recompute pass)
        self._expanded_costs.zero_()
        self._expanded_gradient.zero_()
        self._parallel_for_objectives(self._direct_expanded_cost_grad_fns, self._expanded_proposed_var)

    def _compute_cost_and_gradient_direct(self, var: VarValues):
        """analytic_gradient mode: each term's kernel atomic-adds cost+gradient into shared buffers."""
        self._precompute_terms(var, need_gradient=True)
        self.gradient.zero_()
        self.costs.zero_()
        self._parallel_for_objectives(self._direct_cost_grad_fns, var)
        # Mask the gradient so two-loop curvature (s, y, rho) stays in the active subspace - reduced-DOF L-BFGS.
        self._mask_gradient(self.gradient)

    def _compute_cost_and_gradient_autodiff(self, var: VarValues):
        """Compute residuals, costs, and gradient via autodiff (no Jacobian materialization)."""
        self._autodiff_tape.reset()
        self._autodiff_tape.gradients = {}
        self._autodiff_velocity.zero_()

        with self._autodiff_tape:
            proposed = var.integrate(
                self._autodiff_velocity, out=self._autodiff_proposed_var, tangent_mask=self._active_dof_mask
            )
            self._precompute_terms(proposed, need_gradient=False)
            for offset, fn in zip(self.residual_offsets, self._residual_fns):
                fn(offset, proposed, self.residuals)

        wp.launch(
            kernel=aggregate_residuals_to_costs_kernel,
            dim=self.batch_size,
            inputs=[self.residuals, self.costs],
            device=self.device,
        )
        self._autodiff_tape.backward(grads={self.residuals: self.residuals})
        wp.copy(self.gradient, self._autodiff_velocity.grad)
        self._autodiff_tape.zero()

    def _start(self, var: VarValues) -> None:
        # In folded mode, update position in the next direction launch; it first needs the winner's gradient.
        self._defer_step = (
            self._fold
            and self.config.cuda_graph_mode != "iter"
            and self.config.early_stopping_interval == 0
            and self._iteration_callback is None
        )
        var.invalidate()
        self._compute_cost_and_gradient(var)
        self.history_count.zero_()
        self.history_start.zero_()
        self.rho_history.zero_()  # empty slots stay no-ops in the fixed-length two-loop
        self.last_step_dq.zero_()  # so the first history update sees s = 0 and records nothing
        self.gradient_prev.zero_()
        self.best_costs.fill_(1.0e30)
        self._best_improved.fill_(1)  # before any line search, the starting point is the best there is
        self._best_var.invalidate()  # it keeps the joint positions alone, so `accept` copies nothing else

    def _step_body(self, var: VarValues) -> None:
        take_step = self._defer_step and self._iter > 0
        # the previous step's curvature pair, then the direction + slope (steepest descent when history is empty)
        self._tiled_search_direction(var, select_first=take_step)
        if take_step:
            var.integrate(self.last_step_dq, out=var, tangent_mask=self._active_dof_mask)
        if not self._fold:  # save gradient for next iteration's history update (folding saves it with the winner's)
            wp.copy(self.gradient_prev, self.gradient)

        expanded_batch = self.batch_size * self.n_line_search
        # Each trial reports its actual move after limits; the winner's move becomes the next curvature pair.
        var.gather(self._expanded_indices[:expanded_batch], out=self._expanded_var)
        self._expanded_var.integrate(
            self._candidate_velocities,
            out=self._expanded_proposed_var,
            tangent_mask=self._active_dof_mask,
            out_velocity=self._candidate_velocities,
        )
        self._evaluate_line_search_costs(expanded_batch)
        if not self._defer_step or self._iter == self._num_steps - 1:
            self._tiled_select_best_step(var)
            var.integrate(self.last_step_dq, out=var, tangent_mask=self._active_dof_mask)
        if not self._fold:
            self._compute_cost_and_gradient(var)
        best = self._best_var.view(self.batch_size)  # return the lowest-cost iterate, not the last
        best.accept(self._best_improved, var)
        if self._iter == self._num_steps - 1:
            self._best_improved.fill_(1)
            var.invalidate()
            var.accept(self._best_improved, best)
            wp.copy(self.costs, self.best_costs)

    _BATCH_BUFFERS = (
        "residuals",
        "costs",
        "gradient",
        "gradient_prev",
        "search_direction",
        "last_step_dq",
        "_expanded_var",
        "_best_var",
        "best_costs",
        "_best_improved",
        "initial_slope",
        "s_history",
        "y_history",
        "rho_history",
        "sy_history",
        "history_count",
        "history_start",
        "_winner_expanded_idx",
        "_autodiff_velocity",
        # the line-search batch: `n_line_search` entries per batch element
        "_expanded_proposed_var",
        "_expanded_residuals",
        "_expanded_costs",
        "_candidate_velocities",
        "_expanded_gradient",
    )
