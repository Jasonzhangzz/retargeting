# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportIndexIssue=false
# pyright: reportOperatorIssue=false
# pyright: reportGeneralTypeIssues=false
"""Levenberg-Marquardt solvers.

Linear solves:
- Cholesky: fused tile-local normal equations or accumulated `JᵀJ`/`Jᵀr`.
- Matrix-free conjugate gradients: sparse Jacobians.
"""

from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

import numpy as np
import warp as wp

from robokit.opt.optimizer import (
    Optimizer,
    OptimizerConfig,
    _jv_kernel,
    _sparse_jtr_split_kernel,
    apply_dof_mask_kernel,
)
from robokit.opt.var_values import VarValues
from robokit.terms.task import EagerTask, ResidualTask, Task
from robokit.utils.warp_utils import wp_device_type


TILE_THREADS = 32
# Above _LM_FUSED_MAX_ENTRIES (rows × dofs) the fused solve overflows shared memory; stream J in _LM_SOLVE_CHUNK rows.
_LM_SOLVE_CHUNK = 64
_LM_FUSED_MAX_ENTRIES = 512 * 32
# One tile factors JᵀJ up to this many DOFs; beyond, banded 16×16 tiles are faster and one tile stops fitting.
_LM_DENSE_MAX_DOFS = 47
_CG_EPS = 1e-8
_KERNEL_CACHE: Dict[tuple, Any] = {}


def _cholesky_kernel(D: int, bandwidth: int):
    """Solve `(JᵀJ+λI)δ = -Jᵀr` and compute pred_reduction.

    Cholesky layouts:
    - Up to `_LM_DENSE_MAX_DOFS`: one tile holds `JᵀJ`.
    - Larger systems: 16×16 tiles, skipping blocks outside `bandwidth`.
    """
    dense = D <= _LM_DENSE_MAX_DOFS
    key = ("chol", D) if dense else ("chol", D, bandwidth)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    Dc, Wc = wp.constant(D), wp.constant((bandwidth + 15) // 16)

    def dense_cholesky(
        JtJ: wp.array3d(dtype=wp.float32),
        Jtr: wp.array3d(dtype=wp.float32),
        lm_lambda: wp.array1d(dtype=wp.float32),
        delta_out: wp.array2d(dtype=wp.float32),
        pred_out: wp.array1d(dtype=wp.float32),
    ):
        batch_idx = wp.tid()
        lam = lm_lambda[batch_idx]
        A = wp.tile_load(JtJ[batch_idx], shape=(Dc, Dc))
        diag = wp.tile_zeros(shape=(Dc,), dtype=wp.float32)
        for i in range(Dc):
            diag[i] = lam
        A = wp.tile_diag_add(A, diag)
        g_col = wp.tile_load(Jtr[batch_idx], shape=(Dc, 1))
        g = wp.tile_zeros(shape=(Dc,), dtype=wp.float32)
        for i in range(Dc):
            g[i] = g_col[i, 0]
        delta = wp.tile_cholesky_solve(wp.tile_cholesky(A), wp.tile_map(wp.neg, g))
        wp.tile_store(delta_out[batch_idx], delta)
        ld = wp.tile_zeros(shape=(Dc,), dtype=wp.float32)
        for i in range(Dc):
            ld[i] = lam * delta[i]
        pred_out[batch_idx] = wp.float32(0.5) * wp.tile_sum(wp.tile_map(wp.mul, delta, wp.tile_map(wp.sub, ld, g)))[0]

    def banded_cholesky(
        JtJ: wp.array3d(dtype=wp.float32),
        Jtr: wp.array3d(dtype=wp.float32),
        lm_lambda: wp.array1d(dtype=wp.float32),
        delta_out: wp.array2d(dtype=wp.float32),
        pred_out: wp.array1d(dtype=wp.float32),
    ):
        batch_idx = wp.tid()
        lam = lm_lambda[batch_idx]
        # Factor small tiles in place so long trajectories do not exhaust shared memory.
        for j in range((Dc + 15) // 16):
            A = wp.tile_load(JtJ[batch_idx], shape=(16, 16), offset=(16 * j, 16 * j))
            for k in range(wp.max(0, j - Wc), j):
                L = wp.tile_load(JtJ[batch_idx], shape=(16, 16), offset=(16 * j, 16 * k))
                wp.tile_matmul(-L, wp.tile_transpose(L), A)
            for d in range(16):
                A[d, d] = A[d, d] + lam
            diagonal = wp.tile_cholesky(A)
            wp.tile_store(JtJ[batch_idx], diagonal, offset=(16 * j, 16 * j))
            for i in range(j + 1, wp.min((Dc + 15) // 16, j + Wc + 1)):
                Aij = wp.tile_load(JtJ[batch_idx], shape=(16, 16), offset=(16 * i, 16 * j))
                for k in range(wp.max(0, i - Wc), j):
                    Lik = wp.tile_load(JtJ[batch_idx], shape=(16, 16), offset=(16 * i, 16 * k))
                    Ljk = wp.tile_load(JtJ[batch_idx], shape=(16, 16), offset=(16 * j, 16 * k))
                    wp.tile_matmul(-Lik, wp.tile_transpose(Ljk), Aij)
                Lij = wp.tile_transpose(wp.tile_lower_solve(diagonal, wp.tile_transpose(Aij)))
                wp.tile_store(JtJ[batch_idx], Lij, offset=(16 * i, 16 * j))
        for i in range((Dc + 15) // 16):
            forward_rhs = -wp.tile_load(Jtr[batch_idx], shape=(16, 1), offset=(16 * i, 0))
            for j in range(wp.max(0, i - Wc), i):
                forward_L = wp.tile_load(JtJ[batch_idx], shape=(16, 16), offset=(16 * i, 16 * j))
                y = wp.tile_reshape(wp.tile_load(delta_out[batch_idx], shape=(16,), offset=(16 * j,)), shape=(16, 1))
                wp.tile_matmul(-forward_L, y, forward_rhs)
            forward_diagonal = wp.tile_load(JtJ[batch_idx], shape=(16, 16), offset=(16 * i, 16 * i))
            for d in range(16):
                if 16 * i + d >= Dc:
                    forward_diagonal[d, d] = wp.sqrt(lam)
            y = wp.tile_lower_solve(forward_diagonal, forward_rhs)
            wp.tile_store(delta_out[batch_idx], wp.tile_reshape(y, shape=(16,)), offset=(16 * i,))
        pred = wp.float32(0.0)
        for reverse in range((Dc + 15) // 16):
            i = (Dc + 15) // 16 - 1 - reverse
            back_rhs = wp.tile_reshape(wp.tile_load(delta_out[batch_idx], shape=(16,), offset=(16 * i,)), shape=(16, 1))
            for j in range(i + 1, wp.min((Dc + 15) // 16, i + Wc + 1)):
                Lji = wp.tile_load(JtJ[batch_idx], shape=(16, 16), offset=(16 * j, 16 * i))
                x = wp.tile_reshape(wp.tile_load(delta_out[batch_idx], shape=(16,), offset=(16 * j,)), shape=(16, 1))
                wp.tile_matmul(-wp.tile_transpose(Lji), x, back_rhs)
            back_diagonal = wp.tile_load(JtJ[batch_idx], shape=(16, 16), offset=(16 * i, 16 * i))
            for d in range(16):
                if 16 * i + d >= Dc:
                    back_diagonal[d, d] = wp.sqrt(lam)
            x = wp.tile_reshape(wp.tile_upper_solve(wp.tile_transpose(back_diagonal), back_rhs), shape=(16,))
            wp.tile_store(delta_out[batch_idx], x, offset=(16 * i,))
            g = wp.tile_reshape(wp.tile_load(Jtr[batch_idx], shape=(16, 1), offset=(16 * i, 0)), shape=(16,))
            pred += wp.float32(0.5) * wp.tile_sum(wp.tile_map(wp.mul, x, lam * x - g))[0]
        pred_out[batch_idx] = pred

    cholesky = dense_cholesky if dense else banded_cholesky
    cholesky.__name__ = cholesky.__qualname__ = f"_lm_chol_{D}" if dense else f"_lm_chol_{D}_bw{bandwidth}"
    k = wp.kernel(enable_backward=False, module="unique")(cholesky)
    _KERNEL_CACHE[key] = k
    return k


def _solve_kernel(R: int, D: int):
    """Fused fast-path kernel: wide `J`/`r` + λ → δ, pred_reduction, costs, `JᵀJ`/`Jᵀr` tile-local, no streaming."""
    key = ("solve", R, D)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    Rc, Dc = wp.constant(R), wp.constant(D)

    def solve(
        jacobians: wp.array3d(dtype=wp.float32),
        residuals_col: wp.array3d(dtype=wp.float32),
        lm_lambda: wp.array1d(dtype=wp.float32),
        costs_out: wp.array1d(dtype=wp.float32),
        delta_out: wp.array2d(dtype=wp.float32),
        pred_out: wp.array1d(dtype=wp.float32),
    ):
        batch_idx = wp.tid()
        lam = lm_lambda[batch_idx]
        J = wp.tile_load(jacobians[batch_idx], shape=(Rc, Dc))
        r = wp.tile_load(residuals_col[batch_idx], shape=(Rc, 1))
        Jt = wp.tile_transpose(J)
        H = wp.tile_zeros(shape=(Dc, Dc), dtype=wp.float32)
        wp.tile_matmul(Jt, J, H)
        diag = wp.tile_zeros(shape=(Dc,), dtype=wp.float32)
        for i in range(Dc):
            diag[i] = lam
        A = wp.tile_diag_add(H, diag)
        g_col = wp.tile_zeros(shape=(Dc, 1), dtype=wp.float32)
        wp.tile_matmul(Jt, r, g_col)
        g = wp.tile_zeros(shape=(Dc,), dtype=wp.float32)
        for i in range(Dc):
            g[i] = g_col[i, 0]
        delta = wp.tile_cholesky_solve(wp.tile_cholesky(A), wp.tile_map(wp.neg, g))
        wp.tile_store(delta_out[batch_idx], delta)
        ld = wp.tile_zeros(shape=(Dc,), dtype=wp.float32)
        for i in range(Dc):
            ld[i] = lam * delta[i]
        pred_out[batch_idx] = wp.float32(0.5) * wp.tile_sum(wp.tile_map(wp.mul, delta, wp.tile_map(wp.sub, ld, g)))[0]
        rr = wp.tile_zeros(shape=(1, 1), dtype=wp.float32)
        wp.tile_matmul(wp.tile_transpose(r), r, rr)
        costs_out[batch_idx] = wp.float32(0.5) * rr[0, 0]

    solve.__name__ = solve.__qualname__ = f"_lm_solve_{R}x{D}"
    k = wp.kernel(enable_backward=False, module="unique")(solve)
    _KERNEL_CACHE[key] = k
    return k


def _accum_kernel_chunked(D: int, C: int):
    """Accumulate `JᵀJ`/`Jᵀr`/cost one C×D block at a time to bound shared memory; `_cholesky_kernel` solves."""
    key = ("accum_chunked", D, C)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    Dc, Cc = wp.constant(D), wp.constant(C)

    def accumulate(
        jacobians: wp.array4d(dtype=wp.float32),
        residuals_col: wp.array4d(dtype=wp.float32),
        num_chunks: wp.int32,
        JtJ: wp.array3d(dtype=wp.float32),
        Jtr: wp.array3d(dtype=wp.float32),
        costs: wp.array1d(dtype=wp.float32),
    ):
        batch_idx = wp.tid()
        H = wp.tile_zeros(shape=(Dc, Dc), dtype=wp.float32)
        g = wp.tile_zeros(shape=(Dc, 1), dtype=wp.float32)
        rr = wp.tile_zeros(shape=(1, 1), dtype=wp.float32)
        for c in range(num_chunks):
            Jc = wp.tile_load(jacobians[batch_idx, c], shape=(Cc, Dc))
            rc = wp.tile_load(residuals_col[batch_idx, c], shape=(Cc, 1))
            Jct = wp.tile_transpose(Jc)
            wp.tile_matmul(Jct, Jc, H)
            wp.tile_matmul(Jct, rc, g)
            wp.tile_matmul(wp.tile_transpose(rc), rc, rr)
        wp.tile_store(JtJ[batch_idx], H)
        wp.tile_store(Jtr[batch_idx], g)
        costs[batch_idx] = wp.float32(0.5) * rr[0, 0]

    accumulate.__name__ = accumulate.__qualname__ = f"_lm_accum_chunked_c{C}x{D}"
    k = wp.kernel(enable_backward=False, module="unique")(accumulate)
    _KERNEL_CACHE[key] = k
    return k


def _accum_kernel(R: int, D: int):
    """Add JᵀJ/Jᵀr/cost; tile rows and columns when a full normal block exceeds shared-memory budget."""
    fits = (R + D + 2) * D + 2 * R <= _LM_FUSED_MAX_ENTRIES
    key = ("accum", R, D) if fits else ("accum_blocks", min(R, _LM_SOLVE_CHUNK))
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    if fits:
        Rc, Dc = wp.constant(R), wp.constant(D)

        def accumulate(
            jacobians: wp.array3d(dtype=wp.float32),
            residuals_col: wp.array3d(dtype=wp.float32),
            JtJ: wp.array3d(dtype=wp.float32),
            Jtr: wp.array3d(dtype=wp.float32),
            costs: wp.array1d(dtype=wp.float32),
        ):
            batch_idx = wp.tid()
            J = wp.tile_load(jacobians[batch_idx], shape=(Rc, Dc))
            r = wp.tile_load(residuals_col[batch_idx], shape=(Rc, 1))
            Jt = wp.tile_transpose(J)
            H = wp.tile_load(JtJ[batch_idx], shape=(Dc, Dc))
            wp.tile_matmul(Jt, J, H)
            wp.tile_store(JtJ[batch_idx], H)
            g = wp.tile_load(Jtr[batch_idx], shape=(Dc, 1))
            wp.tile_matmul(Jt, r, g)
            wp.tile_store(Jtr[batch_idx], g)
            rr = wp.tile_zeros(shape=(1, 1), dtype=wp.float32)
            wp.tile_matmul(wp.tile_transpose(r), r, rr)
            costs[batch_idx] = costs[batch_idx] + wp.float32(0.5) * rr[0, 0]
    else:
        Cc, Kc = wp.constant(min(R, _LM_SOLVE_CHUNK)), wp.constant(16)

        def accumulate(
            jacobians: wp.array3d(dtype=wp.float32),
            residuals_col: wp.array3d(dtype=wp.float32),
            JtJ: wp.array3d(dtype=wp.float32),
            Jtr: wp.array3d(dtype=wp.float32),
            costs: wp.array1d(dtype=wp.float32),
        ):
            batch_idx = wp.tid()
            row_blocks = (jacobians.shape[1] + Cc - 1) // Cc
            column_blocks = (jacobians.shape[2] + Kc - 1) // Kc
            rr = wp.tile_zeros(shape=(1, 1), dtype=wp.float32)
            for row in range(row_blocks):
                r = wp.tile_load(residuals_col[batch_idx], shape=(Cc, 1), offset=(row * Cc, 0))
                wp.tile_matmul(wp.tile_transpose(r), r, rr)
                for i in range(column_blocks):
                    Ji = wp.tile_load(jacobians[batch_idx], shape=(Cc, Kc), offset=(row * Cc, i * Kc))
                    Jit = wp.tile_transpose(Ji)
                    g = wp.tile_load(Jtr[batch_idx], shape=(Kc, 1), offset=(i * Kc, 0))
                    wp.tile_matmul(Jit, r, g)
                    wp.tile_store(Jtr[batch_idx], g, offset=(i * Kc, 0))
                    for j in range(column_blocks):
                        Jj = wp.tile_load(jacobians[batch_idx], shape=(Cc, Kc), offset=(row * Cc, j * Kc))
                        H = wp.tile_load(JtJ[batch_idx], shape=(Kc, Kc), offset=(i * Kc, j * Kc))
                        wp.tile_matmul(Jit, Jj, H)
                        wp.tile_store(JtJ[batch_idx], H, offset=(i * Kc, j * Kc))
            costs[batch_idx] = costs[batch_idx] + wp.float32(0.5) * rr[0, 0]

    accumulate.__name__ = accumulate.__qualname__ = f"_lm_accum_{R}x{D}" if fits else f"_lm_accum_blocks_{Cc}"
    k = wp.kernel(enable_backward=False, module="unique")(accumulate)
    _KERNEL_CACHE[key] = k
    return k


def _accept_kernel(R: int):
    """Fused accept/reject on proposed residuals: inlines `0.5‖r_prop‖²`, rho check, λ update, cost write, in one."""
    key = ("accept", R)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    Rc = wp.constant(R)

    def accept(
        costs: wp.array1d(dtype=wp.float32),
        r_prop: wp.array2d(dtype=wp.float32),
        pred: wp.array1d(dtype=wp.float32),
        rho_min: wp.float32,
        gain_ratio_epsilon: wp.float32,
        lam_f: wp.float32,
        lam_lo: wp.float32,
        lam_hi: wp.float32,
        lam: wp.array1d(dtype=wp.float32),
        acc: wp.array1d(dtype=wp.int32),
    ):
        batch_idx = wp.tid()
        s = wp.float32(0.0)
        for i in range(Rc):
            v = r_prop[batch_idx, i]
            s += v * v
        c = wp.float32(0.5) * s
        accepted = pred[batch_idx] > wp.float32(0.0)
        if accepted:
            rho = (costs[batch_idx] - c) / (pred[batch_idx] + gain_ratio_epsilon)
            accepted = rho >= rho_min
        if accepted:
            acc[batch_idx] = 1
            lam[batch_idx] = wp.max(lam[batch_idx] / lam_f, lam_lo)
            costs[batch_idx] = c
        else:
            acc[batch_idx] = 0
            lam[batch_idx] = wp.clamp(lam[batch_idx] * lam_f, lam_lo, lam_hi)

    accept.__name__ = accept.__qualname__ = f"_lm_accept_r{R}"
    k = wp.kernel(enable_backward=False, module="unique")(accept)
    _KERNEL_CACHE[key] = k
    return k


@wp.kernel(enable_backward=False)
def lm_accept_reject_cost_kernel(
    costs_curr: wp.array1d(dtype=wp.float32),
    costs_prop: wp.array1d(dtype=wp.float32),
    pred_red: wp.array1d(dtype=wp.float32),
    rho_min: wp.float32,
    gain_ratio_epsilon: wp.float32,
    lam_f: wp.float32,
    lam_lo: wp.float32,
    lam_hi: wp.float32,
    lam: wp.array1d(dtype=wp.float32),
    acc: wp.array1d(dtype=wp.int32),
):
    """Streaming-path accept/reject (reads precomputed proposed cost scalar)."""
    batch_idx = wp.tid()
    accepted = pred_red[batch_idx] > wp.float32(0.0)
    if accepted:
        rho = (costs_curr[batch_idx] - costs_prop[batch_idx]) / (pred_red[batch_idx] + gain_ratio_epsilon)
        accepted = rho >= rho_min
    if accepted:
        acc[batch_idx] = 1
        lam[batch_idx] = wp.max(lam[batch_idx] / lam_f, lam_lo)
        costs_curr[batch_idx] = costs_prop[batch_idx]
    else:
        acc[batch_idx] = 0
        lam[batch_idx] = wp.clamp(lam[batch_idx] * lam_f, lam_lo, lam_hi)


@wp.kernel(enable_backward=False)
def _mask_hessian_kernel(H: wp.array3d(dtype=wp.float32), mask: wp.array1d(dtype=wp.float32)):
    """Zero `JᵀJ` rows and columns for inactive DOFs (symmetric)."""
    batch_idx, i, j = wp.tid()  # pyright: ignore[reportAssignmentType]
    H[batch_idx, i, j] = H[batch_idx, i, j] * (mask[i] * mask[j])


# --- conjugate gradients on the sparse Jacobian -------------------------------
@wp.kernel
def sparse_jacobian_mv_parallel_kernel(
    values: wp.array2d(dtype=wp.float32),
    col_indices: wp.array1d(dtype=wp.int32),
    residual_nnz_offsets: wp.array1d(dtype=wp.int32),
    vec: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
):
    """`out = J vec`, one thread per residual row."""
    batch_idx, res_idx = wp.tid()  # type: ignore
    acc = wp.float32(0.0)
    for nz_idx in range(residual_nnz_offsets[res_idx], residual_nnz_offsets[res_idx + 1]):
        acc += values[batch_idx, nz_idx] * vec[batch_idx, col_indices[nz_idx]]
    out[batch_idx, res_idx] = acc


@wp.kernel
def zero_masked_jacobian_values_kernel(
    values: wp.array2d(dtype=wp.float32),
    col_indices: wp.array1d(dtype=wp.int32),
    active_dof_mask: wp.array1d(dtype=wp.float32),
):
    batch_idx, nz_idx = wp.tid()  # type: ignore
    values[batch_idx, nz_idx] = values[batch_idx, nz_idx] * active_dof_mask[col_indices[nz_idx]]


@wp.kernel
def batch_dot_parallel_kernel(
    vec_a: wp.array2d(dtype=wp.float32),
    vec_b: wp.array2d(dtype=wp.float32),
    out: wp.array1d(dtype=wp.float32),
):
    batch_idx, dof_idx = wp.tid()  # type: ignore
    wp.atomic_add(out, batch_idx, vec_a[batch_idx, dof_idx] * vec_b[batch_idx, dof_idx])


@wp.kernel
def init_cg_state_parallel_kernel(
    gradient: wp.array2d(dtype=wp.float32),
    r: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    rTr: wp.array1d(dtype=wp.float32),
):
    batch_idx, dof_idx = wp.tid()  # type: ignore
    val = -gradient[batch_idx, dof_idx]
    r[batch_idx, dof_idx] = val
    p[batch_idx, dof_idx] = val
    wp.atomic_add(rTr, batch_idx, val * val)


@wp.kernel
def update_x_and_r_parallel_kernel(
    x: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    r: wp.array2d(dtype=wp.float32),
    Ap: wp.array2d(dtype=wp.float32),
    alpha: wp.array1d(dtype=wp.float32),
    rTr_out: wp.array1d(dtype=wp.float32),
):
    batch_idx, dof_idx = wp.tid()  # type: ignore
    a = alpha[batch_idx]
    x_val = x[batch_idx, dof_idx] + a * p[batch_idx, dof_idx]
    r_val = r[batch_idx, dof_idx] - a * Ap[batch_idx, dof_idx]
    x[batch_idx, dof_idx] = x_val
    r[batch_idx, dof_idx] = r_val
    wp.atomic_add(rTr_out, batch_idx, r_val * r_val)


@wp.kernel
def update_p_parallel_kernel(
    p: wp.array2d(dtype=wp.float32),
    r: wp.array2d(dtype=wp.float32),
    beta: wp.array1d(dtype=wp.float32),
):
    batch_idx, dof_idx = wp.tid()  # type: ignore
    p[batch_idx, dof_idx] = r[batch_idx, dof_idx] + beta[batch_idx] * p[batch_idx, dof_idx]


@wp.kernel
def compute_cg_alpha_kernel(
    rTr: wp.array1d(dtype=wp.float32),
    pAp: wp.array1d(dtype=wp.float32),
    eps: float,
    alpha: wp.array1d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    alpha[batch_idx] = rTr[batch_idx] / (pAp[batch_idx] + eps)


@wp.kernel
def compute_cg_beta_and_update_rTr_kernel(
    rTr: wp.array1d(dtype=wp.float32),
    new_rTr: wp.array1d(dtype=wp.float32),
    eps: float,
    beta: wp.array1d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    beta[batch_idx] = new_rTr[batch_idx] / (rTr[batch_idx] + eps)
    rTr[batch_idx] = new_rTr[batch_idx]


@wp.kernel
def add_lambda_parallel_kernel(
    vec: wp.array2d(dtype=wp.float32),
    lambda_values: wp.array1d(dtype=wp.float32),
    base: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
):
    batch_idx, dof_idx = wp.tid()  # type: ignore
    out[batch_idx, dof_idx] = base[batch_idx, dof_idx] + lambda_values[batch_idx] * vec[batch_idx, dof_idx]


@wp.kernel
def compute_predicted_reduction_fixed_kernel(
    delta: wp.array2d(dtype=wp.float32),
    gradient: wp.array2d(dtype=wp.float32),
    Av: wp.array2d(dtype=wp.float32),
    out: wp.array1d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    grad_dot = wp.float32(0.0)
    quad_dot = wp.float32(0.0)
    for i in range(delta.shape[1]):
        d = delta[batch_idx, i]
        grad_dot += d * gradient[batch_idx, i]
        quad_dot += d * Av[batch_idx, i]
    out[batch_idx] = -grad_dot - wp.float32(0.5) * quad_dot


@dataclass(frozen=True)
class LMOptimizerConfig(OptimizerConfig):
    lm_lambda: float = 1e-3
    """Initial damping."""
    lambda_factor: float = 2.0
    """Damping divides by this on accept and multiplies on reject."""
    lambda_min: float = 1e-5
    """Smallest damping."""
    lambda_max: float = 1e10
    """Largest damping."""
    rho_min: float = 1e-3
    """Minimum gain ratio rho to accept a step."""
    gain_ratio_epsilon: float = 1e-8
    """Stabilizer in the rho denominator; 0 uses the exact ratio."""
    linear_solver: Literal["cholesky", "cg", "torch_cholesky"] = "cholesky"
    """Native Cholesky, matrix-free CG, or optional Torch float64 Cholesky for large dense systems."""
    cg_max_iter: Optional[int] = None
    """CG iterations per step; None: max(4, min(64, tangent_dim // 4))."""
    cg_tol: float = 1e-6
    """Squared CG residual that stops early; checked on the host, so only without CUDA graphs."""
    cg_check_interval: int = 4
    """CG iterations between those host checks."""

    def build(
        self, terms: Sequence[Task], device: wp_device_type, active_dof_mask: Optional[wp.array] = None
    ) -> "LMOptimizer":
        """CG and Torch Cholesky use eager steps; the native backend retains CUDA graph capture."""
        config = replace(self, cuda_graph_mode="none") if self.linear_solver in ("cg", "torch_cholesky") else self
        return LMOptimizer(terms, config, device, active_dof_mask)


class LMOptimizer(Optimizer):
    """Levenberg-Marquardt with Cholesky or conjugate gradients.

    Solver paths:
    - Cholesky/default Jacobian terms: fused fast path.
    - Cholesky/other terms: accumulate `JᵀJ`, then factor in one tile or 16×16 tiles within the sparse-pattern band.
    - CG: reuse sparse/dense Jacobians without forming `JᵀJ`.
    """

    _config_type = LMOptimizerConfig

    def _build(self, var: VarValues) -> None:
        cfg, B, D, dev = self.config, var.batch_size, var.tangent_dim, self.device
        if cfg.linear_solver not in ("cholesky", "cg", "torch_cholesky"):
            raise ValueError("linear_solver must be 'cholesky', 'cg' or 'torch_cholesky'.")
        if self._active_dof_mask is not None and self._active_dof_mask.shape != (D,):
            raise ValueError("active_dof_mask must have shape (total_tangent_dim,).")
        zf = lambda s, dt=wp.float32: wp.zeros(s, dtype=dt, device=dev)  # noqa: E731
        self._split_graphs: Dict[int, tuple] = {}  # batch size -> (propose, accept) graphs around eager hooks
        self.total_tangent_dim = D
        self.max_batch_size = self.batch_size = B
        self._proposed_var = var.clone()
        self.delta, self.lm = zf((B, D)), wp.empty((B,), dtype=wp.float32, device=dev)
        self.costs, self.proposed_costs, self.pred_reduction = zf((B,)), zf((B,)), zf((B,))
        self.accept_mask = zf((B,), wp.int32)
        self._accept_args = (cfg.rho_min, cfg.gain_ratio_epsilon, cfg.lambda_factor, cfg.lambda_min, cfg.lambda_max)
        # Streaming terms run eager prepare/on_step hooks outside captured graphs, forcing split capture.
        self._stream_terms: List[EagerTask] = [t for t in self.terms if isinstance(t, EagerTask)]
        self._fused_mode = len(self._stream_terms) == 0
        self._task_stop = False
        self._wide_J = self._wide_r = self._wide_pr = None
        self._fused_solve = self._fused_accept = self._chunked_accum = None
        if cfg.linear_solver == "cg":
            if not self.terms:
                raise TypeError("linear_solver='cg' needs at least one term.")
            self._build_terms(var)
            self._build_jacobian(var, B)  # merged sparse rows when every term is a SparseTask, else a dense J
            if self._sparse_jac and not self._sparse_rows_sorted:
                raise ValueError("linear_solver='cg' needs each SparseTask to emit its rows in non-decreasing order.")
            if not self._sparse_jac:
                self._dense_jv_kernel = _jv_kernel(self.total_residual_dim, D)
            self.gradient, self.jx = zf((B, D)), zf((B, self.total_residual_dim))
            self.jt_jx, self.cg_r, self.cg_p, self.cg_Ap = zf((B, D)), zf((B, D)), zf((B, D)), zf((B, D))
            self.cg_rTr, self.cg_new_rTr, self.cg_pAp, self.cg_alpha, self.cg_beta = (zf((B,)) for _ in range(5))
            self.cg_max_iter = cfg.cg_max_iter if cfg.cg_max_iter is not None else max(4, min(64, D // 4))
            return

        self.hessian, self.gradient = zf((B, D, D)), zf((B, D, 1))
        bandwidth = D - 1
        if self.terms and self.is_all_sparse:  # the widest column span of any sparse row bounds JᵀJ's band
            bandwidth = 0
            for task in self.terms:
                pattern = task.compute_sparse_jacobian_pattern(var)
                rows, cols = pattern.row_indices.numpy(), pattern.col_indices.numpy()
                low = np.full(task.residual_dim, D, dtype=np.int32)
                high = np.zeros(task.residual_dim, dtype=np.int32)
                np.minimum.at(low, rows, cols)
                np.maximum.at(high, rows, cols)
                bandwidth = max(bandwidth, int(np.max(high - low, initial=0)))
        self._cholesky_kernel = _cholesky_kernel(D, bandwidth)

        # Default terms fill the wide residual/Jacobian buffers at these row offsets; others stream their own.
        self._default_terms: List[ResidualTask] = []
        self._default_offsets: List[int] = []
        ao = 0
        for t in self.terms:
            if type(t).accumulate_normal_equations is ResidualTask.accumulate_normal_equations:
                self._default_terms.append(t)
                self._default_offsets.append(ao)
                ao += t.residual_dim
        has_stream = len(self._default_terms) < len(self.terms)
        # Fast path activates only when CUDA + every term is default (no streaming override).
        is_cuda = dev is not None and dev.is_cuda
        chunk_rows = min(ao, _LM_SOLVE_CHUNK)
        fast = (
            is_cuda
            and cfg.linear_solver != "torch_cholesky"
            and not has_stream
            and (chunk_rows + D + 2) * D + 2 * chunk_rows <= _LM_FUSED_MAX_ENTRIES
        )
        self._use_streams = (
            all(t.parallel_safe for t in self.terms)
            and fast
            and len(self.terms) > 1
            and (len(self.terms) < 6 or B >= 128)
        )
        self._term_streams = [wp.Stream(dev) for _ in self.terms] if self._use_streams else []
        self._term_events = [wp.Event(dev) for _ in self.terms] if self._use_streams else []
        # _R_accum_buf is the padded row count the solve kernel sees (chunked path); `ao` is the true count.
        self._R_accum_buf = ao
        self._chunk_nc = 0
        if fast and ao > 0:
            if (ao + D + 2) * D + 2 * ao > _LM_FUSED_MAX_ENTRIES:
                self._R_accum_buf = -(-ao // _LM_SOLVE_CHUNK) * _LM_SOLVE_CHUNK
                self._chunk_nc = self._R_accum_buf // _LM_SOLVE_CHUNK
                self._chunked_accum = _accum_kernel_chunked(D, _LM_SOLVE_CHUNK)
            else:
                self._fused_solve = _solve_kernel(ao, D)
            self._wide_r = wp.zeros((B, self._R_accum_buf), dtype=wp.float32, device=dev)
            self._wide_J = wp.zeros((B, self._R_accum_buf, D), dtype=wp.float32, device=dev)
        if fast and ao > 0:
            self._wide_pr = wp.zeros((B, ao), dtype=wp.float32, device=dev)
            self._fused_accept = _accept_kernel(ao)

    def _start(self, var: VarValues) -> None:
        var.invalidate()
        self.lm.fill_(self.config.lm_lambda)
        if self.config.max_iter == 0 or self._iteration_callback is not None:  # else the first step scores the start
            self._compute_costs(var, self.costs)

    def _compute_costs(self, var: VarValues, costs: wp.array):
        if self.config.linear_solver == "cg":  # the same reduction scores current and proposed residuals
            self._precompute_terms(var, need_gradient=False)
            self._fill_jacobian(var, self.residuals, need_jacobian=False)
            self._reduce_cost_gradient(self.residuals, costs, None)
            return
        costs.zero_()
        for task in self.terms:
            task.accumulate_normal_equations(var, costs=costs)

    _BATCH_BUFFERS = (
        "_proposed_var",
        "hessian",
        "gradient",
        "delta",
        "lm",
        "costs",
        "proposed_costs",
        "pred_reduction",
        "accept_mask",
        "_wide_r",
        "_wide_J",
        "_wide_pr",
        "residuals",
        "jacobians",
        "_sparse_jac_values",
        "jx",
        "jt_jx",
        "cg_r",
        "cg_p",
        "cg_Ap",
        "cg_rTr",
        "cg_new_rTr",
        "cg_pAp",
        "cg_alpha",
        "cg_beta",
    )

    def _capture(self, var: VarValues, mode: str) -> None:
        """Eager terms (hec's mask alignment) run hooks between propose and accept, so those get two graphs."""
        if self._fused_mode:
            return super()._capture(var, mode)
        if mode != "iter":
            raise ValueError("terms with eager hooks need cuda_graph_mode='iter'")
        # Warmup fills damping before eager terms permit scoring; reset scores after prepare.
        dev, live = self.device, self.batch_size
        initial_var = var.clone()
        var.invalidate()
        self.lm.fill_(self.config.lm_lambda)
        self._step_body(var)
        wp.synchronize()
        self.accept_mask.fill_(1)
        var.accept(self.accept_mask, initial_var)
        wp.synchronize()
        var.invalidate()
        self.lm.fill_(self.config.lm_lambda)
        self._iter = 0
        for t in self._stream_terms:
            t.prepare(var, False, 0)
        with wp.ScopedCapture(device=dev) as cap:
            self._propose_body(var)
        propose = cap.graph
        for t in self._stream_terms:
            t.prepare(self._proposed_var, True, 0)
        with wp.ScopedCapture(device=dev) as cap:
            self._accept_body(var)
        self._split_graphs[live] = (propose, cap.graph)
        self._graphs[(live, mode)] = None  # captured, but `step` must run `_step_body` for the hooks

    def _run_terms(self, terms: List[ResidualTask], offsets: List[int], fn: Callable[[ResidualTask, int], None]):
        """Run `fn(term, offset)` per term, using per-term streams when available."""
        if not self._term_streams:
            for term, off in zip(terms, offsets):
                fn(term, off)
            return
        main = wp.get_stream(self.device)
        init = main.record_event()
        for term, off, stream, evt in zip(terms, offsets, self._term_streams, self._term_events):
            stream.wait_event(init)
            with wp.ScopedStream(stream):
                fn(term, off)
            stream.record_event(evt)
        for evt in self._term_events[: len(terms)]:
            main.wait_event(evt)

    def _apply_linear_operator(self, vec: wp.array, out: wp.array):
        """`out = (JᵀJ + λI) vec` from the filled Jacobian; leaves `JᵀJ vec` in `jt_jx`."""
        B, dev = self.batch_size, self.device
        if self._sparse_jac:
            wp.launch(
                sparse_jacobian_mv_parallel_kernel,
                dim=(B, self.total_residual_dim),
                inputs=[self._sparse_jac_values, self._sparse_cols, self._sparse_row_starts, vec],
                outputs=[self.jx],
                device=dev,
            )
            self.jt_jx.zero_()
            wp.launch(
                _sparse_jtr_split_kernel,
                dim=(B, self._sparse_num_splits),
                inputs=[
                    self.jx,
                    self._sparse_jac_values,
                    self._sparse_value_perm,
                    self._sparse_row_perm,
                    self._sparse_split_cols,
                    self._sparse_split_offsets,
                    0,
                    1,
                ],
                outputs=[None, self.jt_jx],
                device=dev,
            )
        else:  # the tiled products the dense pipeline already uses for Jᵀr
            wp.launch_tiled(
                self._dense_jv_kernel,
                dim=[B],
                inputs=[self.jacobians, vec],
                outputs=[self.jx],
                block_dim=TILE_THREADS,
                device=dev,
            )
            wp.launch_tiled(
                self._dense_jtr_kernel,
                dim=[B],
                inputs=[self.jacobians, self.jx],
                outputs=[self.jt_jx],
                block_dim=TILE_THREADS,
                device=dev,
            )
        wp.launch(
            add_lambda_parallel_kernel,
            dim=(B, self.total_tangent_dim),
            inputs=[vec, self.lm, self.jt_jx],
            outputs=[out],
            device=dev,
        )

    def _compute_sparse_delta(self, skip_early_stop: bool):
        """Solve `(JᵀJ + λI) δ = -g` by CG, then the undamped Gauss-Newton reduction `-gᵀδ - ½δᵀJᵀJδ`."""
        B, D, dev = self.batch_size, self.total_tangent_dim, self.device
        self.delta.zero_()
        self.cg_rTr.zero_()
        wp.launch(
            init_cg_state_parallel_kernel,
            dim=(B, D),
            inputs=[self.gradient],
            outputs=[self.cg_r, self.cg_p, self.cg_rTr],
            device=dev,
        )
        for cg_iter in range(self.cg_max_iter):
            self._apply_linear_operator(self.cg_p, self.cg_Ap)
            self.cg_pAp.zero_()
            wp.launch(
                batch_dot_parallel_kernel, dim=(B, D), inputs=[self.cg_p, self.cg_Ap], outputs=[self.cg_pAp], device=dev
            )
            wp.launch(
                compute_cg_alpha_kernel,
                dim=B,
                inputs=[self.cg_rTr, self.cg_pAp, _CG_EPS],
                outputs=[self.cg_alpha],
                device=dev,
            )
            self.cg_new_rTr.zero_()
            wp.launch(
                update_x_and_r_parallel_kernel,
                dim=(B, D),
                inputs=[self.delta, self.cg_p, self.cg_r, self.cg_Ap, self.cg_alpha],
                outputs=[self.cg_new_rTr],
                device=dev,
            )
            if not skip_early_stop and cg_iter % self.config.cg_check_interval == 0 and cg_iter > 0:
                if self.cg_new_rTr.numpy().max() < self.config.cg_tol:
                    break
            wp.launch(
                compute_cg_beta_and_update_rTr_kernel,
                dim=B,
                inputs=[self.cg_rTr, self.cg_new_rTr, _CG_EPS],
                outputs=[self.cg_beta],
                device=dev,
            )
            wp.launch(update_p_parallel_kernel, dim=(B, D), inputs=[self.cg_p, self.cg_r, self.cg_beta], device=dev)
        self._apply_linear_operator(self.delta, self.cg_Ap)
        wp.launch(
            compute_predicted_reduction_fixed_kernel,
            dim=B,
            inputs=[self.delta, self.gradient, self.jt_jx],
            outputs=[self.pred_reduction],
            device=dev,
        )

    def _propose_body(self, var: VarValues):
        B = self.batch_size
        if self.config.linear_solver == "cg":
            self._precompute_terms(var, need_gradient=True)
            self._fill_jacobian(var, self.residuals, need_jacobian=True)
            if self._active_dof_mask is not None:  # zero frozen columns in J itself, so CG's J·v leaves them too
                if self._sparse_jac:
                    wp.launch(
                        zero_masked_jacobian_values_kernel,
                        dim=self._sparse_jac_values.shape,
                        inputs=[self._sparse_jac_values, self._sparse_cols, self._active_dof_mask],
                        device=self.device,
                    )
                else:
                    rows = B * self.total_residual_dim
                    wp.launch(
                        apply_dof_mask_kernel,
                        dim=(rows, self.total_tangent_dim),
                        inputs=[self.jacobians.reshape((rows, self.total_tangent_dim)), self._active_dof_mask],
                        device=self.device,
                    )
            self._reduce_cost_gradient(self.residuals, self.costs, self.gradient)
            cfg = self.config
            # CG convergence reads back to the host; captured solves use their fixed iteration budget.
            self._compute_sparse_delta(
                skip_early_stop=cfg.cuda_graph_mode != "none" or cfg.early_stopping_interval == 0
            )
        elif self._fused_solve is not None or self._chunked_accum is not None:
            for term in self.terms:
                term.precompute(var, need_gradient=True)
            # Analytic Jacobian kernels write only their row-slice, so zero wide_J once; residuals overwrite theirs.
            self._wide_J.zero_()

            def _fn(t, o):
                t.compute_weighted_residual(var, out_residual=self._wide_r, row_offset=o)
                t.compute_weighted_jacobian(var, out_jacobian=self._wide_J, row_offset=o)

            self._run_terms(self._default_terms, self._default_offsets, _fn)
            # Mask Jacobian columns for frozen DOFs BEFORE forming JᵀJ/Jᵀr so they don't leak into the active-DOF step.
            if self._active_dof_mask is not None:
                rows = B * self._R_accum_buf
                wp.launch(
                    apply_dof_mask_kernel,
                    dim=(rows, self.total_tangent_dim),
                    inputs=[self._wide_J.reshape((rows, self.total_tangent_dim)), self._active_dof_mask],
                    device=self.device,
                )
            if self._chunked_accum is not None:
                # many residuals: accumulate JᵀJ in row-blocks, then solve the small D×D system
                Dd = self.total_tangent_dim
                J4 = self._wide_J.reshape((B, self._chunk_nc, _LM_SOLVE_CHUNK, Dd))
                r4 = self._wide_r.reshape((B, self._chunk_nc, _LM_SOLVE_CHUNK, 1))
                wp.launch_tiled(
                    self._chunked_accum,
                    dim=[B],
                    inputs=[J4, r4, self._chunk_nc],
                    outputs=[self.hessian, self.gradient, self.costs],
                    block_dim=TILE_THREADS,
                    device=self.device,
                )
                wp.launch_tiled(
                    self._cholesky_kernel,
                    dim=[B],
                    inputs=[self.hessian, self.gradient, self.lm],
                    outputs=[self.delta, self.pred_reduction],
                    block_dim=TILE_THREADS,
                    device=self.device,
                )
            else:
                r_col = self._wide_r.reshape((B, self._R_accum_buf, 1))
                wp.launch_tiled(
                    self._fused_solve,
                    dim=[B],
                    inputs=[self._wide_J, r_col, self.lm],
                    outputs=[self.costs, self.delta, self.pred_reduction],
                    block_dim=TILE_THREADS,
                    device=self.device,
                )
        else:
            self.hessian.zero_()
            self.gradient.zero_()
            self.costs.zero_()
            for t in self.terms:
                t.accumulate_normal_equations(var, costs=self.costs, JtJ=self.hessian, Jtr=self.gradient)
            # mask JᵀJ rows/cols and Jᵀr entries for frozen DOFs
            if self._active_dof_mask is not None:
                D = self.total_tangent_dim
                wp.launch(
                    _mask_hessian_kernel,
                    dim=(B, D, D),
                    inputs=[self.hessian, self._active_dof_mask],
                    device=self.device,
                )
                wp.launch(
                    apply_dof_mask_kernel,
                    dim=(B, D),
                    inputs=[self.gradient.reshape((B, D)), self._active_dof_mask],
                    device=self.device,
                )
            if self.config.linear_solver == "torch_cholesky":
                import torch

                stream = (
                    torch.cuda.ExternalStream(wp.get_stream(self.device).cuda_stream, device=self.device.ordinal)
                    if self.device.is_cuda
                    else None
                )
                with torch.cuda.stream(stream) if stream is not None else nullcontext():
                    matrix = wp.to_torch(self.hessian).double()
                    gradient = wp.to_torch(self.gradient).double()
                    damping = wp.to_torch(self.lm)
                    matrix.diagonal(dim1=-2, dim2=-1).add_(damping[:, None])
                    factor, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
                    delta = torch.cholesky_solve(-gradient, factor)[..., 0]
                    delta = torch.where((info == 0)[:, None], delta, 0.0).float()
                    wp.to_torch(self.delta).copy_(delta)
                    wp.to_torch(self.pred_reduction).copy_(
                        0.5 * (delta * (damping[:, None] * delta - wp.to_torch(self.gradient)[..., 0])).sum(-1)
                    )
            else:
                wp.launch_tiled(
                    self._cholesky_kernel,
                    dim=[B],
                    inputs=[self.hessian, self.gradient, self.lm],
                    outputs=[self.delta, self.pred_reduction],
                    block_dim=TILE_THREADS,
                    device=self.device,
                )
        var.integrate(self.delta, out=self._proposed_var, tangent_mask=self._active_dof_mask)

    def _accept_body(self, var: VarValues):
        B, dev = self.batch_size, self.device
        out = [self.lm, self.accept_mask]
        if self._fused_accept is not None:
            for term in self.terms:
                term.precompute(self._proposed_var, need_gradient=False)
            pv = self._proposed_var

            def _fn(t, o):
                t.compute_weighted_residual(pv, out_residual=self._wide_pr, row_offset=o)

            self._run_terms(self._default_terms, self._default_offsets, _fn)
            args = [self.costs, self._wide_pr, self.pred_reduction, *self._accept_args]
            wp.launch(self._fused_accept, dim=B, inputs=args, outputs=out, device=dev)
        else:
            self._compute_costs(self._proposed_var, self.proposed_costs)
            args = [self.costs, self.proposed_costs, self.pred_reduction, *self._accept_args]
            wp.launch(lm_accept_reject_cost_kernel, dim=B, inputs=args, outputs=out, device=dev)
        var.accept(self.accept_mask, self._proposed_var)

    def _step_body(self, var: VarValues) -> None:
        if self._fused_mode:
            self._propose_body(var)
            self._accept_body(var)
            return
        propose, accept = self._split_graphs.get(self.batch_size, (None, None))
        for t in self._stream_terms:
            t.prepare(var, False, self._iter)
        if propose is not None:
            wp.capture_launch(propose)
        else:
            self._propose_body(var)
        for t in self._stream_terms:
            t.prepare(self._proposed_var, True, self._iter)
        if accept is not None:
            wp.capture_launch(accept)
        else:
            self._accept_body(var)
        stop = False
        for t in self._stream_terms:
            stop = t.on_step(self.accept_mask, self._iter, self.costs) or stop
        self._task_stop = stop
