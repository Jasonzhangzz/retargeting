# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
"""Batched iLQR: iLQG on any batched dynamics, every step of it on the device.

iLQG: https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf. MJPC: https://arxiv.org/abs/2212.00541.

Each solve:
1. Nominal rollout: the initial guess, at the feedback scales.
2. Linearization: forward differences of `Dynamics.step` and the terms' residuals.
3. Backward pass: Gauss-Newton costs, control regularization, and a box QP on bounded controls.
4. Line search: one rollout per step size, every instance at once.
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import warp as wp

from robokit.opt.optimizer import OptimizerConfig
from robokit.opt.var_values import Var, VarKey, VarValues, VectorVar
from robokit.terms.simulation.sim_context import SimContext
from robokit.terms.task import Context, ResidualTask
from robokit.utils.warp_utils import wp_device_type


f64 = wp.float64  # Warp 1.10 (the last for Python 3.8) types f64(x) as int-only: hence the ignores
_STEP_SIZES = np.r_[np.logspace(0, -3, 9), 0.0]  # line-search step sizes
_FD_ROWS = 2**15  # most finite-difference rows stepped at once
_UNBOUNDED = 1e30  # stands in for an infinite control bound
_BOX_QP_MAX_ITER = wp.constant(100)
_ARMIJO_DECREASE = wp.constant(0.1)  # the box QP line search's sufficient decrease
_ARMIJO_SHRINK = wp.constant(0.6)


# --- derivative kernels -----------------------------------------------------
@wp.kernel
def _set_fd_rows_kernel(
    knot0: int,
    rows_per_knot: int,
    total_knots: int,
    ctrls: wp.array3d(dtype=float),  # (N, T, nu)
    fd_eps: wp.array(dtype=f64),  # (ndx + nu,)
    ctrl_upper: wp.array(dtype=float),
    state_steps: wp.array2d(dtype=float),  # (rows, ndx)
    row_ctrls: wp.array2d(dtype=float),  # (rows, nu)
    step_sizes: wp.array(dtype=f64),  # (rows,)
):
    """Set each row's state step, controls and step size; a knot's first row steps nothing."""
    row = wp.tid()
    knot = wp.min(knot0 + row // rows_per_knot, total_knots - 1)  # pyright: ignore[reportOperatorIssue]
    column = row % rows_per_knot - 1  # pyright: ignore[reportOperatorIssue]
    horizon, state_dim, ctrl_dim = ctrls.shape[1], state_steps.shape[1], row_ctrls.shape[1]
    instance = knot // (horizon + 1)
    t = knot % (horizon + 1)
    size = f64(0.0)  # pyright: ignore[reportArgumentType]
    if column >= 0:
        size = fd_eps[column]
    for j in range(state_dim):
        state_steps[row, j] = 0.0
    if column >= 0 and column < state_dim:
        state_steps[row, column] = float(size)
    for j in range(ctrl_dim):
        row_ctrls[row, j] = 0.0
        if t < horizon:
            row_ctrls[row, j] = ctrls[instance, t, j]
    if column >= state_dim:
        j = column - state_dim
        if row_ctrls[row, j] + float(size) > ctrl_upper[j]:  # step down at an upper bound
            size = -size
        row_ctrls[row, j] += float(size)
    step_sizes[row] = size


@wp.kernel
def _compute_fd_jacobians_kernel(
    knot0: int,
    rows_per_knot: int,
    next_state_diffs: wp.array2d(dtype=float),  # (rows, ndx) each row's next state less its knot's unstepped one
    residuals: wp.array2d(dtype=float),  # (rows, nr)
    step_sizes: wp.array(dtype=f64),
    dynamics_jacobians: wp.array4d(dtype=f64),  # (N, T, ndx, ndx + nu)
    residual_jacobians: wp.array4d(dtype=f64),  # (N, T + 1, ndx + nu, nr)
):
    """Compute one column of a knot's dynamics and residual Jacobians by forward differences."""
    chunk_knot, column = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    horizon = dynamics_jacobians.shape[1]
    knot = knot0 + chunk_knot
    base = chunk_knot * rows_per_knot
    row = base + column + 1
    instance = knot // (horizon + 1)
    t = knot % (horizon + 1)
    size = step_sizes[row]
    for m in range(residuals.shape[1]):
        residual_jacobians[instance, t, column, m] = (f64(residuals[row, m]) - f64(residuals[base, m])) / size
    if t < horizon:
        for i in range(next_state_diffs.shape[1]):
            dynamics_jacobians[instance, t, i, column] = f64(next_state_diffs[row, i]) / size


@wp.kernel
def _compute_gauss_newton_kernel(
    knot0: int,
    rows_per_knot: int,
    residuals: wp.array2d(dtype=float),  # (rows, nr), read at each knot's unstepped row
    residual_jacobians: wp.array4d(dtype=f64),
    cost_grad: wp.array3d(dtype=f64),  # (N, T + 1, ndx + nu)
    cost_hess: wp.array4d(dtype=f64),  # (N, T + 1, ndx + nu, ndx + nu)
):
    """Compute row i of a knot's Gauss-Newton cost gradient `J'r` and Hessian `J'J`."""
    chunk_knot, i = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    num_knots, knot_dim, residual_dim = residual_jacobians.shape[1], residual_jacobians.shape[2], residuals.shape[1]
    instance = (knot0 + chunk_knot) // num_knots
    t = (knot0 + chunk_knot) % num_knots
    base = chunk_knot * rows_per_knot
    s = f64(0.0)  # pyright: ignore[reportArgumentType]
    for m in range(residual_dim):
        s += residual_jacobians[instance, t, i, m] * f64(residuals[base, m])
    cost_grad[instance, t, i] = s
    for j in range(knot_dim):
        s = f64(0.0)  # pyright: ignore[reportArgumentType]
        for m in range(residual_dim):
            s += residual_jacobians[instance, t, i, m] * residual_jacobians[instance, t, j, m]
        cost_hess[instance, t, i, j] = s


# --- backward pass kernels --------------------------------------------------
@wp.func
def _compute_cholesky_func(
    matrix: wp.array2d(dtype=f64), mu: f64, free: wp.array(dtype=int), factor: wp.array2d(dtype=f64)
) -> int:
    """Factor `matrix + mu I` on the free rows and columns, identity elsewhere; return 0 unless positive definite."""
    for i in range(matrix.shape[0]):
        for j in range(i + 1):
            if free[i] == 0 or free[j] == 0:
                factor[i, j] = f64(0.0)  # pyright: ignore[reportArgumentType]
                if i == j:
                    factor[i, j] = f64(1.0)  # pyright: ignore[reportArgumentType]
            else:
                s = matrix[i, j]
                if i == j:
                    s += mu
                for m in range(j):
                    s -= factor[i, m] * factor[j, m]
                if i == j:
                    if not (s > f64(0.0)):  # pyright: ignore[reportArgumentType]
                        return 0
                    factor[i, i] = wp.sqrt(s)
                else:
                    factor[i, j] = s / factor[j, j]
    return 1


@wp.func
def _compute_cho_solve_func(factor: wp.array2d(dtype=f64), b: wp.array(dtype=f64)):
    """Overwrite `b` with `(L L')^-1 b` for the Cholesky factor `L`."""
    n = factor.shape[0]
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= factor[i, j] * b[j]
        b[i] = s / factor[i, i]
    for ii in range(n):
        i = n - 1 - ii
        s = b[i]
        for j in range(i + 1, n):
            s -= factor[j, i] * b[j]
        b[i] = s / factor[i, i]


@wp.func
def _compute_box_qp_cost_func(
    quu: wp.array2d(dtype=f64), qu: wp.array(dtype=f64), mu: f64, du: wp.array(dtype=f64)
) -> f64:
    """Return the box QP's cost of the step `du`, `du'(qu + (Quu + mu I) du / 2)`."""
    value = f64(0.0)  # pyright: ignore[reportArgumentType]
    for i in range(quu.shape[0]):
        s = mu * du[i]
        for j in range(quu.shape[0]):
            s += quu[i, j] * du[j]
        value += du[i] * (f64(0.5) * s + qu[i])  # pyright: ignore[reportArgumentType]
    return value


@wp.kernel
def _compute_value_hess_ab_kernel(
    t: int,
    dynamics_jacobians: wp.array4d(dtype=f64),
    value_hess: wp.array3d(dtype=f64),
    value_hess_ab: wp.array3d(dtype=f64),  # (N, ndx, ndx + nu)
):
    """Compute entry (i, j) of `Vxx [A B]` at knot t."""
    instance, i, j = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    s = f64(0.0)  # pyright: ignore[reportArgumentType]
    for m in range(value_hess.shape[1]):
        s += value_hess[instance, i, m] * dynamics_jacobians[instance, t, m, j]
    value_hess_ab[instance, i, j] = s


@wp.kernel
def _compute_q_function_kernel(
    t: int,
    dynamics_jacobians: wp.array4d(dtype=f64),
    cost_grad: wp.array3d(dtype=f64),
    cost_hess: wp.array4d(dtype=f64),
    value_grad: wp.array2d(dtype=f64),
    value_hess_ab: wp.array3d(dtype=f64),
    q_grad: wp.array2d(dtype=f64),  # (N, ndx + nu): qx, qu
    q_hess: wp.array3d(dtype=f64),  # (N, ndx + nu, ndx + nu): H + [A B]' Vxx [A B]
):
    """Compute entry (i, j) of knot t's Q-function Hessian, and its gradient entry i when j is 0."""
    instance, i, j = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    state_dim = value_grad.shape[1]
    if j == 0:
        s = cost_grad[instance, t, i]
        for m in range(state_dim):
            s += dynamics_jacobians[instance, t, m, i] * value_grad[instance, m]
        q_grad[instance, i] = s
    if i < state_dim and j >= state_dim:
        return
    s = cost_hess[instance, t, i, j]
    for m in range(state_dim):
        s += dynamics_jacobians[instance, t, m, i] * value_hess_ab[instance, m, j]
    q_hess[instance, i, j] = s


@wp.kernel
def _solve_box_qp_kernel(
    t: int,
    q_grad: wp.array2d(dtype=f64),
    q_hess: wp.array3d(dtype=f64),
    ctrls: wp.array3d(dtype=float),
    ctrl_lower: wp.array(dtype=f64),
    ctrl_upper: wp.array(dtype=f64),
    regularization: wp.array(dtype=f64),
    ok: wp.array(dtype=int),
    factors: wp.array3d(dtype=f64),  # (N, nu, nu) Cholesky factor of the free controls' Quu
    free: wp.array2d(dtype=int),  # (N, nu) 1 where a control's bounds don't bind
    steps: wp.array2d(dtype=f64),  # (N, nu)
    directions: wp.array2d(dtype=f64),
    trials: wp.array2d(dtype=f64),
    grads: wp.array2d(dtype=f64),
    quus: wp.array3d(dtype=f64),
    qus: wp.array2d(dtype=f64),
    feedforward: wp.array3d(dtype=float),  # (N, T + 1, nu), zero at the final knot
):
    """Solve knot t's box QP for its feedforward step, keeping the free controls' factor."""
    instance = wp.tid()
    if ok[instance] == 0:
        return
    factor, free_ctrls, du, direction = factors[instance], free[instance], steps[instance], directions[instance]
    trial, grad, quu, qu = trials[instance], grads[instance], quus[instance], qus[instance]
    ctrl_dim = qu.shape[0]
    state_dim = q_hess.shape[1] - ctrl_dim
    mu = regularization[instance]
    for i in range(ctrl_dim):
        qu[i] = q_grad[instance, state_dim + i]
        free_ctrls[i] = 1
        for j in range(ctrl_dim):
            quu[i, j] = q_hess[instance, state_dim + i, state_dim + j]
    if _compute_cholesky_func(quu, mu, free_ctrls, factor) == 0:
        ok[instance] = 0
        return
    for i in range(ctrl_dim):
        du[i] = -qu[i]
    _compute_cho_solve_func(factor, du)  # the unconstrained step
    out = int(0)
    for i in range(ctrl_dim):
        if du[i] < ctrl_lower[i] - f64(ctrls[instance, t, i]) or du[i] > ctrl_upper[i] - f64(ctrls[instance, t, i]):
            out = 1
    if out == 1:  # projected Newton (Tassa 2014) from the next knot's step
        value = f64(1e300)  # pyright: ignore[reportArgumentType]
        for i in range(ctrl_dim):
            lower, upper = ctrl_lower[i] - f64(ctrls[instance, t, i]), ctrl_upper[i] - f64(ctrls[instance, t, i])
            du[i] = wp.clamp(f64(feedforward[instance, t + 1, i]), lower, upper)
        for _ in range(_BOX_QP_MAX_ITER):
            grad_norm, num_free = f64(0.0), int(0)  # pyright: ignore[reportArgumentType]
            for i in range(ctrl_dim):
                s = qu[i] + mu * du[i]
                for j in range(ctrl_dim):
                    s += quu[i, j] * du[j]
                grad[i] = s
                lower, upper = ctrl_lower[i] - f64(ctrls[instance, t, i]), ctrl_upper[i] - f64(ctrls[instance, t, i])
                free_ctrls[i] = 1
                if (du[i] <= lower and s > f64(0.0)) or (du[i] >= upper and s < f64(0.0)):  # pyright: ignore[reportArgumentType]
                    free_ctrls[i] = 0
                else:
                    grad_norm += s * s
                    num_free += 1
            new_value = _compute_box_qp_cost_func(quu, qu, mu, du)
            small = value - new_value < f64(1e-12) * (f64(1.0) + wp.abs(new_value))  # pyright: ignore[reportArgumentType, reportOperatorIssue]
            if num_free == 0 or wp.sqrt(grad_norm) < f64(1e-8) or small:  # pyright: ignore[reportArgumentType, reportOperatorIssue]
                break
            value = new_value
            _compute_cholesky_func(quu, mu, free_ctrls, factor)
            for i in range(ctrl_dim):
                direction[i] = -grad[i] * f64(free_ctrls[i])
            _compute_cho_solve_func(factor, direction)
            slope = f64(0.0)  # pyright: ignore[reportArgumentType]
            for i in range(ctrl_dim):
                slope += grad[i] * direction[i]
            a = f64(1.0)  # pyright: ignore[reportArgumentType]
            while a > f64(1e-12):  # pyright: ignore[reportArgumentType, reportOperatorIssue]  # projected Armijo
                for i in range(ctrl_dim):
                    lower, upper = (
                        ctrl_lower[i] - f64(ctrls[instance, t, i]),
                        ctrl_upper[i] - f64(ctrls[instance, t, i]),
                    )
                    trial[i] = wp.clamp(du[i] + a * direction[i], lower, upper)
                if _compute_box_qp_cost_func(quu, qu, mu, trial) - value <= f64(_ARMIJO_DECREASE) * a * slope:  # pyright: ignore[reportOperatorIssue]
                    break
                a *= f64(_ARMIJO_SHRINK)
            if a <= f64(1e-12):  # pyright: ignore[reportArgumentType, reportOperatorIssue]
                break
            for i in range(ctrl_dim):
                du[i] = trial[i]
        _compute_cholesky_func(quu, mu, free_ctrls, factor)  # gains on the free controls only
    for j in range(ctrl_dim):
        feedforward[instance, t, j] = float(du[j])


@wp.kernel
def _compute_gains_kernel(
    t: int,
    q_hess: wp.array3d(dtype=f64),
    ok: wp.array(dtype=int),
    factors: wp.array3d(dtype=f64),
    free: wp.array2d(dtype=int),
    columns: wp.array3d(dtype=f64),  # (N, ndx, nu) scratch
    feedback: wp.array4d(dtype=float),  # (N, T, nu, ndx)
):
    """Compute one column of knot t's gains, `-Quu^-1 Qux` on the free controls and zero on clamped ones."""
    instance, column = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    if ok[instance] == 0:
        return
    ctrl_dim = factors.shape[1]
    state_dim = q_hess.shape[1] - ctrl_dim
    gains = columns[instance, column]
    for i in range(ctrl_dim):
        gains[i] = -q_hess[instance, state_dim + i, column] * f64(free[instance, i])
    _compute_cho_solve_func(factors[instance], gains)
    for i in range(ctrl_dim):
        feedback[instance, t, i, column] = float(gains[i] * f64(free[instance, i]))


@wp.kernel
def _compute_q_gain_kernel(
    t: int, q_hess: wp.array3d(dtype=f64), feedback: wp.array4d(dtype=float), q_gain: wp.array3d(dtype=f64)
):
    """Compute entry (j, m) of `Qux + Quu K` at knot t."""
    instance, j, m = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    ctrl_dim = feedback.shape[2]
    state_dim = q_hess.shape[1] - ctrl_dim
    s = q_hess[instance, state_dim + j, m]
    for p in range(ctrl_dim):
        s += q_hess[instance, state_dim + j, state_dim + p] * f64(feedback[instance, t, p, m])
    q_gain[instance, j, m] = s


@wp.kernel
def _compute_value_function_kernel(
    t: int,
    q_grad: wp.array2d(dtype=f64),
    q_hess: wp.array3d(dtype=f64),
    q_gain: wp.array3d(dtype=f64),
    feedforward: wp.array3d(dtype=float),
    feedback: wp.array4d(dtype=float),
    value_grad: wp.array2d(dtype=f64),
    value_hess: wp.array3d(dtype=f64),
):
    """Compute knot t's value Hessian entries (i, m) and (m, i), and on the diagonal its gradient entry i."""
    instance, i, m = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    state_dim, ctrl_dim = value_grad.shape[1], feedforward.shape[2]
    if m < i:
        return
    a, b = f64(q_hess[instance, i, m]), f64(q_hess[instance, m, i])  # qxx is full: H's xx block plus A' Vxx A
    for j in range(ctrl_dim):
        a += f64(feedback[instance, t, j, i]) * q_gain[instance, j, m] + q_hess[instance, state_dim + j, i] * f64(
            feedback[instance, t, j, m]
        )
        b += f64(feedback[instance, t, j, m]) * q_gain[instance, j, i] + q_hess[instance, state_dim + j, m] * f64(
            feedback[instance, t, j, i]
        )
    value_hess[instance, i, m] = f64(0.5) * (a + b)  # pyright: ignore[reportArgumentType]
    value_hess[instance, m, i] = f64(0.5) * (a + b)  # pyright: ignore[reportArgumentType]
    if m == i:
        s = f64(q_grad[instance, i])
        for j in range(ctrl_dim):
            qk = f64(q_grad[instance, state_dim + j])
            for p in range(ctrl_dim):
                qk += q_hess[instance, state_dim + j, state_dim + p] * f64(feedforward[instance, t, p])
            s += f64(feedback[instance, t, j, i]) * qk + q_hess[instance, state_dim + j, i] * f64(
                feedforward[instance, t, j]
            )
        value_grad[instance, i] = s


# --- rollout kernels --------------------------------------------------------
@wp.kernel
def _compute_rollout_ctrls_kernel(
    t: int,
    final: int,
    num_step_sizes: int,
    scale_feedback: int,  # 1: the step size scales the feedback instead of the feedforward (nominal rollouts)
    feedback_cap: float,  # > 0: clip the feedback to this times half the control range
    ctrls: wp.array3d(dtype=float),  # (N, T, nu)
    feedforward: wp.array3d(dtype=float),  # (N, T + 1, nu)
    feedback: wp.array4d(dtype=float),  # (N, T, nu, ndx)
    step_sizes: wp.array(dtype=float),
    ctrl_lower: wp.array(dtype=float),
    ctrl_upper: wp.array(dtype=float),
    state_diffs: wp.array2d(dtype=float),  # (W, ndx) each world's state less its reference
    rollout_ctrls: wp.array2d(dtype=float),  # (W, nu)
    ctrl_history: wp.array3d(dtype=float),  # (T + 1, W, nu)
):
    """Compute each world's clipped knot-t control `u + alpha k + K dx` (zero at the final knot) and record it."""
    world = wp.tid()
    instance = world // num_step_sizes  # pyright: ignore[reportOperatorIssue]
    step_size = world % num_step_sizes  # pyright: ignore[reportOperatorIssue]
    for j in range(rollout_ctrls.shape[1]):
        u = float(0.0)
        if final == 0:
            alpha, fb = float(step_sizes[step_size]), float(0.0)
            for m in range(state_diffs.shape[1]):
                fb += feedback[instance, t, j, m] * state_diffs[world, m]
            if feedback_cap > 0.0:
                cap = feedback_cap * 0.5 * (ctrl_upper[j] - ctrl_lower[j])
                fb = wp.clamp(fb, -cap, cap)
            if scale_feedback == 1:
                u = ctrls[instance, t, j] + alpha * fb
            else:
                u = ctrls[instance, t, j] + alpha * feedforward[instance, t, j] + fb
        u = wp.clamp(u, ctrl_lower[j], ctrl_upper[j])
        rollout_ctrls[world, j] = u
        ctrl_history[t, world, j] = u


@wp.kernel
def _compute_rollout_costs_kernel(num_knots: int, residuals: wp.array2d(dtype=float), costs: wp.array(dtype=f64)):
    """Compute each world's rollout cost, `0.5 |r|^2` over its knots' residuals (knot-major rows)."""
    world = wp.tid()
    s = f64(0.0)  # pyright: ignore[reportArgumentType]
    for t in range(num_knots):
        for m in range(residuals.shape[1]):
            v = f64(residuals[t * costs.shape[0] + world, m])
            s += f64(0.5) * v * v  # pyright: ignore[reportArgumentType]
    costs[world] = s


@wp.kernel
def _accept_best_rollout_kernel(
    num_step_sizes: int,
    nominal: int,
    improve: int,
    min_regularization: f64,
    rollout_costs: wp.array(dtype=f64),  # (W,)
    ok: wp.array(dtype=int),
    costs_before: wp.array(dtype=f64),
    ctrl_history: wp.array3d(dtype=float),  # (T + 1, W, nu)
    feedback: wp.array4d(dtype=float),
    init_ctrls: wp.array3d(dtype=float),
    costs: wp.array(dtype=f64),
    regularization: wp.array(dtype=f64),
    ctrls: wp.array3d(dtype=float),
    gains: wp.array4d(dtype=float),
    taken_rows: wp.array(dtype=int),  # (T + 1) * N: the rollout state each knot takes, knot-major
    taken: wp.array(dtype=int),  # 1 where it takes one
):
    """Accept knot t of each instance's cheapest rollout: of the step sizes, or nominal, of the feedback scales."""
    instance, t = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    batch_size, horizon, num_worlds = costs.shape[0], ctrls.shape[1], rollout_costs.shape[0]
    best, value = int(-1), f64(1e300)  # pyright: ignore[reportArgumentType]
    for a in range(num_step_sizes):
        v = rollout_costs[instance * num_step_sizes + a]
        if wp.isfinite(v) and v < value:
            best, value = a, v
    source, take = int(0), int(0)
    improved = ok[instance] == 1 and best >= 0 and (improve == 0 or value < costs_before[instance])
    if nominal == 1 and best < 0:  # every nominal rollout failed: hold the initial guess, at x0, with no gains
        source, take = instance * num_step_sizes, 1  # the rollouts' first knot: x0
        if t < horizon:
            for j in range(ctrls.shape[2]):
                ctrls[instance, t, j] = init_ctrls[instance, t, j]
        if t == 0:
            costs[instance] = value
    elif nominal == 1 or improved:  # take any best step
        world = instance * num_step_sizes + wp.max(best, 0)
        source, take = t * num_worlds + world, 1
        if t < horizon:
            for j in range(ctrls.shape[2]):
                ctrls[instance, t, j] = ctrl_history[t, world, j]
            if nominal == 0:
                for i in range(gains.shape[2]):
                    for j in range(gains.shape[3]):
                        gains[instance, t, i, j] = feedback[instance, t, i, j]
        if t == 0:
            costs[instance] = value
            if nominal == 0:
                regularization[instance] = wp.max(regularization[instance] / f64(10.0), min_regularization)  # pyright: ignore[reportArgumentType]
    elif t == 0:
        regularization[instance] = regularization[instance] * f64(10.0)  # pyright: ignore[reportArgumentType]
    taken_rows[t * batch_size + instance] = source
    taken[t * batch_size + instance] = take


@wp.kernel
def _set_shifted_init_kernel(
    shift: float,
    ctrls: wp.array3d(dtype=float),
    gains: wp.array4d(dtype=float),
    init_ctrls: wp.array3d(dtype=float),
    init_gains: wp.array4d(dtype=float),
    lo_rows: wp.array(dtype=int),
    hi_rows: wp.array(dtype=int),
    fractions: wp.array(dtype=float),
):
    """Set the initial guess to the last solution `shift` knots on, linear between knots and held at the end."""
    instance, t = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    batch_size, horizon = ctrls.shape[0], ctrls.shape[1]
    at = wp.min(float(t) + shift, float(horizon))
    lo = wp.min(int(at), horizon - 1)
    lo_rows[t * batch_size + instance] = lo * batch_size + instance
    hi_rows[t * batch_size + instance] = (lo + 1) * batch_size + instance
    fractions[t * batch_size + instance] = at - float(lo)
    if t == horizon:
        return
    at = wp.min(float(t) + shift, float(horizon - 1))
    lo = wp.min(int(at), horizon - 2)
    f = at - float(lo)
    for j in range(ctrls.shape[2]):
        init_ctrls[instance, t, j] = (1.0 - f) * ctrls[instance, lo, j] + f * ctrls[instance, lo + 1, j]
        for i in range(gains.shape[3]):
            init_gains[instance, t, j, i] = (1.0 - f) * gains[instance, lo, j, i] + f * gains[instance, lo + 1, j, i]


@wp.kernel
def _set_scaled_residuals_kernel(
    knot: int,  # the rows' clock step, or -1: each row's, from `world_steps`
    final_knot: int,  # the final knot's clock step
    world_steps: wp.array(dtype=wp.vec2i),
    column_scales: wp.array2d(dtype=float),  # (nr, 2) square roots of each column's running and terminal scales
    residuals: wp.array2d(dtype=float),
):
    """Scale each residual by the square root of its term's running scale, or at the final knot its terminal one."""
    row, column = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    t = knot
    if knot < 0:
        t = world_steps[row][1]
    residuals[row, column] = residuals[row, column] * column_scales[column, wp.where(t == final_knot, 1, 0)]


@wp.kernel
def _set_scaled_rows_kernel(scale: wp.array(dtype=float), x: wp.array2d(dtype=float)):
    """Scale row i of `x` by `scale[i]`."""
    i, j = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    x[i, j] = scale[i] * x[i, j]


# --- optimizer --------------------------------------------------------------
class Dynamics:
    """Batched discrete dynamics `x_next = step(x, u)` on `Var` states; subclasses implement `step`.

    Args:
        nu: Controls per step.
        ctrl_range: `(nu, 2)` control bounds, infinite where unbounded; None leaves every control unbounded.
        steps_per_knot: Steps of the terms' clock (`SimContext.CLOCK`) one `step` spans.
    """

    def __init__(self, nu: int, ctrl_range: Optional[np.ndarray] = None, steps_per_knot: int = 1):
        self.nu = nu
        self.ctrl_range = ctrl_range
        self.steps_per_knot = steps_per_knot

    def step(self, x: Var, u: wp.array, x_next: Var, finite_differences: bool = False):
        """Write the states one step after `x` under controls `u` into `x_next`.

        Args:
            x: States, a Var of `batch_size`.
            u: `(batch_size, nu)` float32 controls.
            x_next: The next states, a Var like `x`.
            finite_differences: The rows are finite-difference steps: step them alike (a simulator: one contact order).
        """
        raise NotImplementedError

    def get_context(self, batch_size: int, finite_differences: bool = False) -> Dict[Context, Any]:
        """Return what the terms read of the worlds `step` steps, as they stand after each step.

        Args:
            batch_size: States per step.
            finite_differences: As in `step`.

        Returns:
            The terms' context (`SimContext` for a simulator); empty by default.
        """
        return {}


@dataclass(frozen=True)
class ILQROptimizerConfig(OptimizerConfig):
    """`ILQROptimizer`'s settings; a solve is one CUDA graph, so it never stops early.

    Example:
        >>> ILQROptimizerConfig().max_iter
        1
    """

    max_iter: int = 1
    """Iterations per solve, after the nominal rollout: linearize, backward pass, line search."""
    cuda_graph_mode: Literal["none", "iter", "full"] = "full"
    """"none" launches eagerly; else a CUDA device replays the whole solve as one graph."""
    state_eps: Union[float, Tuple[float, ...]] = 1e-4
    """Finite-difference step of the states: one for every tangent dimension, or one each."""
    ctrl_eps: float = 1e-3
    """Finite-difference step of the controls."""
    feedback_search: bool = True
    """Run the nominal rollout at the best of the step sizes as feedback scale; else at full feedback."""
    improve: bool = False
    """Take a line-search step only if it lowers the cost; else the best step, even 0."""
    feedback_cap: float = 0.0
    """If positive, clip each control's feedback `K dx` to this times half its range."""
    min_regularization: float = 1e-6
    """Control regularization (`Quu + mu I`) every solve starts from and keeps. Larger damps steps."""


class ILQROptimizer:
    """iLQG on a batch of plans at once, every step of it on the device.

    Solve sequence:
    1. `set_start(x0)`: the start states, a Var of `batch_size` instances.
    2. `solve(plan)`: refine `plan` in place: a nominal rollout, then `max_iter` iterations.
    3. Read `states` (knot-major: row `t * batch_size + instance`), `gains` and `costs`.

    Args:
        dynamics: Batched dynamics.
        terms: Residual terms of `VarKey.STATE` and `VarKey.ACTION`; the cost is `0.5 |r|^2` after every step.
        config: Solver settings.
        device: Warp device.
        scales: `(num_terms, 2)` scales of each term's cost: at every knot but the final one, and at the final one.
    """

    # --- setup ---
    def __init__(
        self,
        dynamics: Dynamics,
        terms: Sequence[ResidualTask],
        config: Optional[ILQROptimizerConfig] = None,
        device: Optional[wp_device_type] = None,
        scales: Optional[np.ndarray] = None,
    ):
        self.dynamics = dynamics
        self.terms = list(terms)
        self.scales = np.ones((len(self.terms), 2)) if scales is None else np.asarray(scales, dtype=float)
        self.config = config if config is not None else ILQROptimizerConfig()
        self.device = wp.get_device(device)
        self._x0: Optional[Var] = None
        self._built = False
        self._graph: Optional[wp.Graph] = None

    def set_start(self, x0: Var, step: int = 0):
        """Start the next solves from `x0`, at sim step `step` of the terms' references.

        Args:
            x0: Start states, a Var of `batch_size` instances.
            step: Sim step of the terms' references at the start.
        """
        if self._x0 is None:
            self._x0 = x0.clone()
            self._x0_rows = wp.array(np.arange(x0.batch_size), dtype=int, device=self.device)
            self._clock = wp.array([step, 0, 0], dtype=int, device=self.device)  # `SimContext.CLOCK`
        else:
            x0.gather(self._x0_rows, out=self._x0)
            self._clock[0:1].fill_(step)

    # --- solve ---
    def solve(self, plan: VectorVar, shift: Optional[float] = None) -> Tuple[VectorVar, wp.array]:
        """Refine the controls `plan` from the start states.

        Args:
            plan: `(batch_size, horizon * nu)` controls, refined in place; the initial guess when `shift` is None.
            shift: Knots to shift the last solution by for the initial guess, as a receding horizon.

        Returns:
            `plan` and each instance's float64 cost.
        """
        if self._x0 is None:
            raise ValueError("set_start before solve")
        with wp.ScopedDevice(self.device):
            if not self._built:
                self._build(plan)
            batch_size, horizon, nu = self._ctrls.shape
            ctrls = plan.x.reshape((batch_size, horizon, nu))
            if shift is None:  # open loop from the start
                wp.copy(self._init_ctrls, ctrls)
                self._init_gains.zero_()
                self._x0.gather(self._start_rows, out=self._init_states)
            else:  # the states between knots: lo + fraction (hi - lo), on their manifold
                wp.launch(
                    _set_shifted_init_kernel,
                    (batch_size, horizon + 1),
                    inputs=[float(shift), self._ctrls, self.gains],
                    outputs=[self._init_ctrls, self._init_gains, self._lo_rows, self._hi_rows, self._fractions],
                )
                lo_states = self.states.gather(self._lo_rows, out=self._lo_states)
                hi_states = self.states.gather(self._hi_rows, out=self._hi_states)
                diffs = lo_states.difference(hi_states, out=self._shift_diffs)
                wp.launch(_set_scaled_rows_kernel, diffs.shape, inputs=[self._fractions], outputs=[diffs])
                lo_states.integrate(diffs, out=self._init_states)
            capturing = self.device.is_cuda and wp.get_stream().is_capturing
            if self._graph is None and self.config.cuda_graph_mode != "none" and self.device.is_cuda and not capturing:
                self._solve_body()  # compile, and let the dynamics allocate, before capturing
                with wp.ScopedCapture() as capture:
                    self._solve_body()
                self._graph = capture.graph
            if self._graph is not None and not capturing:
                wp.capture_launch(self._graph)
            else:
                self._solve_body()
            wp.copy(ctrls, self._ctrls)
        return plan, self.costs

    def _solve_body(self):
        """Run the nominal rollout, then per iteration the linearization, backward pass and line search."""
        self._regularization.fill_(self.config.min_regularization)
        wp.copy(self.gains, self._init_gains)  # the last gains stand if a backward pass fails
        init = (self._init_states, self._init_ctrls, self._zero_feedforward, self._init_gains, self._feedback_scales)
        self._compute_rollouts(*init, nominal=True)
        for _ in range(self.config.max_iter):
            self._compute_derivatives()
            self._compute_backward_pass()
            line_search = (self.states, self._ctrls, self._feedforward, self._feedback, self._step_sizes)
            self._compute_rollouts(*line_search, nominal=False)

    # --- build ---
    def _build(self, plan: VectorVar):
        """Allocate every buffer at the first plan's size and resolve the terms on the worlds they read."""
        config, dynamics, x0 = self.config, self.dynamics, self._x0
        assert x0 is not None
        batch_size = x0.batch_size
        state_dim = x0.tangent_dim
        nu = dynamics.nu
        horizon = plan.tangent_dim // nu
        num_knots = horizon + 1  # the last knot has only the terminal cost
        knot_dim = state_dim + nu  # a knot's tangent columns: states, then controls
        num_step_sizes = len(_STEP_SIZES)
        num_worlds = batch_size * num_step_sizes  # every step size of every instance, instance-major
        self._rows_per_knot = knot_dim + 1  # the unstepped knot, then one row per tangent column
        self._knots_per_chunk = min(max(1, _FD_ROWS // self._rows_per_knot), batch_size * num_knots)
        num_rows = self._knots_per_chunk * self._rows_per_knot
        f32, i32 = wp.float32, wp.int32

        def zeros(*shape: int, dtype: Any = f64) -> wp.array:
            return wp.zeros(shape, dtype=dtype)

        def index(rows: Any) -> wp.array:
            return wp.array(np.asarray(rows).ravel(), dtype=i32)

        def states(n: int) -> Var:
            return x0.gather(wp.zeros(n, dtype=i32))

        # settings
        ctrl_range = np.full((nu, 2), [-np.inf, np.inf]) if dynamics.ctrl_range is None else dynamics.ctrl_range
        lower, upper = np.clip(ctrl_range, -_UNBOUNDED, _UNBOUNDED).T
        self._ctrl_lower, self._ctrl_upper = wp.array(lower, dtype=f32), wp.array(upper, dtype=f32)
        self._ctrl_lower64, self._ctrl_upper64 = wp.array(lower, dtype=f64), wp.array(upper, dtype=f64)
        fd_eps = np.r_[np.broadcast_to(config.state_eps, (state_dim,)), np.full(nu, config.ctrl_eps)]
        self._fd_eps = wp.array(fd_eps, dtype=f64)
        self._step_sizes = wp.array(_STEP_SIZES, dtype=f32)
        self._feedback_scales = wp.array(_STEP_SIZES if config.feedback_search else np.ones(num_step_sizes), dtype=f32)

        # the solution, with states knot-major (row t * batch_size + instance)
        self._ctrls = zeros(batch_size, horizon, nu, dtype=f32)
        self.gains = zeros(batch_size, horizon, nu, state_dim, dtype=f32)
        self.costs = zeros(batch_size)
        self.states = states(num_knots * batch_size)

        # the initial guess, and the shift's scratch
        self._init_ctrls = zeros(batch_size, horizon, nu, dtype=f32)
        self._init_gains = zeros(batch_size, horizon, nu, state_dim, dtype=f32)
        self._init_states = states(num_knots * batch_size)
        self._zero_feedforward = zeros(batch_size, num_knots, nu, dtype=f32)
        self._start_rows = index(np.tile(np.arange(batch_size), num_knots))
        self._lo_rows = zeros(num_knots * batch_size, dtype=i32)
        self._hi_rows = zeros(num_knots * batch_size, dtype=i32)
        self._fractions = zeros(num_knots * batch_size, dtype=f32)
        self._lo_states = states(num_knots * batch_size)
        self._hi_states = states(num_knots * batch_size)
        self._shift_diffs = zeros(num_knots * batch_size, state_dim, dtype=f32)

        # rollouts, knot-major (row t * num_worlds + world)
        self._rollout_states = states((horizon + 2) * num_worlds)
        self._rollout_knots = [self._rollout_states.view(num_worlds, t * num_worlds) for t in range(horizon + 2)]
        self._rollout_refs = states(num_knots * num_worlds)
        self._rollout_ref_knots = [self._rollout_refs.view(num_worlds, t * num_worlds) for t in range(num_knots)]
        worlds = np.arange(num_worlds)
        self._rollout_ref_rows = index([t * batch_size + worlds // num_step_sizes for t in range(num_knots)])
        self._rollout_start_rows = index(worlds // num_step_sizes)
        self._rollout_diffs = zeros(num_worlds, state_dim, dtype=f32)
        self._rollout_ctrls = zeros(num_worlds, nu, dtype=f32)
        self._rollout_ctrl_history = zeros(num_knots, num_worlds, nu, dtype=f32)
        self._rollout_costs = zeros(num_worlds)
        self._costs_before = zeros(batch_size)
        self._taken_rows = zeros(num_knots * batch_size, dtype=i32)
        self._taken = zeros(num_knots * batch_size, dtype=i32)
        self._taken_states = states(num_knots * batch_size)

        # finite differences, a chunk of knots at a time: row r steps column r % rows_per_knot - 1 of its knot
        rows = np.arange(num_rows)
        self._fd_state_rows, self._fd_chunk_steps = [], []  # each row's knot state, and its (instance, step)
        for knot0 in range(0, batch_size * num_knots, self._knots_per_chunk):
            knots = np.minimum(knot0 + rows // self._rows_per_knot, batch_size * num_knots - 1)  # spare rows: the last
            self._fd_state_rows.append(index(knots % num_knots * batch_size + knots // num_knots))
            steps = np.c_[knots // num_knots, knots % num_knots * dynamics.steps_per_knot].astype(np.int32)
            self._fd_chunk_steps.append(wp.array(steps, dtype=wp.vec2i))
        self._fd_unstepped_rows = index(rows // self._rows_per_knot * self._rows_per_knot)
        self._fd_states = states(num_rows)
        self._fd_next_states = states(num_rows)
        self._fd_unstepped_states = states(num_rows)
        self._fd_state_steps = zeros(num_rows, state_dim, dtype=f32)
        self._fd_ctrls = zeros(num_rows, nu, dtype=f32)
        self._fd_next_diffs = zeros(num_rows, state_dim, dtype=f32)
        self._fd_step_sizes = zeros(num_rows)
        self._fd_world_steps = zeros(num_rows, dtype=wp.vec2i)

        # the terms, resolved on the rollouts' worlds and on the finite-difference rows
        self._rollout_vars = [
            VarValues({VarKey.STATE: x, VarKey.ACTION: VectorVar(self._rollout_ctrls)})
            for x in self._rollout_knots[:num_knots]
        ]
        self._fd_vars = VarValues({VarKey.STATE: self._fd_states, VarKey.ACTION: VectorVar(self._fd_ctrls)})
        shared = {SimContext.CLOCK: self._clock, SimContext.NUM_INSTANCES: batch_size}
        rollout_context = {**dynamics.get_context(num_worlds), **shared, SimContext.WORLDS_PER_INSTANCE: num_step_sizes}
        fd_context = {**dynamics.get_context(num_rows, True), **shared, SimContext.WORLDS_PER_INSTANCE: 1}
        fd_context[SimContext.WORLD_STEPS] = self._fd_world_steps
        self._rollout_terms = [copy.copy(term) for term in self.terms]
        self._fd_terms = [copy.copy(term) for term in self.terms]
        for term in self._rollout_terms:
            term.resolve(self._rollout_vars[0], rollout_context)
        for term in self._fd_terms:
            term.resolve(self._fd_vars, fd_context)
        residual_dims = [term.residual_dim for term in self._rollout_terms]
        residual_dim = sum(residual_dims)
        self._column_scales = wp.array(np.repeat(np.sqrt(self.scales), residual_dims, axis=0), dtype=f32)
        self._rollout_residuals = zeros(num_knots * num_worlds, residual_dim, dtype=f32)
        knot_residuals = [self._rollout_residuals[t * num_worlds : (t + 1) * num_worlds] for t in range(num_knots)]
        self._rollout_knot_residuals = cast(List[wp.array], knot_residuals)
        self._fd_residuals = zeros(num_rows, residual_dim, dtype=f32)

        # derivatives at every knot
        self.dynamics_jacobians = zeros(batch_size, horizon, state_dim, knot_dim)
        self._residual_jacobians = zeros(batch_size, num_knots, knot_dim, residual_dim)
        self._cost_grad = zeros(batch_size, num_knots, knot_dim)
        self._cost_hess = zeros(batch_size, num_knots, knot_dim, knot_dim)

        # backward pass, with the box QP's scratch, and the feedforward steps and feedback gains it finds
        self._regularization = zeros(batch_size)
        self._backward_ok = zeros(batch_size, dtype=int)
        self._value_grad = zeros(batch_size, state_dim)
        self._value_hess = zeros(batch_size, state_dim, state_dim)
        self._q_grad = zeros(batch_size, knot_dim)
        self._q_hess = zeros(batch_size, knot_dim, knot_dim)
        self._value_hess_ab = zeros(batch_size, state_dim, knot_dim)
        self._q_gain = zeros(batch_size, nu, state_dim)
        self._gain_columns = zeros(batch_size, state_dim, nu)
        self._qp_factors = zeros(batch_size, nu, nu)
        self._qp_free = zeros(batch_size, nu, dtype=int)
        self._qp_steps = zeros(batch_size, nu)
        self._qp_directions = zeros(batch_size, nu)
        self._qp_trials = zeros(batch_size, nu)
        self._qp_grads = zeros(batch_size, nu)
        self._qp_quus = zeros(batch_size, nu, nu)
        self._qp_qus = zeros(batch_size, nu)
        self._feedforward = zeros(batch_size, num_knots, nu, dtype=f32)
        self._feedback = zeros(batch_size, horizon, nu, state_dim, dtype=f32)
        self._built = True

    # --- rollouts ---
    def _compute_rollouts(
        self,
        ref_states: Var,
        ctrls: wp.array,
        feedforward: wp.array,
        feedback: wp.array,
        scales: wp.array,
        nominal: bool,
    ):
        """Roll out each instance at every scale of its feedforward (nominal: feedback) and keep the cheapest."""
        x0, config = self._x0, self.config
        assert x0 is not None
        batch_size, horizon, _ = self._ctrls.shape
        num_worlds, num_step_sizes = self._rollout_ctrls.shape[0], self._step_sizes.shape[0]
        x0.gather(self._rollout_start_rows, out=self._rollout_knots[0])
        ref_states.gather(self._rollout_ref_rows, out=self._rollout_refs)  # each world's reference, every knot
        for t in range(horizon + 1):
            self._rollout_ref_knots[t].difference(self._rollout_knots[t], out=self._rollout_diffs)
            wp.launch(
                _compute_rollout_ctrls_kernel,
                num_worlds,
                inputs=[
                    t,
                    int(t == horizon),
                    num_step_sizes,
                    int(nominal),
                    config.feedback_cap,
                    ctrls,
                    feedforward,
                    feedback,
                    scales,
                    self._ctrl_lower,
                    self._ctrl_upper,
                    self._rollout_diffs,
                ],
                outputs=[self._rollout_ctrls, self._rollout_ctrl_history],
            )
            self.dynamics.step(self._rollout_knots[t], self._rollout_ctrls, self._rollout_knots[t + 1])
            step = t * self.dynamics.steps_per_knot  # the terms' clock
            self._clock[1:2].fill_(step)
            self._compute_residuals(self._rollout_terms, self._rollout_vars[t], self._rollout_knot_residuals[t], step)
        wp.launch(
            _compute_rollout_costs_kernel,
            num_worlds,
            inputs=[horizon + 1, self._rollout_residuals],
            outputs=[self._rollout_costs],
        )
        wp.copy(self._costs_before, self.costs)  # every knot compares against the cost before this acceptance
        wp.launch(
            _accept_best_rollout_kernel,
            (batch_size, horizon + 1),
            inputs=[
                num_step_sizes,
                int(nominal),
                int(config.improve),
                config.min_regularization,
                self._rollout_costs,
                self._backward_ok,
                self._costs_before,
                self._rollout_ctrl_history,
                self._feedback,
                self._init_ctrls,
            ],
            outputs=[self.costs, self._regularization, self._ctrls, self.gains, self._taken_rows, self._taken],
        )
        self._rollout_states.gather(self._taken_rows, out=self._taken_states)
        self.states.accept(self._taken, self._taken_states)

    # --- derivatives ---
    def _compute_derivatives(self):
        """Compute every knot's dynamics and residual Jacobians by forward differences, and its Gauss-Newton costs."""
        batch_size, horizon, _ = self._ctrls.shape
        total_knots = batch_size * (horizon + 1)
        knot_dim, num_rows = self._cost_grad.shape[2], self._fd_step_sizes.shape[0]
        for chunk, (state_rows, chunk_steps) in enumerate(zip(self._fd_state_rows, self._fd_chunk_steps)):
            knot0 = chunk * self._knots_per_chunk
            chunk_knots = min(self._knots_per_chunk, total_knots - knot0)
            wp.launch(
                _set_fd_rows_kernel,
                num_rows,
                inputs=[knot0, self._rows_per_knot, total_knots, self._ctrls, self._fd_eps, self._ctrl_upper],
                outputs=[self._fd_state_steps, self._fd_ctrls, self._fd_step_sizes],
            )
            self.states.gather(state_rows, out=self._fd_states)
            self._fd_states.integrate(self._fd_state_steps, out=self._fd_states)
            self.dynamics.step(self._fd_states, self._fd_ctrls, self._fd_next_states, finite_differences=True)
            wp.copy(self._fd_world_steps, chunk_steps)
            self._compute_residuals(self._fd_terms, self._fd_vars, self._fd_residuals, -1)
            self._fd_next_states.gather(self._fd_unstepped_rows, out=self._fd_unstepped_states)
            self._fd_unstepped_states.difference(self._fd_next_states, out=self._fd_next_diffs)
            wp.launch(
                _compute_fd_jacobians_kernel,
                (chunk_knots, knot_dim),
                inputs=[knot0, self._rows_per_knot, self._fd_next_diffs, self._fd_residuals, self._fd_step_sizes],
                outputs=[self.dynamics_jacobians, self._residual_jacobians],
            )
            wp.launch(
                _compute_gauss_newton_kernel,
                (chunk_knots, knot_dim),
                inputs=[knot0, self._rows_per_knot, self._fd_residuals, self._residual_jacobians],
                outputs=[self._cost_grad, self._cost_hess],
            )

    def _compute_residuals(self, terms: List[ResidualTask], var: VarValues, out: wp.array, knot: int):
        """Compute every term's scaled residuals of `var` at clock step `knot` (-1: each row's), side by side in `out`."""
        offset = 0
        for term in terms:
            term.precompute(var)
            term.compute_weighted_residual(var, out_residual=out, row_offset=offset)
            offset += term.residual_dim
        inputs = [knot, self._ctrls.shape[1] * self.dynamics.steps_per_knot, self._fd_world_steps, self._column_scales]
        wp.launch(_set_scaled_residuals_kernel, out.shape, inputs=inputs, outputs=[out])

    # --- backward pass ---
    def _compute_backward_pass(self):
        """Compute every knot's feedforward step and feedback gains, from the last knot back."""
        batch_size, horizon, nu = self._ctrls.shape
        state_dim, knot_dim = self._value_grad.shape[1], self._q_grad.shape[1]
        # the final knot's value is its cost
        final_grad = self._cost_grad[:, horizon, :state_dim]
        final_hess = self._cost_hess[:, horizon, :state_dim, :state_dim]
        wp.copy(self._value_grad, final_grad)  # pyright: ignore[reportArgumentType]
        wp.copy(self._value_hess, final_hess)  # pyright: ignore[reportArgumentType]
        self._feedforward[:, horizon].zero_()
        self._backward_ok.fill_(1)
        for t in reversed(range(horizon)):
            wp.launch(
                _compute_value_hess_ab_kernel,
                (batch_size, state_dim, knot_dim),
                inputs=[t, self.dynamics_jacobians, self._value_hess],
                outputs=[self._value_hess_ab],
            )
            wp.launch(
                _compute_q_function_kernel,
                (batch_size, knot_dim, knot_dim),
                inputs=[
                    t,
                    self.dynamics_jacobians,
                    self._cost_grad,
                    self._cost_hess,
                    self._value_grad,
                    self._value_hess_ab,
                ],
                outputs=[self._q_grad, self._q_hess],
            )
            wp.launch(
                _solve_box_qp_kernel,
                batch_size,
                inputs=[
                    t,
                    self._q_grad,
                    self._q_hess,
                    self._ctrls,
                    self._ctrl_lower64,
                    self._ctrl_upper64,
                    self._regularization,
                ],
                outputs=[
                    self._backward_ok,
                    self._qp_factors,
                    self._qp_free,
                    self._qp_steps,
                    self._qp_directions,
                    self._qp_trials,
                    self._qp_grads,
                    self._qp_quus,
                    self._qp_qus,
                    self._feedforward,
                ],
            )
            wp.launch(
                _compute_gains_kernel,
                (batch_size, state_dim),
                inputs=[t, self._q_hess, self._backward_ok, self._qp_factors, self._qp_free],
                outputs=[self._gain_columns, self._feedback],
            )
            wp.launch(
                _compute_q_gain_kernel,
                (batch_size, nu, state_dim),
                inputs=[t, self._q_hess, self._feedback],
                outputs=[self._q_gain],
            )
            wp.launch(
                _compute_value_function_kernel,
                (batch_size, state_dim, state_dim),
                inputs=[t, self._q_grad, self._q_hess, self._q_gain, self._feedforward, self._feedback],
                outputs=[self._value_grad, self._value_hess],
            )
