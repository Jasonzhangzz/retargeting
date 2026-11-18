"""Small active-set box QP recovered from the pinned SDK's offline finger IK.

Pure Python/NumPy implementation; no SDK or numerical optimizer is called.
The native routine at ELF VA 0xdab511 uses at most five variables, fixed 5x5
row-major Hessian storage, an explicit Cholesky solve and at most 4*n+8 passes.
This reproduces its bounded-iteration behavior, including singular-H and
inconsistent-bound fallbacks, rather than guaranteeing a generic QP optimum.
"""
import math
import numpy as np

BOUND_TOL = 1e-9
PIVOT_TOL = 1e-12
DIRECTION_TOL = 1e-12


def solve_box_qp(H, g, lower, upper, return_trace=False):
    """Solve ``0.5*delta.T@H@delta + g.T@delta`` with the SDK's box-QP path.

    Accept finite vectors of length 0..5 and H of shape (n,n) or (5,5).
    For padded H only its leading n rows/columns are used. H normally is
    symmetric positive definite, but it is not symmetrized or regularized.
    Default return: float64 delta. With return_trace=True return a dictionary
    with delta, iterations, active_set, status and events. Active statuses are
    0=free, 1=lower, 2=upper, 3=fixed/inconsistent. Traces are diagnostic only.
    """
    g = np.asarray(g, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    if g.ndim != 1 or not 0 <= len(g) <= 5:
        raise ValueError('g must be a vector of length 0..5')
    n = len(g)
    if lower.shape != (n,) or upper.shape != (n,):
        raise ValueError('lower and upper must have the same shape as g')
    if H.shape not in ((n, n), (5, 5)):
        raise ValueError('H must have shape (n,n) or (5,5)')
    H = H[:n, :n]
    if not all(np.isfinite(a).all() for a in (H, g, lower, upper)):
        raise ValueError('box-QP inputs must be finite')

    # Python scalars ensure each product/addition rounds separately, with no
    # BLAS reduction, FMA, matrix reordering, or NumPy scalar promotion.
    h = H.tolist()
    gg, lo, hi = g.tolist(), lower.tolist(), upper.tolist()
    delta = [0.0] * n
    active = [0] * n
    events = []
    for i in range(n):
        # SSE maxsd/minsd choose the second operand on equality, including
        # signed zero: maxsd(lower,+0), then minsd(upper,that result).
        clipped = lo[i] if lo[i] > 0.0 else 0.0
        clipped = hi[i] if hi[i] < clipped else clipped
        if lo[i] > hi[i] + BOUND_TOL:
            delta[i], active[i] = clipped, 3
        elif hi[i] - lo[i] <= BOUND_TOL:
            delta[i], active[i] = (lo[i] + hi[i]) * 0.5, 3
        else:
            delta[i] = clipped
    if return_trace:
        events.append({'event': 'initialize', 'delta': delta.copy(), 'active_set': active.copy()})

    status = 'iteration_limit'
    iterations = 0
    for iteration in range(4*n + 8):
        iterations = iteration + 1
        free = [i for i in range(n) if active[i] == 0]
        m = len(free)
        factor_ok = True
        if m:
            rhs = [0.0] * m
            reduced = [[0.0]*m for _ in range(m)]
            for a, i in enumerate(free):
                value = -gg[i]
                for j in range(n):
                    if active[j] != 0:
                        value -= h[i][j] * delta[j]
                rhs[a] = value
                for b, j in enumerate(free):
                    reduced[a][b] = h[i][j]

            chol = [[0.0]*m for _ in range(m)]
            for a in range(m):
                # Native starts Cholesky sums from -0.0 (VA 0x2445c8).
                total = -0.0
                for k in range(a):
                    total += chol[a][k] * chol[a][k]
                pivot = reduced[a][a] - total
                if pivot <= PIVOT_TOL:
                    factor_ok = False
                    if return_trace:
                        events.append({'event': 'cholesky_failed', 'iteration': iterations,
                                       'free_index': a, 'variable': free[a], 'pivot': pivot})
                    break
                chol[a][a] = math.sqrt(pivot)
                for b in range(a+1, m):
                    total = -0.0
                    for k in range(a):
                        total += chol[b][k] * chol[a][k]
                    chol[b][a] = (reduced[b][a] - total) / chol[a][a]

            if factor_ok:
                intermediate = [0.0] * m
                for a in range(m):
                    total = 0.0
                    for k in range(a):
                        total += chol[a][k] * intermediate[k]
                    intermediate[a] = (rhs[a] - total) / chol[a][a]
                candidate = [0.0] * m
                for a in range(m-1, -1, -1):
                    total = 0.0
                    for k in range(a+1, m):
                        total += chol[k][a] * candidate[k]
                    candidate[a] = (intermediate[a] - total) / chol[a][a]

                fraction = 1.0
                blocker = -1
                blocker_status = 0
                for a, i in enumerate(free):
                    direction = candidate[a] - delta[i]
                    if direction > DIRECTION_TOL:
                        trial_fraction = (hi[i] - delta[i]) / direction
                        side = 2
                    elif direction < -DIRECTION_TOL:
                        trial_fraction = (lo[i] - delta[i]) / direction
                        side = 1
                    else:
                        continue
                    # Strict comparison: first blocking variable wins ties;
                    # a boundary reached at exactly fraction=1 stays free.
                    if fraction > trial_fraction:
                        fraction, blocker, blocker_status = trial_fraction, i, side
                fraction = 0.0 if fraction < 0.0 else fraction
                for a, i in enumerate(free):
                    delta[i] = (candidate[a] - delta[i]) * fraction + delta[i]
                if blocker != -1:
                    delta[blocker] = lo[blocker] if blocker_status == 1 else hi[blocker]
                    active[blocker] = blocker_status
                    if return_trace:
                        events.append({'event': 'block', 'iteration': iterations,
                                       'variable': blocker, 'side': blocker_status,
                                       'fraction': fraction, 'delta': delta.copy()})
                    continue
                if return_trace:
                    events.append({'event': 'free_solution', 'iteration': iterations,
                                   'free': free, 'delta': delta.copy()})

        # Also reached on Cholesky failure: inspect active KKT signs using
        # the unchanged feasible delta; do not substitute a generic solver.
        gradient = [0.0] * n
        scale = 1.0
        for i in range(n):
            value = gg[i]
            for j in range(n):
                value += h[i][j] * delta[j]
            gradient[i] = value
            scale = max(scale, abs(value))
        worst = scale * BOUND_TOL
        released = -1
        for i in range(n):
            if active[i] == 1:
                violation = -gradient[i]
            elif active[i] == 2:
                violation = gradient[i]
            else:
                continue
            if violation > worst:
                worst, released = violation, i
        if released == -1:
            status = 'stationary_active_set' if factor_ok else 'cholesky_failed'
            break
        active[released] = 0
        if return_trace:
            events.append({'event': 'release', 'iteration': iterations, 'variable': released,
                           'violation': worst, 'gradient': gradient.copy()})

    result = np.asarray(delta, dtype=np.float64)
    if return_trace:
        return {'delta': result, 'iterations': iterations,
                'active_set': active.copy(), 'status': status, 'events': events}
    return result
