"""Fused CUDA fixed-projection response, one thread per candidate/wrench.

Newton, backtracking, finite-state selection and exact forward sensitivities
execute on the device. Torch remains the differentiable CPU/reference path.
"""
from functools import lru_cache

import torch
import warp as wp

from .comfree_equilibrium import EquilibriumResponse

M6 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
M612 = wp.types.matrix(shape=(6, 12), dtype=wp.float64)
V6 = wp.types.vector(length=6, dtype=wp.float64)


@wp.func
def _inverse6(matrix: M6):
    aug = M612(wp.float64(0.0))
    for i in range(6):
        for j in range(6):
            aug[i, j] = matrix[i, j]
        aug[i, i + 6] = wp.float64(1.0)
    valid = int(1)
    for k in range(6):
        pivot = k
        largest = wp.abs(aug[k, k])
        for i in range(k + 1, 6):
            value = wp.abs(aug[i, k])
            if value > largest:
                largest = value
                pivot = i
        if largest < wp.float64(1.e-30) or not wp.isfinite(largest):
            valid = 0
        else:
            if pivot != k:
                for j in range(12):
                    value = aug[k, j]
                    aug[k, j] = aug[pivot, j]
                    aug[pivot, j] = value
            reciprocal = wp.float64(1.0) / aug[k, k]
            for j in range(12):
                aug[k, j] = aug[k, j] * reciprocal
            for i in range(6):
                if i != k:
                    value = aug[i, k]
                    for j in range(12):
                        aug[i, j] = aug[i, j] - value * aug[k, j]
    inverse = M6(wp.float64(0.0))
    for i in range(6):
        for j in range(6):
            if valid != 0:
                inverse[i, j] = aug[i, j + 6]
    return inverse, valid


@lru_cache(maxsize=64)
def _kernel(contacts, ray_count, jacobian):
    n = wp.constant(contacts)
    rays_per_contact = wp.constant(ray_count)
    inverse_ray_count = wp.constant(1. / ray_count)
    rows = wp.constant(contacts * ray_count)
    derivatives = wp.constant(jacobian)
    U = wp.types.matrix(shape=(6, contacts), dtype=wp.float64)
    F = wp.types.vector(length=contacts * ray_count, dtype=wp.float64)

    @wp.kernel(module="unique", enable_backward=False)
    def response(alpha: wp.array2d(dtype=wp.float64), A: wp.array3d(dtype=wp.float64),
                 drag: wp.array2d(dtype=wp.float64), loads: wp.array2d(dtype=wp.float64),
                 iterations: int, search_steps: int, damping: wp.float64,
                 factor: wp.float64, balance_out: wp.array3d(dtype=wp.float64),
                 force_out: wp.array3d(dtype=wp.float64), velocity_out: wp.array3d(dtype=wp.float64),
                 balance_jac: wp.array4d(dtype=wp.float64), force_jac: wp.array4d(dtype=wp.float64),
                 finite_out: wp.array2d(dtype=wp.int32)):
        b, s = wp.tid()
        u = V6(wp.float64(0.0))
        sensitivity = U(wp.float64(0.0))
        best_u = u
        best_sensitivity = sensitivity
        best_score = wp.float64(1.e300)
        best_effort = wp.float64(1.e300)
        full_h = M6(wp.float64(0.0))
        for r in range(rows):
            for i in range(6):
                for j in range(6):
                    full_h[i, j] = full_h[i, j] + A[b, r, i] * drag[b, r] * A[b, r, j]
        trace = wp.float64(0.0)
        for i in range(6):
            trace = trace + full_h[i, i]
        for i in range(6):
            full_h[i, i] = full_h[i, i] + damping * wp.max(trace, wp.float64(1.0))
        full_inverse, full_valid = _inverse6(full_h)
        for iteration in range(iterations + 1):
            force = F(wp.float64(0.0))
            balance = V6(wp.float64(0.0))
            energy = wp.float64(0.0)
            for i in range(6):
                balance[i] = loads[s, i]
                energy = energy - loads[s, i] * u[i]
            for r in range(rows):
                dot = wp.float64(0.0)
                for i in range(6):
                    dot = dot + A[b, r, i] * u[i]
                value = wp.max(wp.float64(0.0), alpha[b, r // rays_per_contact] * wp.float64(inverse_ray_count) - drag[b, r] * dot)
                force[r] = value
                energy = energy + wp.float64(0.5) * value * value / drag[b, r]
                for i in range(6):
                    balance[i] = balance[i] + A[b, r, i] * value
            score = wp.dot(balance, balance)
            effort = wp.float64(0.0)
            for contact in range(n):
                normal = wp.float64(0.0)
                for r in range(rays_per_contact):
                    normal = normal + force[contact * rays_per_contact + r]
                effort = effort + normal * normal
            roundoff = wp.float64(1.4210854715202004e-14) * wp.max(wp.float64(1.0), wp.max(wp.abs(score), wp.abs(best_score)))
            if wp.isfinite(score) and wp.isfinite(effort):
                if score < best_score - roundoff or (wp.abs(score - best_score) <= roundoff and effort < best_effort):
                    best_score = score
                    best_effort = effort
                    best_u = u
                    best_sensitivity = sensitivity
            if iteration == iterations:
                break
            h = M6(wp.float64(0.0))
            direct = U(wp.float64(0.0))
            for r in range(rows):
                if force[r] > wp.float64(0.0):
                    for i in range(6):
                        if wp.static(derivatives):
                            direct[i, r // rays_per_contact] = direct[i, r // rays_per_contact] + A[b, r, i] * wp.float64(inverse_ray_count)
                        for j in range(6):
                            h[i, j] = h[i, j] + A[b, r, i] * drag[b, r] * A[b, r, j]
            rhs = direct
            if wp.static(derivatives):
                rhs = direct - h * sensitivity
            trace = wp.float64(0.0)
            for i in range(6):
                trace = trace + h[i, i]
            for i in range(6):
                h[i, i] = h[i, i] + damping * wp.max(trace, wp.float64(1.0))
            inverse, newton_valid = _inverse6(h)
            delta_newton = inverse * balance
            delta_full = full_inverse * balance
            derivative_newton = U(wp.float64(0.0))
            derivative_full = U(wp.float64(0.0))
            if wp.static(derivatives):
                derivative_newton = inverse * rhs
                derivative_full = full_inverse * rhs
            best_energy = wp.float64(1.e300)
            chosen_kind = int(-1)
            chosen_step = wp.float64(0.0)
            step = wp.float64(1.0)
            for candidate in range(search_steps + 1):
                delta = delta_newton
                numerical = newton_valid
                if candidate == search_steps:
                    delta = delta_full
                    numerical = full_valid
                    step = wp.float64(1.0)
                length = wp.max(wp.sqrt(wp.dot(delta, delta)), wp.float64(1.e-30))
                scale = wp.min(wp.float64(1.0), wp.float64(1.e6) / length)
                direction = delta * (scale * step)
                trial_u = u + direction
                trial_energy = wp.float64(0.0)
                for i in range(6):
                    trial_energy = trial_energy - loads[s, i] * trial_u[i]
                for r in range(rows):
                    dot = wp.float64(0.0)
                    for i in range(6):
                        dot = dot + A[b, r, i] * trial_u[i]
                    value = wp.max(wp.float64(0.0), alpha[b, r // rays_per_contact] * wp.float64(inverse_ray_count) - drag[b, r] * dot)
                    trial_energy = trial_energy + wp.float64(0.5) * value * value / drag[b, r]
                tolerance = wp.float64(3.552713678800501e-15) * wp.max(wp.float64(1.0), wp.abs(energy))
                if numerical != 0 and wp.isfinite(trial_energy) and trial_energy <= energy - wp.float64(1.e-4) * wp.dot(balance, direction) + tolerance:
                    if trial_energy < best_energy:
                        best_energy = trial_energy
                        chosen_kind = int(candidate == search_steps)
                        chosen_step = step
                step = step * factor
            if chosen_kind >= 0:
                delta = delta_newton
                derivative = derivative_newton
                if chosen_kind == 1:
                    delta = delta_full
                    derivative = derivative_full
                length = wp.max(wp.sqrt(wp.dot(delta, delta)), wp.float64(1.e-30))
                scale = wp.min(wp.float64(1.0), wp.float64(1.e6) / length)
                if wp.static(derivatives):
                    if scale < wp.float64(1.0):
                        for contact in range(n):
                            radial = wp.float64(0.0)
                            for i in range(6):
                                radial = radial + delta[i] * derivative[i, contact]
                            radial = radial / (length * length)
                            for i in range(6):
                                derivative[i, contact] = scale * (derivative[i, contact] - delta[i] * radial)
                    else:
                        derivative = derivative * scale
                    sensitivity = sensitivity + derivative * chosen_step
                u = u + delta * (scale * chosen_step)
        finite = int(wp.isfinite(best_score))
        balance = V6(wp.float64(0.0))
        for i in range(6):
            velocity_out[b, s, i] = best_u[i]
            balance[i] = loads[s, i]
        for r in range(rows):
            dot = wp.float64(0.0)
            for i in range(6):
                dot = dot + A[b, r, i] * best_u[i]
            value = wp.max(wp.float64(0.0), alpha[b, r // rays_per_contact] * wp.float64(inverse_ray_count) - drag[b, r] * dot)
            force_out[b, s, r] = value
            if not wp.isfinite(value):
                finite = 0
            for i in range(6):
                balance[i] = balance[i] + A[b, r, i] * value
            if wp.static(derivatives):
                for contact in range(n):
                    row_derivative = wp.float64(0.0)
                    if value > wp.float64(0.0):
                        if contact == r // rays_per_contact:
                            row_derivative = wp.float64(inverse_ray_count)
                        for i in range(6):
                            row_derivative = row_derivative - drag[b, r] * A[b, r, i] * best_sensitivity[i, contact]
                    force_jac[b, s, r, contact] = row_derivative
        for i in range(6):
            balance_out[b, s, i] = balance[i]
            if wp.static(derivatives):
                for contact in range(n):
                    row_derivative = wp.float64(0.0)
                    for r in range(rows):
                        row_derivative = row_derivative + A[b, r, i] * force_jac[b, s, r, contact]
                    balance_jac[b, s, i, contact] = row_derivative
                    if not wp.isfinite(row_derivative):
                        finite = 0
        finite_out[b, s] = finite
    return response


def equilibrium_response_cuda(preloads, rays, row_drag, loads, *, iterations=16,
                              line_search_steps=8, damping=1.e-10, tolerance=1.e-6,
                              jacobian=False, backtracking_factor=.25):
    if not preloads.is_cuda:
        raise ValueError("Fused ComFree response requires CUDA tensors")
    wp.init()
    batch, contacts = preloads.shape
    rows = rays.shape[1]
    if rays.shape != (batch, rows, 6) or rows % contacts or row_drag.shape != (batch, rows):
        raise ValueError("Invalid fixed contact ray dimensions")
    valid = (torch.isfinite(preloads).all(1) & (preloads >= 0).all(1)
             & torch.isfinite(rays).all((1, 2)) & torch.isfinite(row_drag).all(1)
             & (row_drag > 0).all(1) & torch.isfinite(loads).all())
    A = torch.where(valid[:, None, None], rays, 0.).double().contiguous()
    c = torch.where(valid[:, None], row_drag, 1.).double().contiguous()
    alpha = torch.where(valid[:, None], preloads, 0.).double().contiguous()
    w = torch.where(torch.isfinite(loads), loads, 0.).double().contiguous()
    scenarios = len(loads)
    velocity = torch.empty((batch, scenarios, 6), dtype=torch.float64, device=preloads.device)
    balance = torch.empty_like(velocity)
    force = torch.empty((batch, scenarios, rows), dtype=torch.float64, device=preloads.device)
    jb = torch.empty((batch, scenarios, 6, contacts) if jacobian else (1, 1, 1, 1), dtype=torch.float64, device=preloads.device)
    jf = torch.empty((batch, scenarios, rows, contacts) if jacobian else (1, 1, 1, 1), dtype=torch.float64, device=preloads.device)
    finite = torch.empty((batch, scenarios), dtype=torch.int32, device=preloads.device)
    # All inputs/outputs belong to the current Torch stream. Importing a
    # dependency from Warp's unrelated default stream would invalidate capture.
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(preloads.device)), sync_enter=False):
        wp.launch(_kernel(contacts, rows // contacts, jacobian), dim=(batch, scenarios), inputs=[
            wp.from_torch(alpha), wp.from_torch(A), wp.from_torch(c), wp.from_torch(w),
            iterations, line_search_steps, wp.float64(damping), wp.float64(backtracking_factor),
            wp.from_torch(balance), wp.from_torch(force), wp.from_torch(velocity),
            wp.from_torch(jb), wp.from_torch(jf), wp.from_torch(finite)], block_dim=64)
    valid = valid & (finite != 0).all(1)
    converged = valid[:, None] & (balance.norm(dim=-1) <= tolerance * w.abs().amax().clamp_min(1.))
    dtype = preloads.dtype
    return EquilibriumResponse(balance.to(dtype), force.to(dtype), velocity.to(dtype),
        jb.to(dtype) if jacobian else None, jf.to(dtype) if jacobian else None, converged, valid)
