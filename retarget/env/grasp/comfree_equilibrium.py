"""Fixed-gap ComFree equilibrium with a physically derived 6D response.

Only contact preloads are independent inputs. The response velocity is solved
from force balance and is never a grasp optimization variable. A fixed-budget
semismooth Newton solve preserves the actual residual of infeasible samples.
"""

from dataclasses import dataclass
import math

import torch


@dataclass
class EquilibriumResponse:
    balance: torch.Tensor
    row_forces: torch.Tensor
    velocity: torch.Tensor
    balance_jacobian: torch.Tensor | None
    row_force_jacobian: torch.Tensor | None
    converged: torch.Tensor
    valid: torch.Tensor


def equilibrium_response(preloads, rays, row_drag, loads, *, iterations=16,
                         line_search_steps=8, damping=1.e-10, tolerance=1.e-6,
                         jacobian=False, backtracking_factor=.25, rays_per_contact=None):
    """Solve ``lambda=relu(alpha/R-c*A*u), A.T*lambda+w=0``.

    Inputs have shapes ``(B,n)``, ``(B,R*n,6)``, ``(B,R*n)`` and ``(S,6)``.
    The fixed ray count R is inferred when omitted; an explicit value must
    match the input exactly. Four-ray point contacts remain fully compatible,
    while six-ray soft-finger pyramids include both signed torsional rays.
    All line-search candidates are evaluated together. ``jacobian=True`` uses
    exact forward sensitivities of the executed piecewise smooth iterations,
    including selected backtracking steps and clipped Newton increments. It
    therefore retains the residual derivative of nonconverged samples too.
    The unclipped implicit derivative at convergence is the same limit:
    ``du/dalpha = (A.T*D*C*A)^-1*A.T*D*S``.

    ``velocity`` is a virtual fixed-gap response, not a simulated physical hold
    velocity. Its magnitude is not used to certify force closure.

    Each scenario returns the finite iterate with the smallest actual squared
    balance residual, including the initial and final states. The convex dual
    energy may be unbounded for infeasible wrench cones; decreasing that energy
    must not arbitrarily inflate their returned closure cost. Sensitivities
    follow the same selected state, without replacing its residual by zero.
    Residuals equal within Float64 roundoff prefer smaller actual normal-force
    effort, so negligible numerical differences cannot select huge self-stress.
    """
    if preloads.ndim != 2 or min(preloads.shape) == 0:
        raise ValueError("Preloads must have nonempty shape (batch, contacts)")
    batch, contacts = preloads.shape
    if (rays.ndim != 3 or rays.shape[0] != batch or rays.shape[2] != 6
            or rays.shape[1] == 0 or rays.shape[1] % contacts != 0):
        raise ValueError("Each contact must have the same positive number of fixed six-dimensional rays")
    inferred_rays = rays.shape[1] // contacts
    if rays_per_contact is None:
        rays_per_contact = inferred_rays
    if (isinstance(rays_per_contact, bool) or not isinstance(rays_per_contact, int)
            or rays_per_contact < 1 or rays_per_contact != inferred_rays):
        raise ValueError("rays_per_contact must be a positive integer matching the input rows")
    if row_drag.shape != rays.shape[:2]:
        raise ValueError("Row drag must have one value for each fixed ray")
    if loads.ndim != 2 or loads.shape[-1] != 6 or not len(loads):
        raise ValueError("External loads must have nonempty shape (scenarios, 6)")
    if (preloads.dtype not in (torch.float32, torch.float64)
            or any(x.dtype != preloads.dtype or x.device != preloads.device
                   for x in (rays, row_drag, loads))):
        raise ValueError("Equilibrium inputs must share float32/float64 dtype and device")
    for label, value in (("iterations", iterations), ("line_search_steps", line_search_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if not math.isfinite(damping) or damping <= 0 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Equilibrium damping and tolerance must be finite and positive")
    if not math.isfinite(backtracking_factor) or not 0 < backtracking_factor < 1:
        raise ValueError("Backtracking factor must lie strictly between zero and one")
    valid = (torch.isfinite(preloads).all(1) & (preloads >= 0).all(1)
             & torch.isfinite(rays).all((1, 2)) & torch.isfinite(row_drag).all(1)
             & (row_drag > 0).all(1) & torch.isfinite(loads).all())
    # The six-dimensional metric can be nearly singular even for valid finite
    # contact cones. Float64 keeps a structural missing axis finite rather than
    # turning it into an artificial invalid sample through a failed factorization.
    A = torch.where(valid[:, None, None], rays, 0.).to(torch.float64)
    c = torch.where(valid[:, None], row_drag, 1.).to(torch.float64)
    alpha = torch.where(valid[:, None], preloads, 0.).to(torch.float64)
    w = torch.where(torch.isfinite(loads), loads, 0.).to(torch.float64)
    scenarios = len(loads)
    bias = alpha.repeat_interleave(rays_per_contact, dim=-1) / rays_per_contact
    velocity = A.new_zeros(batch, scenarios, 6)
    identity = torch.eye(6, dtype=A.dtype, device=A.device)
    steps = backtracking_factor ** torch.arange(line_search_steps, dtype=A.dtype, device=A.device)
    source = torch.eye(contacts, dtype=A.dtype, device=A.device).repeat_interleave(rays_per_contact, 0) / rays_per_contact
    sensitivity = A.new_zeros(batch, scenarios, 6, contacts) if jacobian else None
    epsilon = torch.finfo(A.dtype).eps
    full_hessian = torch.einsum("bri,br,brj->bij", A, c, A)
    full_ridge = damping * full_hessian.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1.)
    full_inverse, full_status = torch.linalg.solve_ex(full_hessian + full_ridge[:, None, None] * identity,
                                                     identity.expand(batch, -1, -1), check_errors=False)
    full_numerical = (full_status == 0) & torch.isfinite(full_inverse).all((1, 2))
    full_inverse = torch.where(full_numerical[:, None, None], full_inverse, 0.)
    best_score = A.new_full((batch, scenarios), torch.inf)
    best_effort = torch.full_like(best_score, torch.inf)
    best_velocity = velocity
    best_sensitivity = sensitivity

    def retain_best(balance, force, current_velocity, current_sensitivity):
        score = balance.square().sum(-1)
        normal_force = force.reshape(batch, scenarios, contacts, rays_per_contact).sum(-1)
        effort = normal_force.square().sum(-1)
        finite = (torch.isfinite(score) & torch.isfinite(force).all(-1)
                  & torch.isfinite(current_velocity).all(-1) & torch.isfinite(effort))
        finite_best = torch.isfinite(best_score)
        scale = torch.maximum(score.abs(), torch.where(finite_best, best_score.abs(), 0.)).clamp_min(1.)
        roundoff = 64. * epsilon * scale
        improved = finite & (~finite_best | (score < best_score - roundoff)
                             | ((score - best_score).abs() <= roundoff) & (effort < best_effort))
        chosen_velocity = torch.where(improved[..., None], current_velocity, best_velocity)
        chosen_sensitivity = (torch.where(improved[..., None, None], current_sensitivity, best_sensitivity)
                              if jacobian else None)
        return (torch.where(improved, score, best_score), torch.where(improved, effort, best_effort),
                chosen_velocity, chosen_sensitivity)

    for _ in range(iterations):
        force = torch.relu(bias[:, None] - c[:, None] * torch.einsum("brj,bsj->bsr", A, velocity))
        balance = torch.einsum("brj,bsr->bsj", A, force) + w
        best_score, best_effort, best_velocity, best_sensitivity = retain_best(balance, force, velocity, sensitivity)
        active = (force > 0).to(A.dtype)
        hessian = torch.einsum("bri,bsr,brj->bsij", A, active * c[:, None], A)
        ridge = damping * hessian.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1.)
        matrix = hessian + ridge[:, :, None, None] * identity
        if jacobian:
            direct = torch.einsum("bri,bsr,rn->bsin", A, active, source)
            rhs = torch.cat((balance[..., None], direct - hessian @ sensitivity), dim=-1)
        else:
            rhs = balance[..., None]
        update, status = torch.linalg.solve_ex(matrix, rhs, check_errors=False)
        numerical = ((status == 0) & torch.isfinite(update).all((-1, -2)))
        update = torch.where(numerical[:, :, None, None], update, 0.)
        # The full-ray metric is a global quadratic majorizer. Its descent
        # candidate also starts zero-preload samples whose active Hessian is
        # identically zero, and prevents stalling at a changing active set.
        full_update = full_inverse[:, None] @ rhs
        updates = torch.stack((update, full_update), dim=2)
        delta = updates[..., 0]
        length = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.e-30)
        scale = (1.e6 / length).clamp_max(1.)
        if jacobian:
            derivative = updates[..., 1:]
            radial = (delta[..., None] * derivative).sum(-2) / length.square()[..., None]
            clipped = scale[..., None, None] * (derivative - delta[..., None] * radial[..., None, :])
            derivative = torch.where((scale < 1.)[..., None, None], clipped, derivative)
        delta = delta * scale[..., None]
        directions = torch.cat((steps[None, None, :, None] * delta[:, :, 0, None], delta[:, :, 1, None]), dim=2)
        trials = velocity[:, :, None] + directions
        trial_force = torch.relu(bias[:, None, None] - c[:, None, None]
            * torch.einsum("brj,bslj->bslr", A, trials))
        energy = .5 * (force.square() / c[:, None]).sum(-1) - (w * velocity).sum(-1)
        trial_energy = .5 * (trial_force.square() / c[:, None, None]).sum(-1) - (w[None, :, None] * trials).sum(-1)
        descent = (balance[:, :, None] * directions).sum(-1)
        roundoff = 16. * epsilon * energy.abs().clamp_min(1.)
        numerical_candidates = torch.cat((numerical[:, :, None].expand(-1, -1, line_search_steps),
                                          full_numerical[:, None, None].expand(-1, scenarios, 1)), dim=2)
        acceptable = (torch.isfinite(trial_energy) & numerical_candidates
                      & (trial_energy <= energy[:, :, None]
                         - 1.e-4 * descent + roundoff[:, :, None]))
        index = torch.where(acceptable, trial_energy, torch.inf).argmin(-1)
        accepted = acceptable.any(-1)
        chosen = directions.gather(2, index[..., None, None].expand(-1, -1, 1, 6)).squeeze(2)
        velocity = velocity + torch.where(accepted[..., None], chosen, 0.)
        if jacobian:
            derivatives = torch.cat((steps[None, None, :, None, None] * derivative[:, :, 0, None],
                                      derivative[:, :, 1, None]), dim=2)
            chosen_derivative = derivatives.gather(2, index[..., None, None, None].expand(-1, -1, 1, 6, contacts)).squeeze(2)
            sensitivity = sensitivity + torch.where(accepted[..., None, None], chosen_derivative, 0.)

    force = torch.relu(bias[:, None] - c[:, None] * torch.einsum("brj,bsj->bsr", A, velocity))
    balance = torch.einsum("brj,bsr->bsj", A, force) + w
    best_score, best_effort, best_velocity, best_sensitivity = retain_best(balance, force, velocity, sensitivity)
    velocity, sensitivity = best_velocity, best_sensitivity
    force = torch.relu(bias[:, None] - c[:, None] * torch.einsum("brj,bsj->bsr", A, velocity))
    balance = torch.einsum("brj,bsr->bsj", A, force) + w
    force_jacobian = balance_jacobian = None
    if jacobian:
        active = (force > 0).to(A.dtype)
        force_jacobian = active[..., None] * (source[None, None]
            - c[:, None, :, None] * torch.einsum("brj,bsjn->bsrn", A, sensitivity))
        balance_jacobian = torch.einsum("brj,bsrn->bsjn", A, force_jacobian)
        valid = valid & torch.isfinite(force_jacobian).all((1, 2, 3)) & torch.isfinite(balance_jacobian).all((1, 2, 3))
    valid = valid & torch.isfinite(best_score).all(1) & torch.isfinite(force).all((1, 2)) & torch.isfinite(balance).all((1, 2))
    force_scale = w.abs().amax().clamp_min(1.)
    converged = valid[:, None] & (balance.norm(dim=-1) <= tolerance * force_scale)
    def output(value):
        return value.to(preloads.dtype) if value is not None else None

    return EquilibriumResponse(output(balance), output(force), output(velocity), output(balance_jacobian),
                               output(force_jacobian), converged, valid)
