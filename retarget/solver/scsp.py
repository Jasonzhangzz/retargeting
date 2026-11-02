"""Contact-combination CEM and differentiable force/action solves with ADMM.

The two subproblems exchange a task-defined consensus vector. The humanoid
task uses contact positions and contact forces, with physical units scaled by
the task. Contact combinations and their force gradients are evaluated together
on the input tensor device. Forces use deterministic projected gradient
optimization; CEM samples only the finite catalog of contact combinations.
Only IPOPT's small action vector and gradient are
copied to the CPU; the trajectory and its derivatives stay in Torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable

import numpy as np
import torch

from .base import NoValidCandidate, Result


@dataclass(frozen=True)
class SCSPSettings:
    admm_iterations: int = 4
    rho: float = 10.0
    abs_tol: float = 1.0e-3
    rel_tol: float = 1.0e-3
    candidate_population: int = 24
    candidate_iterations: int = 3
    force_iterations: int = 20
    force_step_size: float = 0.1
    force_backtracks: int = 6
    force_tolerance: float = 1.0e-5
    elite_fraction: float = 0.2
    smoothing: float = 0.6
    min_probability: float = 0.02
    ipopt_max_iter: int = 80
    ipopt_max_cpu_time: float = 30.0
    ipopt_tol: float = 1.0e-5
    ipopt_acceptable_tol: float = 1.0e-4
    ipopt_print_level: int = 0

    def __post_init__(self):
        for name in (
            "admm_iterations", "candidate_population", "candidate_iterations",
            "force_iterations", "force_backtracks", "ipopt_max_iter",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"scsp.{name} must be a positive integer")
        for name in ("rho", "abs_tol", "rel_tol", "ipopt_tol", "ipopt_acceptable_tol", "ipopt_max_cpu_time", "force_step_size", "force_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"scsp.{name} must be finite and positive")
        for name in ("elite_fraction", "smoothing"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"scsp.{name} must lie in (0, 1]")
        if not math.isfinite(self.min_probability) or not 0 < self.min_probability < 1:
            raise ValueError("scsp.min_probability must lie in (0, 1)")
        if (
            isinstance(self.ipopt_print_level, bool)
            or not isinstance(self.ipopt_print_level, int)
            or not 0 <= self.ipopt_print_level <= 12
        ):
            raise ValueError("scsp.ipopt_print_level must be an integer in [0, 12]")


@dataclass
class ObjectEvaluation:
    cost: torch.Tensor  # (N,), without the augmented Lagrangian
    consensus: torch.Tensor  # (N, *consensus_shape)
    payload: Any = None


@dataclass
class RobotEvaluation:
    cost: torch.Tensor  # scalar, without the augmented Lagrangian
    consensus: torch.Tensor  # (*consensus_shape,)
    payload: Any = None


@dataclass
class ObjectSelection:
    indices: torch.Tensor
    forces: torch.Tensor
    consensus: torch.Tensor
    cost: float
    payload: Any = None


@dataclass
class SCSPProblem:
    """Coupled subproblems with optional contact-gated consensus constraints.

    ``prepare_iteration`` refreshes the robot consensus and Boolean mask once
    per ADMM round. That mask stays fixed throughout both subproblems; omitted
    masks activate every component for task-independent problems.
    """
    mean: torch.Tensor
    candidate_counts: torch.Tensor
    force_mean: torch.Tensor
    initial_consensus: torch.Tensor
    object_evaluate: Callable[[torch.Tensor, torch.Tensor], ObjectEvaluation]
    robot_evaluate: Callable[[torch.Tensor, ObjectSelection], RobotEvaluation]
    force_project: Callable = lambda indices, forces: forces
    lower: torch.Tensor | None = None
    upper: torch.Tensor | None = None
    consensus_mask: torch.Tensor | None = None
    prepare_iteration: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor] | None] | None = None


@dataclass
class SCSPResult(Result):
    selection: ObjectSelection
    consensus: torch.Tensor
    dual: torch.Tensor
    converged: bool


def _validate_mask(mask, reference):
    if (
        not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
        or mask.shape != reference.shape or mask.device != reference.device
    ):
        raise ValueError("SCSP consensus_mask must be Boolean and match the consensus shape/device")
    return mask.detach().clone()


def _mask_for(problem):
    active = getattr(problem, "_active_consensus_mask", None)
    if active is not None:
        return active
    if problem.consensus_mask is not None:
        return _validate_mask(problem.consensus_mask, problem.initial_consensus)
    return torch.ones_like(problem.initial_consensus, dtype=torch.bool)


def _masked(value, mask):
    # Sanitize before differences and squares: an inactive NaN must not leak
    # into the objective, multiplier or convergence tests through NaN * 0.
    return torch.where(mask, value, 0.0)


def _validate_consensus(value, reference, mask):
    if (
        not isinstance(value, torch.Tensor) or not value.is_floating_point()
        or value.shape != reference.shape or value.device != reference.device
    ):
        raise ValueError("SCSP robot consensus must match initial_consensus shape/device and be floating")
    if not bool(torch.isfinite(_masked(value, mask)).all()):
        raise ValueError("SCSP active robot consensus entries must be finite")


def _validate_problem(problem):
    if not isinstance(problem, SCSPProblem):
        raise TypeError("scsp requires an SCSPProblem with object and robot subproblems")
    tensors = (problem.mean, problem.force_mean, problem.initial_consensus)
    if any(not isinstance(x, torch.Tensor) or not x.is_floating_point() for x in tensors):
        raise ValueError("SCSP actions, force parameters and consensus must be floating tensors")
    if (
        problem.mean.ndim != 1 or not problem.mean.numel()
        or problem.force_mean.ndim != 1 or not problem.force_mean.numel()
        or problem.initial_consensus.ndim < 1 or not problem.initial_consensus.numel()
    ):
        raise ValueError("SCSP requires nonempty action/force vectors and consensus")
    if any(x.device != problem.mean.device for x in tensors):
        raise ValueError("SCSP tensors must share the action device")
    if any(not bool(torch.isfinite(x).all()) for x in tensors[:2]):
        raise ValueError("SCSP initial tensors must be finite")
    mask = (torch.ones_like(problem.initial_consensus, dtype=torch.bool)
            if problem.consensus_mask is None else _validate_mask(problem.consensus_mask, problem.initial_consensus))
    _validate_consensus(problem.initial_consensus, problem.initial_consensus, mask)
    if problem.prepare_iteration is not None and not callable(problem.prepare_iteration):
        raise ValueError("SCSP prepare_iteration must be callable")
    counts = torch.as_tensor(problem.candidate_counts, device=problem.mean.device)
    if (
        not counts.numel() or counts.is_floating_point() or counts.dtype == torch.bool
        or bool((counts < 1).any())
    ):
        raise ValueError("SCSP candidate_counts must contain positive integers")
    if math.prod(int(count) for count in counts.flatten().tolist()) > 121:
        raise ValueError("SCSP contact catalog must contain at most 121 combinations")
    for name in ("lower", "upper"):
        bound = getattr(problem, name)
        if bound is not None and (bound.shape != problem.mean.shape or bool(torch.isnan(bound).any())):
            raise ValueError(f"SCSP {name} must match the action vector and contain no NaNs")
    if problem.lower is not None and problem.upper is not None:
        if bool((problem.lower > problem.upper).any()):
            raise ValueError("SCSP action lower bounds exceed upper bounds")
    return counts.long()


def _augmented(cost, consensus, robot, dual, rho, mask=None):
    if cost.ndim != 1 or consensus.shape != (len(cost), *robot.shape):
        raise ValueError("object_evaluate must return cost (N,) and consensus (N, *initial_consensus.shape)")
    mask = torch.ones_like(robot, dtype=torch.bool) if mask is None else mask
    residual = _masked(consensus, mask) - _masked(robot, mask) + _masked(dual, mask)
    score = cost + 0.5 * rho * residual.flatten(1).square().sum(1)
    valid = torch.isfinite(cost) & torch.isfinite(residual).flatten(1).all(1)
    return torch.where(valid, score, torch.inf)


def _payload_at(payload, index, count):
    """Extract one batched payload without retaining the CEM population."""
    if isinstance(payload, torch.Tensor):
        value = payload[index] if payload.ndim and len(payload) == count else payload
        return value.detach().clone()
    if isinstance(payload, dict):
        return {key: _payload_at(value, index, count) for key, value in payload.items()}
    if is_dataclass(payload) and not isinstance(payload, type):
        return type(payload)(**{
            field.name: _payload_at(getattr(payload, field.name), index, count)
            for field in fields(payload)
        })
    if isinstance(payload, tuple):
        return tuple(_payload_at(value, index, count) for value in payload)
    if isinstance(payload, list):
        return [_payload_at(value, index, count) for value in payload]
    return payload


def _force_score(problem, indices, forces, robot, dual, rho):
    evaluation = problem.object_evaluate(indices, forces)
    score = _augmented(evaluation.cost, evaluation.consensus, robot, dual, rho, _mask_for(problem))
    return torch.where(torch.isfinite(forces).all(1), score, torch.inf)


def _force_optimize(problem, settings, indices, robot, dual, incumbent=None):
    """Deterministic batched projected Adam with independent line searches.

    Each combination owns one force vector, gradient and line-search step.
    There are no force particles or random force perturbations. Backtracking
    accepts only finite non-increasing augmented costs for each combination.
    """
    combinations = len(indices)
    initial = problem.force_mean if incumbent is None else incumbent.forces
    with torch.no_grad():
        current = problem.force_project(indices, initial.expand(combinations, -1).clone())
        if current.shape != (combinations, problem.force_mean.numel()):
            raise ValueError("force_project must preserve the force-parameter shape")
        current = current.detach()
        score = _force_score(problem, indices, current, robot, dual, settings.rho)
    metrics = dict(force_gradient_steps=0, force_gradient_batches=0,
                   force_evaluations=combinations, force_line_search_evaluations=0)
    if incumbent is not None:
        # A deterministic nominal restart avoids inheriting an invalid or poor
        # force seed when the selected surface normal changes.
        with torch.no_grad():
            restart = problem.force_project(indices, problem.force_mean.expand_as(current).clone())
            restart_score = _force_score(problem, indices, restart, robot, dual, settings.rho)
            use_restart = restart_score < score
            current = torch.where(use_restart[:, None], restart, current)
            score = torch.minimum(score, restart_score)
        metrics["force_evaluations"] += combinations
    best, best_score = current.clone(), score.clone()
    active = torch.isfinite(score) & torch.isfinite(current).all(1)
    moment = torch.zeros_like(current)
    variance = torch.zeros_like(current)
    for iteration in range(settings.force_iterations):
        ids = torch.where(active)[0]
        if not ids.numel():
            break
        with torch.enable_grad():
            variable = current[ids].detach().clone().requires_grad_(True)
            local_score = _force_score(problem, indices[ids], variable, robot, dual, settings.rho)
            finite = torch.isfinite(local_score)
            objective = torch.where(finite, local_score, 0.0).sum()
            if not objective.requires_grad:
                raise ValueError("SCSP force optimization requires differentiable object_evaluate")
            gradient = torch.autograd.grad(objective, variable, allow_unused=True)[0]
        if gradient is None:
            raise ValueError("SCSP object cost/consensus is disconnected from force parameters")
        gradient = gradient.detach()
        metrics["force_gradient_steps"] += len(ids)
        metrics["force_gradient_batches"] += 1
        metrics["force_evaluations"] += len(ids)
        finite = finite & torch.isfinite(gradient).all(1)
        small = gradient.abs().amax(1) <= settings.force_tolerance
        active[ids] = finite & ~small
        gradient = torch.where(finite[:, None], gradient, 0.0)
        moment[ids] = 0.9 * moment[ids] + 0.1 * gradient
        variance[ids] = 0.999 * variance[ids] + 0.001 * gradient.square()
        corrected_moment = moment[ids] / (1.0 - 0.9 ** (iteration + 1))
        corrected_variance = variance[ids] / (1.0 - 0.999 ** (iteration + 1))
        direction = corrected_moment / (corrected_variance.sqrt() + 1.0e-12)
        # Momentum can point uphill after crossing a minimum. A normalized
        # current gradient restores descent before projected backtracking.
        descent = (direction * gradient).sum(1) > 0
        fallback = gradient / gradient.abs().clamp_min(1.0e-12)
        direction = torch.where(descent[:, None], direction, fallback)
        full_direction = torch.zeros_like(current)
        full_direction[ids] = direction
        pending = active.clone()
        steps = current.new_full((combinations,), settings.force_step_size)
        for _ in range(settings.force_backtracks):
            trial_ids = torch.where(pending)[0]
            if not trial_ids.numel():
                break
            with torch.no_grad():
                trial = problem.force_project(
                    indices[trial_ids],
                    current[trial_ids] - steps[trial_ids, None] * full_direction[trial_ids],
                )
                trial_score = _force_score(problem, indices[trial_ids], trial, robot, dual, settings.rho)
                accepted = torch.isfinite(trial_score) & (trial_score <= score[trial_ids])
                accepted_ids = trial_ids[accepted]
                if accepted_ids.numel():
                    displacement = (trial[accepted] - current[accepted_ids]).abs().amax(1)
                    current[accepted_ids] = trial[accepted]
                    score[accepted_ids] = trial_score[accepted]
                    improved = trial_score[accepted] < best_score[accepted_ids]
                    improved_ids = accepted_ids[improved]
                    best[improved_ids] = trial[accepted][improved]
                    best_score[improved_ids] = trial_score[accepted][improved]
                    active[accepted_ids] &= displacement > settings.force_tolerance
                pending[accepted_ids] = False
                steps[trial_ids] *= 0.5
            metrics["force_evaluations"] += len(trial_ids)
            metrics["force_line_search_evaluations"] += len(trial_ids)
        # A line search with no finite improving projected step has exhausted
        # this combination's local update; retain its best finite force vector.
        active[pending] = False
    return best.detach(), best_score.detach(), metrics


def _contact_catalog(counts):
    size = math.prod(int(value) for value in counts.flatten().tolist())
    if size > 121:
        raise ValueError("SCSP contact catalog must contain at most 121 combinations")
    axes = [torch.arange(int(value), device=counts.device) for value in counts.flatten()]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(size, *counts.shape)


def _object_cem(problem, settings, counts, robot, dual, rng, incumbent):
    """Categorical CEM without replacement from the finite contact catalog."""
    catalog = _contact_catalog(counts)
    flat_catalog = catalog.reshape(len(catalog), -1)
    flat_counts = counts.flatten()
    max_count = int(flat_counts.max())
    valid_categories = torch.arange(max_count, device=counts.device)[None] < flat_counts[:, None]
    probabilities = valid_categories.to(problem.mean.dtype) / flat_counts[:, None]
    visited = torch.zeros(len(catalog), device=counts.device, dtype=torch.bool)
    catalog_score = problem.mean.new_full((len(catalog),), torch.inf)
    best_indices = None
    best_force = None
    best_cost = float("inf")
    metrics = dict(contact_universe=len(catalog), contact_combinations=0,
                   force_gradient_steps=0, force_gradient_batches=0,
                   force_evaluations=0, force_line_search_evaluations=0)
    visited_indices = []
    for iteration in range(settings.candidate_iterations):
        available = torch.where(~visited)[0]
        if not available.numel():
            break
        count = min(settings.candidate_population, len(available))
        chosen = []
        if iteration == 0:
            # Always evaluate the dataset's closest projection. The previous
            # ADMM selection is another deterministic contact proposal.
            if incumbent is not None:
                matches = (flat_catalog == incumbent.indices.flatten()).all(1)
                chosen.append(int(torch.where(matches)[0][0]))
            if len(chosen) < count and 0 not in chosen:
                chosen.append(0)
        remaining = ~visited.clone()
        if chosen:
            remaining[chosen] = False
        remaining_ids = torch.where(remaining)[0]
        if len(chosen) < count:
            coordinates = torch.arange(flat_catalog.shape[1], device=counts.device)
            log_weight = probabilities[coordinates[None], flat_catalog[remaining_ids]].log().sum(1)
            weights = torch.softmax(log_weight, 0)
            random_ids = torch.multinomial(weights, count - len(chosen), replacement=False, generator=rng)
            chosen.extend(remaining_ids[random_ids].tolist())
        ids = torch.tensor(chosen, device=counts.device, dtype=torch.long)
        visited[ids] = True
        indices = catalog[ids]
        visited_indices.append(indices)
        warm = incumbent if best_force is None else ObjectSelection(best_indices, best_force, robot, 0.0)
        forces, scores, force_metrics = _force_optimize(problem, settings, indices, robot, dual, warm)
        catalog_score[ids] = scores
        for key, value in force_metrics.items():
            metrics[key] += value
        winner = int(scores.argmin())
        if float(scores[winner]) < best_cost:
            best_cost = float(scores[winner])
            best_indices, best_force = indices[winner].clone(), forces[winner].clone()
        finite = torch.isfinite(catalog_score) & visited
        if bool(finite.any()):
            elite_count = max(1, math.ceil(int(finite.sum()) * settings.elite_fraction))
            elite_ids = torch.argsort(catalog_score, stable=True)[:elite_count]
            elite = flat_catalog[elite_ids]
            frequency = torch.nn.functional.one_hot(elite, max_count).to(probabilities.dtype).mean(0)
            frequency = frequency + settings.min_probability * valid_categories
            frequency = frequency / frequency.sum(1, keepdim=True)
            probabilities = torch.lerp(probabilities, frequency, settings.smoothing)
            probabilities = torch.where(valid_categories, probabilities, 0.0)
    metrics["contact_combinations"] = int(visited.sum())
    metrics["visited_indices"] = torch.cat(visited_indices) if visited_indices else catalog[:0]
    problem._object_search_statistics = metrics
    if best_force is None:
        raise NoValidCandidate([dict(stage="object_cem", contact_combinations=int(visited.sum()))])
    with torch.no_grad():
        check = problem.object_evaluate(best_indices[None], best_force[None])
        score = _augmented(check.cost, check.consensus, robot, dual, settings.rho, _mask_for(problem))
    if not bool(torch.isfinite(score[0])) or not bool(torch.isfinite(best_force).all()):
        raise NoValidCandidate([dict(stage="object_cem_restoration", contact_combinations=int(visited.sum()))])
    metrics["force_evaluations"] += 1
    return ObjectSelection(
        best_indices.detach(), best_force.detach(), check.consensus[0].detach().clone(),
        float(check.cost[0]), _payload_at(check.payload, 0, 1),
    ), metrics["contact_combinations"], metrics["force_evaluations"]


class _TorchObjective:
    """Cache IPOPT's objective and the task callback's Torch derivative.

    Correctness of that derivative depends on the task keeping its complete
    forward model in the graph, or freezing a consistent local forward model.
    """

    def __init__(self, problem, selection, dual, rho):
        self.problem, self.selection = problem, selection
        self.dual, self.rho = dual, rho
        self.mask = _mask_for(problem).detach().clone()
        self.x, self.value, self.derivative = None, None, None
        self.evaluated = 0
        self.best_x, self.best_value = None, float("inf")

    def _update(self, x):
        if self.x is not None and np.array_equal(x, self.x):
            return
        with torch.enable_grad():
            variable = torch.as_tensor(x, dtype=self.problem.mean.dtype, device=self.problem.mean.device).clone().requires_grad_(True)
            evaluation = self.problem.robot_evaluate(variable, self.selection)
            if evaluation.cost.ndim != 0 or evaluation.consensus.shape != self.dual.shape:
                raise ValueError("robot_evaluate must return a scalar cost and initial_consensus-shaped consensus")
            residual = (_masked(self.selection.consensus, self.mask)
                        - _masked(evaluation.consensus, self.mask)
                        + _masked(self.dual, self.mask))
            value = evaluation.cost + 0.5 * self.rho * residual.square().sum()
            if not value.requires_grad:
                raise ValueError("SCSP IPOPT requires a differentiable robot_evaluate")
            gradient = torch.autograd.grad(value, variable, allow_unused=True)[0]
        if gradient is None:
            raise ValueError("SCSP robot cost/consensus is disconnected from the action vector")
        if not bool(torch.isfinite(value)) or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("SCSP IPOPT received a nonfinite cost or action derivative")
        self.evaluated += 1
        self.x = np.array(x, dtype=np.float64, copy=True)
        self.value = float(value.detach())
        self.derivative = gradient.detach().double().cpu().numpy().copy()
        if self.value < self.best_value:
            self.best_value, self.best_x = self.value, self.x.copy()

    def objective(self, x):
        self._update(x)
        return self.value

    def gradient(self, x):
        self._update(x)
        return self.derivative


def _load_ipopt():
    try:
        import cyipopt
    except ImportError as exc:
        raise ImportError(
            "SCSP action optimization requires cyipopt and IPOPT. Install the "
            "system IPOPT library/development package and requirements-scsp.txt."
        ) from exc
    return cyipopt


def _robot_ipopt(problem, selection, actions, dual, settings, cyipopt):
    objective = _TorchObjective(problem, selection, dual, settings.rho)
    initial = actions.detach().double().cpu().numpy().copy()
    lower = np.full_like(initial, -np.inf) if problem.lower is None else problem.lower.detach().double().cpu().numpy()
    upper = np.full_like(initial, np.inf) if problem.upper is None else problem.upper.detach().double().cpu().numpy()
    initial = np.clip(initial, lower, upper)
    nlp = cyipopt.Problem(n=len(initial), m=0, problem_obj=objective, lb=lower, ub=upper)
    for key, value in {
        "hessian_approximation": "limited-memory", "max_iter": settings.ipopt_max_iter,
        "max_cpu_time": settings.ipopt_max_cpu_time,
        "tol": settings.ipopt_tol, "acceptable_tol": settings.ipopt_acceptable_tol,
        "print_level": settings.ipopt_print_level, "sb": "yes",
        "bound_relax_factor": 0.0,
    }.items():
        nlp.add_option(key, value)
    try:
        solution, info = nlp.solve(initial)
    finally:
        nlp.close()
    status = int(info["status"])
    message = info.get("status_msg", "")
    if isinstance(message, bytes):
        message = message.decode("utf-8", errors="replace")
    success = status in (0, 1, 6)
    # A bounded iteration/time budget can produce a useful finite iterate.
    # Report this explicitly and never count it as an ADMM convergence result.
    if not success:
        if status not in (-1, -4) or objective.best_x is None:
            raise RuntimeError(f"SCSP IPOPT failed with status {status}: {message}")
        solution = objective.best_x
    actions = torch.as_tensor(solution, device=actions.device, dtype=actions.dtype).detach()
    with torch.no_grad():
        evaluation = problem.robot_evaluate(actions, selection)
    if not bool(torch.isfinite(evaluation.cost)) or not bool(torch.isfinite(_masked(evaluation.consensus, objective.mask)).all()):
        raise FloatingPointError("SCSP IPOPT final robot trajectory is nonfinite")
    return actions, evaluation, dict(
        ipopt_status=status, ipopt_success=success, ipopt_message=str(message),
        ipopt_evaluated=objective.evaluated,
    ), objective.evaluated + 1


def solve(problem, settings, seed=0):
    """Alternate contact CEM / force gradients and robot-action IPOPT.

    ``initial_consensus`` is the current robot-side value. Both callbacks return
    their own unaugmented costs; this function owns the scaled ADMM dual and
    applies the same masked ``z - y + dual`` residual to each subproblem.
    Inactive contacts contribute no equality penalty, multiplier or stopping
    residual. Robot base-cost optimization still runs when no equality is active.
    """
    settings = settings.scsp if hasattr(settings, "scsp") else settings
    if not isinstance(settings, SCSPSettings):
        raise TypeError("scsp requires SCSPSettings")
    counts = _validate_problem(problem)
    cyipopt = _load_ipopt()  # fail before spending a GPU CEM budget
    rng = torch.Generator(device=problem.mean.device).manual_seed(seed)
    actions = problem.mean.detach().clone()
    robot = problem.initial_consensus.detach().clone()
    mask = (torch.ones_like(robot, dtype=torch.bool) if problem.consensus_mask is None
            else _validate_mask(problem.consensus_mask, robot))
    problem._active_consensus_mask = mask
    dual = torch.zeros_like(robot)
    selection, history = None, []
    sampled = evaluated = 0
    visited_combinations = set()
    converged = False
    for iteration in range(settings.admm_iterations):
        previous_mask = mask
        prepared_count = 0
        if problem.prepare_iteration is not None:
            with torch.no_grad():
                prepared = problem.prepare_iteration(actions.detach().clone())
            prepared_count = 1
            evaluated += 1
            if prepared is not None:
                if not isinstance(prepared, tuple) or len(prepared) != 2:
                    raise ValueError("SCSP prepare_iteration must return (robot_consensus, mask) or None")
                prepared_robot, prepared_mask = prepared
                mask = _validate_mask(prepared_mask, problem.initial_consensus)
                _validate_consensus(prepared_robot, problem.initial_consensus, mask)
                robot = prepared_robot.detach().clone()
        # Retain multipliers only for constraints active in consecutive rounds.
        # Newly activated contacts start with zero multiplier; inactive entries
        # never accumulate a position or force discrepancy.
        dual = torch.where(mask & previous_mask, dual, 0.0)
        problem._active_consensus_mask = mask.detach().clone()
        previous_robot = robot
        with torch.no_grad():
            selection, combinations, calls = _object_cem(
                problem, settings, counts, robot, dual, rng, selection
            )
        search = problem._object_search_statistics
        visited_combinations.update(
            tuple(int(value) for value in row)
            for row in search["visited_indices"].reshape(combinations, -1).tolist()
        )
        sampled = len(visited_combinations)
        evaluated += calls
        actions, check, ipopt, calls = _robot_ipopt(
            problem, selection, actions, dual, settings, cyipopt
        )
        evaluated += calls
        robot = check.consensus.detach().clone()
        residual = _masked(selection.consensus, mask) - _masked(robot, mask)
        dual = _masked(dual + residual, mask)
        primal = float(torch.linalg.vector_norm(residual))
        dual_residual = settings.rho * float(torch.linalg.vector_norm(
            _masked(robot, mask) - _masked(previous_robot, mask)
        ))
        active_components = int(mask.sum())
        primal_tol = math.sqrt(active_components) * settings.abs_tol + settings.rel_tol * max(
            float(torch.linalg.vector_norm(_masked(selection.consensus, mask))),
            float(torch.linalg.vector_norm(_masked(robot, mask))),
        )
        dual_tol = math.sqrt(active_components) * settings.abs_tol + settings.rel_tol * settings.rho * float(torch.linalg.vector_norm(dual))
        converged = primal <= primal_tol and dual_residual <= dual_tol and ipopt["ipopt_success"]
        history.append(dict(
            iteration=iteration, object_cost=selection.cost, robot_cost=float(check.cost),
            cost=selection.cost + float(check.cost), rho=settings.rho,
            primal_residual=primal, dual_residual=dual_residual,
            primal_tolerance=primal_tol, dual_tolerance=dual_tol,
            converged=converged, constraints_active=active_components,
            prepare_evaluated=prepared_count, unique_contact_combinations=sampled,
            **{key: value for key, value in search.items() if key != "visited_indices"},
            **ipopt,
        ))
        if converged:
            break
    return SCSPResult(
        parameters=actions, cost=history[-1]["cost"], history=history,
        sampled=sampled, evaluated=evaluated, selection=selection,
        consensus=robot, dual=dual, converged=converged,
    )
