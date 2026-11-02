"""SPIDER-style top-decile weighted sampling with annealing and early stopping."""

import torch
from .base import Result, NoValidCandidate, evaluate


def weights(cost, temperature):
    finite = torch.isfinite(cost)
    if not bool(finite.any()):
        return None
    ids = torch.argsort(torch.where(finite, cost, torch.inf), stable=True)[
        : max(1, int(0.1 * len(cost)))
    ]
    ids = ids[finite[ids]]
    c = cost[ids]
    std = c.std(correction=1) if len(c) > 1 else c.new_zeros(())
    w = torch.softmax(-(c - c.mean()) / ((std + 1.0e-2) * temperature), 0)
    return ids, w


def refine_mean(problem, mean, settings):
    """Refine the MPPI mean on its existing objective, with monotone steps.

    Descent uses standardized action coordinates: a unit step has the scale of
    ``problem.sigma``. Only the current mean retains an autograd graph; line
    search trials and the population retain the original no-grad behavior.
    """
    mean = problem.project(mean[None])[0].detach()
    info = dict(
        gradient_accepted_steps=0,
        gradient_evaluated=0,
        gradient_initial_cost=float("inf"),
        gradient_final_cost=float("inf"),
    )
    for step in range(settings.gradient_steps):
        with torch.enable_grad():
            variable = mean.detach().requires_grad_(True)
            result = evaluate(problem, problem.project(variable[None]))
            objective = result.cost[0]
            info["gradient_evaluated"] += 1
            current_cost = float(objective.detach())
            if step == 0:
                info["gradient_initial_cost"] = current_cost
            info["gradient_final_cost"] = current_cost
            if not bool(result.valid[0]):
                break
            if not objective.requires_grad:
                raise ValueError(
                    "gradient_steps requires a differentiable problem.evaluate"
                )
            gradient = torch.autograd.grad(
                objective, variable, allow_unused=True
            )[0]
        if gradient is None:
            raise ValueError(
                "gradient_steps requires a cost connected to the action parameters"
            )
        if not bool(torch.isfinite(gradient).all()):
            break
        sigma = problem.sigma.to(mean).abs()
        scaled = sigma * gradient
        norm = torch.linalg.vector_norm(scaled)
        if not bool(torch.isfinite(norm)) or not bool(norm > 0):
            break
        direction = sigma * (scaled / norm)
        accepted = False
        with torch.no_grad():
            for backtrack in range(settings.gradient_backtracks):
                trial = problem.project(
                    (mean - settings.gradient_step_size * 0.5**backtrack * direction)[
                        None
                    ]
                )
                check = evaluate(problem, trial)
                info["gradient_evaluated"] += 1
                if bool(check.valid[0]) and float(check.cost[0]) < current_cost:
                    mean = trial[0].detach()
                    info["gradient_final_cost"] = float(check.cost[0])
                    info["gradient_accepted_steps"] += 1
                    accepted = True
                    break
        if not accepted:
            break
    return mean, info


def solve(problem, s, seed):
    rng = torch.Generator(device=problem.mean.device).manual_seed(seed)
    mean = problem.mean.clone()
    best = mean.clone()
    best_cost = float("inf")
    history = []
    sampled = 0
    gradient_evaluated = 0
    for i in range(s.iterations):
        scale = s.noise_ratio ** (i / s.iterations)
        noise = (
            torch.randn(
                (s.population, len(mean)),
                device=mean.device,
                dtype=mean.dtype,
                generator=rng,
            )
            * problem.sigma
            * scale
        )
        noise[0] = 0
        count = int(s.population * s.exploit_ratio)
        if count:
            noise[-count:] *= s.exploit_noise
        values = problem.project(mean + noise)
        e = evaluate(problem, values)
        sampled += s.population
        winner = int(torch.where(e.valid, e.cost, torch.inf).argmin())
        if bool(e.valid[winner]) and float(e.cost[winner]) < best_cost:
            best = values[winner].clone()
            best_cost = float(e.cost[winner])
        selected = weights(e.search, s.temp)
        if selected is not None:
            ids, w = selected
            mean = (w.to(values.dtype)[:, None] * values[ids]).sum(0)
        improvement = float(e.cost[0] - torch.where(e.valid, e.cost, torch.inf).min())
        gradient_info = {}
        if s.gradient_steps:
            mean, gradient_info = refine_mean(problem, mean, s)
            gradient_evaluated += gradient_info["gradient_evaluated"]
            refined_cost = gradient_info["gradient_final_cost"]
            if refined_cost < best_cost:
                best, best_cost = mean.clone(), refined_cost
            if gradient_info["gradient_accepted_steps"]:
                improvement = max(
                    improvement,
                    gradient_info["gradient_initial_cost"] - refined_cost,
                )
        history.append(
            dict(
                iteration=i,
                noise_ratio=scale,
                best_cost=best_cost,
                improvement=improvement,
                invalid=int((~e.valid).sum()),
                **gradient_info,
            )
        )
        if len(history) >= s.improvement_check_steps and all(
            h["improvement"] < s.improvement_threshold
            for h in history[-s.improvement_check_steps :]
        ):
            break
    final = evaluate(problem, problem.project(mean[None]))
    if bool(final.valid[0]):
        return Result(
            problem.project(mean[None])[0],
            float(final.cost[0]),
            history,
            sampled,
            sampled + gradient_evaluated + 1,
        )
    if not torch.isfinite(torch.tensor(best_cost)):
        raise NoValidCandidate(history)
    return Result(best, best_cost, history, sampled, sampled + gradient_evaluated + 1)
