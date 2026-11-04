"""CARD elite-mean CEM, with fixed geometric noise annealing."""

import math
import torch
from .base import Result, NoValidCandidate, evaluate


def elite_mean(candidates, scores, fraction, temperature):
    order = torch.argsort(
        torch.where(torch.isfinite(scores), scores, torch.inf), stable=True
    )
    ids = order[: max(1, math.ceil(len(scores) * fraction))]
    ids = ids[torch.isfinite(scores[ids])]
    if not len(ids):
        return None
    c = scores[ids]
    weights = torch.softmax(
        -(c - c.min()) / (c.std(correction=0).clamp_min(1.0e-12) * temperature), 0
    )
    return (weights.to(candidates.dtype)[:, None] * candidates[ids]).sum(0)


def solve(problem, s, seed):
    if problem.mean.ndim == 2:
        return solve_batched(problem, s, seed)
    generator = torch.Generator(device=problem.mean.device).manual_seed(seed)
    mean = problem.mean.clone()
    best = mean.clone()
    best_cost = float("inf")
    history = []
    for i in range(s.iterations):
        scale = s.noise_ratio ** (i / s.iterations)
        values = (
            mean
            + torch.randn(
                (s.population, len(mean)),
                device=mean.device,
                dtype=mean.dtype,
                generator=generator,
            )
            * problem.sigma
            * scale
        )
        values[0] = best
        if s.population > 1:
            values[1] = mean
        values = problem.project(values)
        result = evaluate(problem, values)
        candidate = int(torch.where(result.valid, result.cost, torch.inf).argmin())
        if bool(result.valid[candidate]) and float(result.cost[candidate]) < best_cost:
            best = values[candidate].clone()
            best_cost = float(result.cost[candidate])
        update = elite_mean(values, result.search, s.elite_fraction, s.temp)
        if update is not None:
            mean = update
        history.append(
            dict(
                iteration=i,
                noise_ratio=scale,
                best_cost=best_cost,
                minimum_cost=float(
                    torch.where(result.valid, result.cost, torch.inf).min()
                ),
                invalid=int((~result.valid).sum()),
            )
        )
    if not math.isfinite(best_cost):
        raise NoValidCandidate(history)
    check = evaluate(problem, best[None])
    if not bool(check.valid[0]):
        raise NoValidCandidate(history)
    return Result(
        best,
        float(check.cost[0]),
        history,
        s.population * s.iterations,
        s.population * s.iterations + 1,
    )


def solve_batched(problem, s, seed):
    mean = problem.mean.clone()
    batch, dimension = mean.shape
    generators = [torch.Generator(device=mean.device).manual_seed(seed + row) for row in range(batch)]
    best = mean.clone()
    best_cost = mean.new_full((batch,), torch.inf)
    rows = torch.arange(batch, device=mean.device)
    elite_count = max(1, math.ceil(s.population * s.elite_fraction))
    history = []
    for iteration in range(s.iterations):
        scale = s.noise_ratio ** (iteration / s.iterations)
        noise = torch.stack([torch.randn((s.population, dimension), device=mean.device,
            dtype=mean.dtype, generator=g) for g in generators])
        values = mean[:, None] + noise * problem.sigma[:, None] * scale
        values[:, 0] = best
        if s.population > 1:
            values[:, 1] = mean
        values = problem.project(values)
        result = evaluate(problem, values)
        valid_cost = torch.where(result.valid, result.cost, torch.inf)
        candidate = valid_cost.argmin(1)
        candidate_cost = valid_cost[rows, candidate]
        improved = candidate_cost < best_cost
        best = torch.where(improved[:, None], values[rows, candidate], best)
        best_cost = torch.minimum(best_cost, candidate_cost)
        order = torch.argsort(torch.where(torch.isfinite(result.search), result.search, torch.inf), dim=1, stable=True)
        ids = order[:, :elite_count]
        scores = result.search.gather(1, ids)
        finite = torch.isfinite(scores)
        number = finite.sum(1).clamp_min(1)
        clean = torch.where(finite, scores, 0.)
        average = clean.sum(1) / number
        variance = torch.where(finite, (clean-average[:, None]).square(), 0.).sum(1) / number
        minimum = torch.where(finite, scores, torch.inf).min(1).values
        logits = -(scores-minimum[:, None]) / (variance.sqrt().clamp_min(1.e-12)[:, None] * s.temp)
        logits = torch.where(finite, logits, -torch.inf)
        has_update = finite.any(1)
        weights = torch.softmax(torch.where(has_update[:, None], logits, torch.zeros_like(logits)), 1)
        elites = values.gather(1, ids[:, :, None].expand(-1, -1, dimension))
        update = (weights.to(values.dtype)[:, :, None] * elites).sum(1)
        mean = torch.where(has_update[:, None], update, mean)
        history.append(dict(iteration=iteration, noise_ratio=scale, best_cost=best_cost.clone(),
            minimum_cost=valid_cost.min(1).values, invalid=(~result.valid).sum(1)))
        if problem.observer is not None:
            problem.observer(iteration+1, best, best_cost,
                torch.full((batch,), (iteration+1)*s.population, device=mean.device, dtype=torch.long))
    check = evaluate(problem, best[:, None])
    cost = torch.where(check.valid[:, 0] & torch.isfinite(best_cost), check.cost[:, 0], torch.inf)
    return Result(best, cost, history, batch*s.population*s.iterations, batch*(s.population*s.iterations+1))
