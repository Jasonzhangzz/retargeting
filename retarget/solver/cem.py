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
