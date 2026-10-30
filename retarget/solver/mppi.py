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


def solve(problem, s, seed):
    rng = torch.Generator(device=problem.mean.device).manual_seed(seed)
    mean = problem.mean.clone()
    best = mean.clone()
    best_cost = float("inf")
    history = []
    sampled = 0
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
        history.append(
            dict(
                iteration=i,
                noise_ratio=scale,
                best_cost=best_cost,
                improvement=improvement,
                invalid=int((~e.valid).sum()),
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
            sampled + 1,
        )
    if not torch.isfinite(torch.tensor(best_cost)):
        raise NoValidCandidate(history)
    return Result(best, best_cost, history, sampled, sampled + 1)
