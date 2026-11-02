"""Dense-control SPIDER MPPI adapter, with correlated knot perturbations.

Derived from facebookresearch/spider 44717007 (CC BY-NC 4.0).
See third_party/SPIDER/dex/UPSTREAM.json and licenses/SPIDER.txt.
"""

import time
import torch

from retarget.env.humanoid.interp import interp


def noise_scale(population, knots, channels, *, first=.5, last=1., exploit_ratio=.01,
                exploit_noise=.01):
    temporal = torch.logspace(torch.log10(channels.new_tensor(first)),
                             torch.log10(channels.new_tensor(last)), knots,
                             device=channels.device, dtype=channels.dtype)
    scale = (temporal[:, None]*channels[None]).expand(population, -1, -1).clone()
    scale[0] = 0.
    count = int(population*exploit_ratio)
    # Fix the original [-0:] small-budget slicing edge case.
    if count:
        scale[-count:] *= exploit_noise
    return scale


def reward_weights(rewards, temperature):
    """Original top-decile standardized softmax, excluding invalid rollouts."""
    finite = torch.isfinite(rewards)
    if not bool(finite.any()):
        raise FloatingPointError("All SPIDER MPPI candidates have invalid dynamics/cost")
    count = max(1, int(.1*len(rewards)))
    values, ids = torch.topk(torch.where(finite, rewards, -torch.inf), count)
    keep = torch.isfinite(values)
    values, ids = values[keep], ids[keep]
    std = values.std() if len(values) > 1 else values.new_zeros(())
    weights = torch.softmax((values-values.mean())/(std+.01)/temperature, 0)
    return ids, weights


def optimize(mean, evaluate, settings, *, knots, knot_steps, channel_noise, seed,
             first_noise=.5, last_noise=1.,return_best=False):
    """Return the original generic SPIDER weighted dense mean, not FA controls.

    ``evaluate`` receives [N,H,nu] absolute controls and restores the physical
    initial state for every population. No virtual object actuators are added.
    """
    if settings.gradient_steps:
        raise ValueError("SPIDER sampling runs without gradient refinement")
    if mean.shape != (knots*knot_steps, len(channel_noise)):
        raise ValueError("Dense SPIDER nominal controls disagree with the knot grid")
    scale = noise_scale(settings.population, knots, channel_noise,
        first=first_noise, last=last_noise, exploit_ratio=settings.exploit_ratio,
        exploit_noise=settings.exploit_noise)
    rng = torch.Generator(device=mean.device).manual_seed(seed)
    current = mean.clone()
    best, best_cost = current.clone(), float("inf")
    history = []
    started = time.perf_counter()
    for iteration in range(settings.iterations):
        noise = torch.randn(scale.shape, generator=rng, device=mean.device,
                            dtype=mean.dtype)*scale*settings.noise_ratio**(iteration/settings.iterations)
        candidates = current[None] + interp(noise, knot_steps)
        costs = evaluate(candidates)
        ids, weights = reward_weights(-costs, settings.temp)
        winner = ids[0]
        if float(costs[winner]) < best_cost:
            best, best_cost = candidates[winner].clone(), float(costs[winner])
        improvement = float(costs[0]-costs[winner])
        current = (weights[:, None, None]*candidates[ids]).sum(0)
        history.append(dict(iteration=iteration, best_cost=best_cost,
            improvement=improvement, invalid=int((~torch.isfinite(costs)).sum())))
        if len(history) >= settings.improvement_check_steps and all(
                row["improvement"] < settings.improvement_threshold
                for row in history[-settings.improvement_check_steps:]):
            break
    final = float(evaluate(current[None])[0])
    if not torch.isfinite(mean.new_tensor(final)):
        current, final = best, best_cost
    if return_best and best_cost<final:
        current,final=best,best_cost
    return current, dict(cost=final, seconds=time.perf_counter()-started,
        iterations=len(history), sampled=len(history)*settings.population,
        evaluated=len(history)*settings.population+1, history=history,
        policy="best evaluated MPPI sequence including weighted mean" if return_best else "generic SPIDER dense weighted mean; finite-candidate fallback")


def shift_controls(controls, executed, reference_tail):
    if not 0 < executed <= len(controls) or reference_tail.shape != controls[:executed].shape:
        raise ValueError("SPIDER control shift must append one reference action per executed step")
    return torch.cat((controls[executed:], reference_tail), 0)
