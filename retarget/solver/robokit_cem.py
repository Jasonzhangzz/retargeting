"""CEM refit extension of Robokit's GaussianSampler and population lifecycle."""
from dataclasses import dataclass
import math
import numpy as np
import warp as wp
from robokit.opt.sampler import GaussianSampler, GaussianSamplerConfig
from robokit.opt.population_solver import PopulationSolver


class CEMSampler(GaussianSampler):
    def _move(self, scores: wp.array, out_delta: wp.array) -> None:
        # Drawing, bounded-noise correction, variable integration, seeds and
        # population ownership remain Robokit's GaussianSampler implementation.
        cost = scores.numpy().reshape(self.instances, self.num_draws)
        noise = self.noise.numpy().reshape(self.instances, self.num_draws, self.tangent_dim)
        old, floor = self.std.numpy(), self.std_reset.numpy() * self.cem_min_sigma_fraction
        delta, spread = np.zeros_like(old), old.copy()
        for branch in range(self.instances):
            valid = np.flatnonzero(np.isfinite(cost[branch]))
            if not len(valid):
                continue
            ordered = valid[np.argsort(cost[branch, valid], kind='stable')]
            elite = ordered[:max(1, math.ceil(len(ordered) * self.cem_elite_fraction))]
            delta[branch] = self.cem_refit * noise[branch, elite].mean(0)
            spread[branch] = np.maximum((1 - self.cem_refit) * old[branch]
                + self.cem_refit * noise[branch, elite].std(0), floor[branch])
        self.std.assign(spread)
        out_delta.assign(delta)


@dataclass(frozen=True)
class CEMSamplerConfig(GaussianSamplerConfig):
    branch_sigma: object = None
    cem_elite_fraction: float = .2
    cem_refit: float = .5
    cem_min_sigma_fraction: float = .05

    def build(self, var, num_candidates: int, active_dof_mask=None) -> CEMSampler:
        width = var.get(self.var_key).tangent_dim
        offset = var.tangent_offset(self.var_key)
        sampler = CEMSampler(var_key=self.var_key, num_draws=self.num_draws, tangent_dim=width,
            instances=var.batch_size, seed=self.seed, init_std=self.init_std, min_std=0.,
            active_dof_mask=None if active_dof_mask is None else active_dof_mask[offset:offset + width],
            device=var.device)
        sampler.cem_elite_fraction = self.cem_elite_fraction
        sampler.cem_refit = self.cem_refit
        sampler.cem_min_sigma_fraction = self.cem_min_sigma_fraction
        if self.branch_sigma is not None:
            sampler.std_reset.assign(np.asarray(self.branch_sigma, np.float32).reshape(var.batch_size, width))
            sampler.reset()
        return sampler


class CEMPopulationSolver(PopulationSolver):
    def _launch_select(self, scores, width_in: int, width_out: int, out, stable: bool = False):
        # Original tiled selection requests one CUDA thread per candidate. A
        # 2048-proposal CEM ends on those draws (MPPI ends on its single mean),
        # so use Robokit's existing loop kernel above CUDA's 1024-thread limit.
        if width_in > 1024:
            from robokit.opt.population_solver import _select_best_k_kernel
            wp.launch(_select_best_k_kernel, dim=self.batch_size,
                inputs=[scores, width_in, out], device=self.device)
        else:
            super()._launch_select(scores, width_in, width_out, out, stable)
