#!/usr/bin/env python3
"""Run SPIDER's unmodified sampling update on the current mjwarp objective.

Only the rollout callback is adapted. MJWarp knots, contacts, dynamics, MPC
period and scalar cost remain unchanged. This is not an MJWarp reproduction.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/spider/source'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))

import numpy as np
import torch
from spider.config import Config, compute_noise_schedule
from spider.optimizers.sampling import make_optimize_fn, make_optimize_once_fn
from retarget.humanoid import HumanoidProblem, ProjectionADMM, Settings, run
from run_humanoid import task_paths


class SpiderMJWarpAdapter:
    """Use upstream sampling, top-decile weights, annealing and early stopping."""
    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.persist = False
        self.optimize = make_optimize_fn(make_optimize_once_fn(self.rollout))

    def rollout(self, config, env, controls, ref_slice, env_param):
        b = self.backend
        flat = controls.reshape(config.num_samples, -1)
        scores = torch.empty(config.num_samples, device=config.device, dtype=controls.dtype)
        for start in range(0, config.num_samples, b.worlds):
            count = min(b.worlds, config.num_samples - start)
            batch = flat[0:1].expand(b.worlds, -1).clone()
            batch[:count] = flat[start:start+count]
            values, _ = b.evaluate(batch, False)
            scores[start:start+count] = values[:count]
        # MJWarp objective is already averaged over the full prediction horizon.
        # terminate_resample=False in the upstream config; terminate is unused.
        terminate = torch.zeros(config.num_samples, device=config.device, dtype=torch.bool)
        return controls, -scores, terminate, {'mjwarp_cost': scores}

    def update(self, variables, evaluate, *, iterations, horizon, advance,
               continue_window=False, evaluate_on_device=False):
        b, c = self.backend, self.config
        assert evaluate_on_device and iterations == c.max_num_iterations
        x = torch.as_tensor(variables, device=c.device, dtype=torch.float64)
        if not bool((x == x[0]).all()):
            raise ValueError('Upstream sampling requires one common nominal')
        with torch.no_grad():
            before, _ = evaluate(x, False)
            if not torch.allclose(before, before[:1].expand_as(before), rtol=1e-5, atol=1e-5):
                raise ValueError('Samples must share a fixed contact objective')
            controls, info = self.optimize(c, None, x[0].reshape(horizon, -1), (),)
            exported = controls.reshape(1, -1).expand_as(x).clone()
            values, _ = evaluate(exported, False)
        updates = int(info['opt_steps'][0])
        return exported.cpu().numpy(), int(values[0] < before[0]), dict(
            method='SPIDER_source_sampling_mjwarp_rollout', iterations=updates,
            max_iterations=c.max_num_iterations, samples_per_iteration=c.num_samples,
            sampled_trajectories=c.num_samples*updates, gradient_evaluations=0,
            value_before=float(before[0]), value_after=float(values[0]),
            final_values=values.cpu().tolist(), selected_candidate=0,
            full_objective_safeguard=False, export='upstream_top_decile_weighted_mean',
            upstream_info={k: np.asarray(v).tolist() for k, v in info.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['humanoid', 'humanoid-object'], required=True)
    parser.add_argument('--samples', type=int, default=2048)
    parser.add_argument('--iterations', type=int, default=16)
    parser.add_argument('--worlds', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--frames', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 20 or args.iterations < 1 or args.worlds < 1:
        parser.error('Use >=20 samples, positive iterations and worlds')
    settings = Settings(action_solver='mppi', multistarts=args.worlds,
                        action_iterations=args.iterations, mppi_samples=args.samples,
                        mppi_temperature=.6 if args.task == 'humanoid' else 1.,
                        mppi_update='weighted-mean',
                        seed=args.seed)
    scene, reference = task_paths(ROOT/'data/spider', args.task)
    problem = HumanoidProblem(scene, reference, settings=settings)
    c = Config(device=settings.device, embodiment_type='humanoid' if args.task == 'humanoid' else 'humanoid_object',
               nu=problem.model.nu, horizon=.8, knot_dt=.1, knot_steps=1,
               num_samples=args.samples, max_num_iterations=args.iterations,
               temperature=.6 if args.task == 'humanoid' else 1.,
               joint_noise_scale=.1, first_ctrl_noise_scale=.5, last_ctrl_noise_scale=1.,
               final_noise_scale=.1, use_torch_compile=False,
               env_params_list=[[{}] for _ in range(args.iterations)],
               num_trace_uniform_samples=0, num_trace_topk_samples=0)
    compute_noise_schedule(c)
    # MJWarp parameters are normalized by control_scale. Sampling physical
    # radians with upstream noise thus requires converting to these units.
    c.noise_scale /= settings.control_scale
    torch.manual_seed(args.seed)

    def factory(p):
        solver = ProjectionADMM(p)
        solver.action_optimizer = SpiderMJWarpAdapter(solver.backend, c)
        solver.action_solver_name = 'SPIDER_source_sampling_with_mjwarp_rollout_callback'
        return solver

    run(problem, args.output, frames=args.frames, solver_factory=factory)
    provenance = dict(
        source_sha256={str(path.relative_to(SOURCE)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [SOURCE/'spider/optimizers/sampling.py', SOURCE/'spider/config.py', SOURCE/'spider/interp.py']},
        optimizer='unmodified upstream make_optimize_fn(make_optimize_once_fn(mjwarp_callback))',
        samples=args.samples, max_iterations=args.iterations, seed=args.seed,
        temperature=c.temperature, improvement_threshold=c.improvement_threshold,
        beta_traj=c.beta_traj, exploit_ratio=c.exploit_ratio,
        first_knot_noise_rad=.05, last_knot_noise_rad=.1,
        differences_from_full_spider=['MJWarp dynamics and current v7 cost with contact ADMM',
                                     'mjwarp normalized 8 knots and mjwarp interpolation, no upstream dense control interpolation',
                                     'mjwarp 1/30 s MPC replanning; source uses 0.1 s',
                                     'no domain randomization; terminate_resample=False as upstream default',
                                     'upstream stopping threshold in current objective units'])
    (args.output/'spider_adapter.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
