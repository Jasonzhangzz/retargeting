#!/usr/bin/env python3
"""Localize full-horizon derivative sensitivity to individual physics stages."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import warp as wp

from retarget.humanoid import HumanoidProblem, Settings
from retarget.humanoid_mjwarp import MJWarpHumanoid
from retarget.humanoid_terms import TERMS, stack_host
from run_humanoid import task_paths


def audit(task, output):
    scene, reference = task_paths(ROOT/'data/spider', task)
    problem = HumanoidProblem(scene, reference, settings=Settings(multistarts=1, trajectory_cache=False))
    backend = MJWarpHumanoid(problem)
    backend.prepare(problem.initial_state(0), 1, problem.aref[0])
    backend.targets.penalty.zero_()
    x = np.zeros(backend.variables.shape)
    value, gradient = backend.evaluate(x, True)
    direction = gradient/np.linalg.norm(gradient)
    baseline = dict(value=float(value[0]), coordinate_gradient_directional=float(np.sum(gradient*direction)))
    weights = backend.model.weights.numpy().copy()
    terms = []
    # Hold direction fixed so all contributions add to the same derivative.
    for j, name in enumerate(TERMS[:-2]):
        w = np.zeros_like(weights); w[j] = weights[j]
        backend.model.weights.assign(w); backend.dirty = True
        val, g = backend.evaluate(x, True)
        checks = []
        for eps in (.0001, .001, .01, .05):
            plus = backend.evaluate(x+eps*direction, False)[0][0]
            minus = backend.evaluate(x-eps*direction, False)[0][0]
            checks.append(dict(epsilon=eps, finite_difference=float((plus-minus)/(2*eps))))
        terms.append(dict(term=name, value=float(val[0]), coordinate_gradient_directional=float(np.sum(g*direction)), checks=checks))
    backend.model.weights.assign(weights); backend.dirty = True
    backend.evaluate(x, True)
    backend.diagnose()
    baseline_terms = stack_host([s.job.terms for s in backend.stages])[:, 0]
    probes = []
    for eps in (.0001, .001, .01, .05):
        states, losses, forces, actives = [], [], [], []
        for sign in (1, -1):
            backend.evaluate(x+sign*eps*direction, False)
            states.append(stack_host([s.job.qpos for s in backend.stages])[:, 0])
            losses.append(stack_host([s.job.loss for s in backend.stages])[:, 0])
            forces.append(stack_host([s.job.constraint_force for s in backend.stages])[:, 0])
            actives.append(stack_host([s.job.active for s in backend.stages])[:, 0])
        probes.append(dict(epsilon=eps, finite_difference=((losses[0]-losses[1])/(2*eps)).tolist(),
            root_position_delta_m=np.linalg.norm(states[0][:,:3]-states[1][:,:3], axis=1).tolist(),
            constraint_force_delta_norm=np.linalg.norm(forces[0]-forces[1], axis=1).tolist(),
            contact_active_set_difference_count=np.sum(actives[0]!=actives[1], axis=1).tolist()))
    rows = [dict(stage=i, time=i*backend.model.dt, finite_difference=probes[2]['finite_difference'][i], baseline_terms=dict(zip(TERMS, baseline_terms[i].tolist())))
            for i in range(len(backend.stages))]
    output.mkdir(parents=True, exist_ok=True)
    result = dict(task=task, window=0, horizon_seconds=.8, consensus_penalty=False,
                  baseline=baseline, terms=terms, stages=rows, probes=probes,
                  note='Unoptimized reference control, exact initial reference state; this isolates dynamics sensitivity before ADMM, rather than reproducing its contact-force update.')
    (output/f'{task}_horizon.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(task=task, baseline=baseline, terms=terms), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['humanoid', 'humanoid-object', 'all'], default='all')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/humanoid/step_audit')
    args = parser.parse_args()
    for task in (['humanoid', 'humanoid-object'] if args.task == 'all' else [args.task]):
        audit(task, args.output)
