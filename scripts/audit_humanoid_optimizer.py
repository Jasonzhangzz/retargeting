#!/usr/bin/env python3
"""Trace actual MPC action gradients, including full-length failing windows.

Extra finite differences are diagnostics only. After each probe, the exact
original action and its MJWarp state are rebuilt before AdamW continues.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import torch

from retarget.humanoid import HumanoidProblem, ProjectionADMM, Settings
from run_humanoid import task_paths


def host(value):
    return value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else np.array(value, copy=True)


def audit(task, output, frames, action_iterations, admm_iterations, learning_rate):
    scene, reference = task_paths(ROOT/'data/spider', task)
    settings = Settings(action_iterations=action_iterations, admm_iterations=admm_iterations, adamw_lr=learning_rate)
    problem = HumanoidProblem(scene, reference, settings=settings)
    solver = ProjectionADMM(problem)
    backend = solver.backend
    evaluate = backend.evaluate
    trace, parameters, gradients, controls = [], [], [], []
    context = {}

    def checked_evaluate(x, gradient):
        value, g = evaluate(x, gradient)
        if not gradient:
            return value, g
        xx, gg, ff = host(x), host(g), host(value)
        norms = np.linalg.norm(gg, axis=1)
        direction = gg/np.maximum(norms[:, None], 1e-30)
        checks = []
        for eps in (.001, .01, .05):
            plus = host(evaluate(xx+eps*direction, False)[0])
            minus = host(evaluate(xx-eps*direction, False)[0])
            difference = (plus-minus)/(2*eps)
            checks.append(dict(epsilon_normalized=eps, coordinate_gradient=norms.tolist(), finite_difference=difference.tolist(),
                               relative_error=(np.abs(difference-norms)/np.maximum(np.maximum(np.abs(difference), norms), 1e-9)).tolist()))
        restored_value, restored_gradient = evaluate(x, True)
        np.testing.assert_allclose(host(restored_value),ff,rtol=2e-6,atol=3e-6)
        assert np.isfinite(host(restored_gradient)).all()
        parameters.append(xx); gradients.append(gg)
        trace.append(dict(**context, callback=len(trace), value=ff.tolist(), gradient_norm=norms.tolist(), checks=checks))
        return restored_value, restored_gradient

    backend.evaluate = checked_evaluate
    state = problem.initial_state(0)
    previous = problem.aref[0].copy()
    qs, vs, histories = [state.qpos.copy()], [state.qvel.copy()], []
    count = len(problem.qref)-1 if frames is None else min(frames, len(problem.qref)-1)
    for window in range(count):
        context.update(window=window, reference_frame=window+1)
        first_callback = len(trace)
        rollout, history = solver.solve(state, [window+1], previous)
        selected = history[-1]['selected']
        for row in trace[first_callback:]:
            row['selected'] = selected
        controls.append(rollout['controls_dense'])
        mujoco.mj_copyData(state, problem.model, rollout['next_state'])
        previous = solver.actions[0].copy()
        qs.append(state.qpos.copy()); vs.append(state.qvel.copy()); histories.append(history)
        print(task, window+1, '/', count, 'height', state.qpos[2], 'cost', history[-1]['action_cost_after'],
              'first gradient FD', [r['relative_error'][selected] for r in trace[first_callback]['checks']], flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output/f'{task}_optimizer.json').write_text(json.dumps(dict(settings=asdict(settings), weights=asdict(problem.weights),
        finite_difference_purpose='validation_only; original workspace restored before returning to AdamW',
        callbacks=trace, histories=histories), indent=2)+'\n')
    np.savez_compressed(output/f'{task}_optimizer.npz', parameters=parameters, gradient=gradients,
                        qpos=qs, qvel=vs, controls=controls)
    selected_errors = np.array([[c['relative_error'][row['selected']] for c in row['checks']] for row in trace])
    gains = np.array([h[-1]['action_cost_before']-h[-1]['action_cost_after'] for h in histories])
    result = dict(task=task, windows=count, callbacks=len(trace),
                  coordinate_fd_relative_error_median=np.median(selected_errors, axis=0).tolist(),
                  coordinate_fd_relative_error_max=np.max(selected_errors, axis=0).tolist(),
                  zero_action_improvement_windows=int(np.sum(gains<=1e-6)),
                  root_height_min_m=float(np.min(np.array(qs)[:, 2])),
                  root_position_rms_m=float(np.sqrt(np.mean(np.sum((np.array(qs)[:,:3]-problem.qref[:count+1,:3])**2, axis=1)))))
    with np.load(reference.parent/'trajectory_mjwp.npz') as released:
        mjwarp_action = np.array(controls).reshape(-1, problem.model.nu)
        source_action = released['ctrl'].reshape(-1, problem.model.nu)[:len(mjwarp_action)]
    action_error = np.sqrt(np.mean((mjwarp_action-source_action)**2, axis=1))
    result['action_comparison_by_recorded_step_ordinal'] = dict(
        rmse_rad=float(np.sqrt(np.mean((mjwarp_action-source_action)**2))), first_step_rmse_rad=float(action_error[0]),
        note='Separately optimized executions: initial CPU pre-step, interpolation, objective and MPC period differ. This comparison does not isolate optimizer quality.')
    with (output/f'{task}_actions.csv').open('w') as f:
        f.write('physics_step,nominal_time_s,action_rmse_rad\n')
        for i, error in enumerate(action_error):
            f.write(f'{i+1},{(i+1)*backend.model.dt:.8f},{error:.8f}\n')
    np.savez_compressed(output/f'{task}_actions.npz', mjwarp_control=mjwarp_action,
                        spider_published_control=source_action, rmse_rad=action_error)
    prior_name = 'runpy_budget_aligned_amass' if task=='humanoid' else 'runpy_budget_aligned_omomo'
    prior = ROOT/'outputs/humanoid'/prior_name
    if (prior/'rollout.npz').exists() and (prior/'execution.npz').exists():
        with np.load(prior/'rollout.npz') as old, np.load(prior/'execution.npz') as execution:
            result['reproduces_previous_failed_run_bitwise'] = bool(
                np.array_equal(qs, old['qpos']) and np.array_equal(controls, execution['ctrl_substeps']))
    (output/f'{task}_optimizer_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['humanoid', 'humanoid-object', 'all'], default='all')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/humanoid/step_audit')
    parser.add_argument('--frames', type=int)
    parser.add_argument('--action-iterations', type=int, default=1)
    parser.add_argument('--admm-iterations', type=int, default=1)
    parser.add_argument('--adamw-lr', type=float, default=.3)
    args = parser.parse_args()
    for task in (['humanoid', 'humanoid-object'] if args.task == 'all' else [args.task]):
        audit(task, args.output, args.frames, args.action_iterations, args.admm_iterations, args.adamw_lr)
