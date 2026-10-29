#!/usr/bin/env python3
"""Decompose fixed-window mjwarp gradients and rescore with pinned SPIDER pose cost.

This is a diagnostic: it leaves the production objective and optimizer intact.
SPIDER scoring uses the SAME mjwarp states/targets, not a new physics rollout.
"""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from compare_spider_humanoid import source_functions
from retarget.humanoid import HumanoidProblem, ProjectionADMM, Settings, Weights
from retarget.humanoid_terms import stack_host


def spider_pose_cost(qpos, reference):
    source = source_functions()
    cfg = SimpleNamespace(embodiment_type='humanoid_object', device='cpu', nv=41, nu=29,
                          base_pos_rew_scale=1., base_rot_rew_scale=1., joint_rew_scale=1.,
                          pos_rew_scale=1., rot_rew_scale=.3)
    qpos, reference = np.broadcast_arrays(qpos, reference)
    error = source['_diff_qpos'](cfg, torch.as_tensor(qpos.reshape(-1, 43).copy(), dtype=torch.float64),
                                 torch.as_tensor(reference.reshape(-1, 43).copy(), dtype=torch.float64))
    weights = source['_weight_diff_qpos'](cfg).double()
    weighted = (error * weights).numpy().reshape(qpos.shape[:-1] + (41,))
    parts = {name: np.sum(weighted[..., part] ** 2, axis=-1)
             for name, part in [('root_position', slice(0, 3)), ('root_rotation', slice(3, 6)),
                                ('joints', slice(6, 35)), ('box_position', slice(35, 38)),
                                ('box_rotation', slice(38, 41))]}
    return np.sqrt(sum(parts.values())), parts


def cosine(first, second):
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / denominator) if denominator > 0 else None


def analyze(case, output, checkpoints):
    config = json.loads((case / 'config.json').read_text())
    with np.load(case / 'rollout.npz') as data:
        saved = {key: data[key] for key in data.files}
    settings = Settings(**{k: v for k, v in config['settings'].items()
                           if k in {f.name for f in fields(Settings)}})
    p = HumanoidProblem(config['scene'], config['reference'], ref_dt=config['ref_dt'],
                        settings=settings, weights=Weights(**config['weights']))
    solver = ProjectionADMM(p)
    b = solver.backend
    records = []
    original_update = solver.action_optimizer.update
    pending = {}

    def costs(x):
        value, _ = b.evaluate(x, False)
        b.diagnose()
        from retarget.humanoid_terms import TERMS
        terms = stack_host([stage.job.terms for stage in b.stages]).sum(axis=0)
        q = stack_host([b.q0] + [rt.recorded.result.qpos for rt in b.steps])
        source, _ = spider_pose_cost(q[1:], b.targets.qpos.numpy()[1:, None, :])
        return dict(total=value.tolist(), terms=dict(zip(TERMS, terms.T.tolist())),
                    spider_pose_mean=source.mean(axis=0).tolist())

    def instrumented(x, evaluate, **kwargs):
        first = int(b.frame_times[0] + 1)
        if first not in checkpoints:
            return original_update(x, evaluate, **kwargs)
        weights, penalty = b.model.weights.numpy().copy(), b.targets.penalty.numpy().copy()
        before = costs(x)
        _, total_gradient = b.evaluate(x, True)
        gradients = {}
        blocks = dict(root_pose=[0, 1], joints_velocity=[2, 3], keypoints=[4, 5],
                      upright=[6], balance=[7], regularizers=[8, 9, 10, 11], wrench=[12], hand_position=[13], consensus=[])
        try:
            for name, ids in blocks.items():
                subset = np.zeros_like(weights)
                subset[ids] = weights[ids]
                b.model.weights.assign(subset)
                b.targets.penalty.assign(penalty * [1, 0] if name == 'consensus' else np.zeros_like(penalty))
                b.dirty = True
                _, gradients[name] = b.evaluate(x, True)
        finally:
            b.model.weights.assign(weights)
            b.targets.penalty.assign(penalty)
            b.dirty = True
        summed = sum(gradients.values())
        relative_error = np.linalg.norm(summed - total_gradient) / max(np.linalg.norm(total_gradient), 1.e-12)
        if relative_error > .005:
            raise AssertionError(f'Gradient decomposition mismatch: {relative_error}')
        # Restore the ordinary callback and allow its two AdamW updates.
        result = original_update(x, evaluate, **kwargs)
        after = costs(result[0])
        pending[first] = dict(frame=first, before=before, after=after, gradients=gradients,
                              total_gradient=total_gradient, delta=result[0] - x,
                              gradient_decomposition_relative_error=float(relative_error))
        return result

    solver.action_optimizer.update = instrumented
    for window, first in enumerate(saved['reference_frames']):
        if first > max(checkpoints):
            break
        state = mujoco.MjData(p.model)
        state.qpos[:], state.qvel[:] = saved['qpos'][window], saved['qvel'][window]
        state.time = window * p.ref_dt
        mujoco.mj_forward(p.model, state)
        previous = p.aref[first - 1] if window == 0 else saved['ctrl'][window - 1]
        rollout, history = solver.solve(state, [first], previous)
        np.testing.assert_array_equal(rollout['next_state'].qpos, saved['qpos'][window + 1])
        np.testing.assert_array_equal(solver.actions[0], saved['ctrl'][window])
        if int(first) in pending:
            entry = pending.pop(int(first))
            selected = history[-1]['selected']
            gradients = {name: values[selected] for name, values in entry.pop('gradients').items()}
            gradient, delta = entry.pop('total_gradient')[selected], entry.pop('delta')[selected]
            for phase in ('before', 'after'):
                entry[phase] = {key: ({name: values[selected] for name, values in value.items()}
                                     if isinstance(value, dict) else value[selected])
                                for key, value in entry[phase].items()}
            entry.update(selected=selected, total_gradient_norm=float(np.linalg.norm(gradient)),
                parameter_delta_norm=float(np.linalg.norm(delta)),
                gradients={name: dict(norm=float(np.linalg.norm(value)),
                    cosine_with_total=cosine(value, gradient),
                    dot_actual_step=float(value @ delta)) for name, value in gradients.items()},
                pairwise_cosines={name + '__' + other: cosine(value, gradients[other])
                                  for name, value in gradients.items() for other in gradients if name < other})
            records.append(entry)
            output.mkdir(parents=True, exist_ok=True)
            (output / 'fixed_window_gradients.json').write_text(json.dumps(records, indent=2) + '\n')
            print('frame', first, 'original prefix matched exactly, gradient decomposition error',
                  entry['gradient_decomposition_relative_error'], flush=True)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoints', type=int, nargs='+', default=[1, 20, 40, 80, 110])
    args = parser.parse_args()
    analyze(args.case, args.output, args.checkpoints)
