#!/usr/bin/env python3
"""Probe additional optimization with state, controls and contacts held fixed."""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from retarget.fa.adamw import AdamW
from retarget.humanoid import HumanoidProblem, ProjectionADMM, Settings, Weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--frame', type=int, default=40)
    parser.add_argument('--iterations', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.case/'config.json').read_text())
    with np.load(args.case/'rollout.npz') as saved:
        arrays = {k: saved[k] for k in saved.files}
    settings = Settings(**{k: v for k, v in config['settings'].items()
                           if k in {f.name for f in fields(Settings)}})
    p = HumanoidProblem(config['scene'], config['reference'], ref_dt=config['ref_dt'],
                        settings=settings, weights=Weights(**config['weights']))
    solver = ProjectionADMM(p)
    original_update = solver.action_optimizer.update
    record = {}

    def probe(x, evaluate, **kwargs):
        result = original_update(x, evaluate, **kwargs)
        if int(solver.backend.frame_times[0]+1) != args.frame:
            return result
        value, gradient = evaluate(result[0], True)
        selected = int(np.argmin(value))
        optimizer = AdamW(learning_rate=settings.adamw_lr,
                          weight_decay=settings.adamw_weight_decay,
                          device=solver.backend.device)
        extra_kwargs = {**kwargs, 'iterations': args.iterations, 'continue_window': False}
        extra, _, diagnostic = optimizer.update(result[0], evaluate, **extra_kwargs)
        value_after, gradient_after = evaluate(extra, True)
        record.update(frame=args.frame, extra_iterations=args.iterations,
                      original_selected=selected, cost_before=value.tolist(),
                      cost_after=value_after.tolist(),
                      gradient_norm_before=np.linalg.norm(gradient, axis=1).tolist(),
                      gradient_norm_after=np.linalg.norm(gradient_after, axis=1).tolist(),
                      fixed_state_contacts_dual_forces_weights=True,
                      probe_exported_to_execution=False)
        evaluate(result[0], False)
        return result

    solver.action_optimizer.update = probe
    for window, first in enumerate(arrays['reference_frames']):
        if first > args.frame:
            break
        state = mujoco.MjData(p.model)
        state.qpos[:], state.qvel[:] = arrays['qpos'][window], arrays['qvel'][window]
        state.time = window*p.ref_dt
        mujoco.mj_forward(p.model, state)
        previous = p.aref[first-1] if window == 0 else arrays['ctrl'][window-1]
        rollout, _ = solver.solve(state, [first], previous)
        np.testing.assert_array_equal(rollout['next_state'].qpos, arrays['qpos'][window+1])
        np.testing.assert_array_equal(solver.actions[0], arrays['ctrl'][window])
    if not record:
        raise ValueError('Frame not found')
    record['executed_prefix_qpos_and_controls_match_exactly'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
