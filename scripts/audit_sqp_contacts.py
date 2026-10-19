"""Compare the actual AdamW mesh contact block and SQP's surface planner.

Requires the same CUDA environment as run.py. The audit deliberately evaluates
AdamW's rollout to capture its exact subproblem inputs. SQP does not do this
in production. This does not assert equality of alternating closed-loop plans.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forceaware.config import load_retarget_config
from forceaware.runtime import setup_runtime
from forceaware.workflow import _load_runtime_models, _collision_output_layout
from forceaware.targets import build_index, load_sequence, make_window, initial_state
from retarget.fa.solver import optimizer_class, SolverSettings
import comfree_warp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/sqp_grasp_audit')
    parser.add_argument('--frame', type=int, default=23)
    parser.add_argument('--contact-rho', type=float, default=SolverSettings().contact_rho)
    args = parser.parse_args()
    root = ROOT/'data/example'
    cfg = load_retarget_config(root/'sequence', root/'forceaware.yaml', contactaware_dir=root/'contactaware')
    cfg = replace(cfg, optimizer=replace(cfg.optimizer, multistart_count=1, n_iter=1))
    setup_runtime(cfg)
    models = _load_runtime_models(cfg, comfree_warp)
    model = models.planner_cpu
    index = build_index(model)
    body, enabled = _collision_output_layout(models.planner_device)
    sequence = load_sequence(cfg, output_body_ids=body, enabled_output_indices=enabled)
    settings = SolverSettings(hand_solver='adamw', contact_rho=args.contact_rho)
    solver = optimizer_class(settings)(model, models.planner_device, index,
                                      guidance_count=sequence.contact.mask.shape[1], cfg=cfg)
    q, v = initial_state(model, index, sequence, start_frame=args.frame, ref_dt=cfg.simulator.time_grid.ref_dt)
    targets = make_window(sequence, executed_steps=0, horizon=solver.backend.settings.horizon,
                          time_grid=cfg.simulator.time_grid, start_frame=args.frame,
                          end_frame=args.frame, hold_frame=-1)
    solver.backend._prepare_window(q, np.zeros_like(v), targets,
        warm=np.zeros(index.action_dim*(solver.backend.settings.horizon+1), np.float32),
        previous_action=q[index.hand_qpos], seed=0, start_time=0)
    raw = solver.backend.control.raw.numpy()
    variables = raw[:, index.action_dim:].astype(float)/solver.variable_scale
    solver._forward_variables = None
    solver.evaluations = solver.forward_calls = solver.backward_calls = 0
    solver.backward_reuses = solver.value_reuses = 0
    captured = []
    original = solver.contact_explorer.solve
    def capture(*a, **kw):
        captured.append(np.array(a[2], copy=True))
        return original(*a, **kw)
    solver.contact_explorer.solve = capture
    solver._contact_block(variables)
    solver.contact_explorer.solve = original
    adamw = np.stack([p.c.numpy()[0] for p in solver.objective.points])
    supplied = []
    cursor = 0
    workspaces = (*solver.objective.training.immediate, solver.objective.training.terminal)
    for workspace in workspaces:
        job = workspace.job
        weight = job.target.query_mask.numpy()*job.target.query_weight.numpy()
        if job.query_enabled and np.any(weight > 0):
            supplied.append(captured[cursor]); cursor += 1
        else:
            supplied.append(np.zeros_like(adamw[0]))
    matched = solver._shared_contact_plan(q, targets, hand_points_object=np.asarray(supplied)).points
    measured, _ = solver._optimize_direct_contacts(q, targets)
    # Complete an actual AdamW alternating window from the same measured state.
    plan = solver.solve(q, np.zeros_like(v), targets,
        warm=np.zeros(index.action_dim*(solver.backend.settings.horizon+1), np.float32),
        previous_action=q[index.hand_qpos], allow_init=False, seed=0,
        execution_steps=cfg.simulator.time_grid.action_substeps, start_time=0)
    final_adamw = np.stack([p.c.numpy()[0] for p in solver.objective.points])
    report = dict(frame=args.frame, contact_rho=args.contact_rho,
        comparison='actual AdamW first mesh contact block vs SQP helper, exactly equal inputs',
        same_input_max_delta_mm=float(np.max(np.linalg.norm(adamw-matched, axis=-1)))*1000,
        same_input_equal=bool(np.array_equal(adamw, matched)),
        measured_vs_rollout_max_delta_mm=float(np.max(np.linalg.norm(adamw-measured, axis=-1)))*1000,
        final_adamw_vs_sqp_max_delta_mm=float(np.max(np.linalg.norm(final_adamw-measured, axis=-1)))*1000,
        final_adamw_vs_sqp_equal=bool(np.array_equal(final_adamw, measured)),
        adamw_contact_rounds=settings.contact_rounds,
        shape=list(matched.shape), surface_solver='bounded_mesh_wrench_search',
        sqp_runtime_uses_forward_comfree_rollout=False, full_alternating_outputs_equal=False)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'mesh_contacts.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.output/'mesh_contacts.npz', adamw=adamw,
                        sqp_same_input=matched, sqp_measured_input=measured, rollout_x=supplied,
                        adamw_final_after_alternation=final_adamw)
    print(json.dumps(report, indent=2))
    if not report['same_input_equal']:
        raise AssertionError('Contact subproblem parity failed')


if __name__ == '__main__':
    main()
