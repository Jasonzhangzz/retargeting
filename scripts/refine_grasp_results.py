#!/usr/bin/env python3
"""Compare static CARD CEM and AdamW using saved original grasp inputs."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from retarget.grasp import StaticHand, build_scene
    from retarget.grasp_initialization import pose_in_hand
    from retarget.grasp_audit import GraspAudit
    from retarget.grasp_runtime import GraspRuntime
    from retarget.grasp_device import solve_nodes
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'reports/grasp_validation_20261028/background_results/summary.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--types', type=int, nargs='+')
    parser.add_argument('--scales', type=float, nargs='+')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--methods', choices=['cem', 'adamw'], nargs='+', default=['cem'])
    parser.add_argument('--gpu-nodes', type=int, default=4)
    parser.add_argument('--population', type=int, default=1024)
    parser.add_argument('--generations', type=int, default=16)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--explicit-pose', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--explicit-horizon', type=int, default=4)
    parser.add_argument('--explicit-refinement-steps', type=int, default=8)
    parser.add_argument('--audit', action='store_true', help='Independently validate saved FK, mesh and closure after solving')
    opts = parser.parse_args()
    if opts.gpu_nodes < 1 or opts.population < 2 or opts.generations < 1 or opts.rounds < 1:
        parser.error('Positive node/generation/round counts and population >= 2 required')
    source = json.loads(opts.source.read_text())
    rows = [r for r in source if (not opts.types or int(r['template'].split('_')[0]) in opts.types)
            and (not opts.scales or any(np.isclose(r['primitive']['radial_scale'], s) for s in opts.scales))]
    if opts.limit:
        rows = rows[:opts.limit]
    tasks = [(method, row) for row in rows for method in opts.methods]
    opts.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    results = {method: [] for method in opts.methods}

    def save_status(status):
        count = sum(map(len, results.values()))
        elapsed = perf_counter()-started
        (opts.output/'status.json').write_text(json.dumps(dict(status=status, completed=count,
            total=len(tasks), wall_seconds=elapsed,
            estimated_remaining_seconds=(elapsed/count*(len(tasks)-count) if count else None)), indent=2))

    save_status('running')
    for offset in range(0, len(tasks), opts.gpu_nodes):
        prepared = []
        for method, row in tasks[offset:offset+opts.gpu_nodes]:
            begun = perf_counter()
            old = Path(row['case'])
            report = json.loads((old/'result.json').read_text())
            args = SimpleNamespace(**report['settings'])
            args.solver_config = Path(args.solver_config)
            args.grasp_optimizer = 'adamw' if method == 'adamw' else 'cem'
            args.explicit_pose = bool(method == 'cem' and opts.explicit_pose)
            from retarget.grasp_explicit import DEFAULTS as explicit_defaults
            for key, value in explicit_defaults.items():
                setattr(args, 'explicit_'+key, getattr(args, 'explicit_'+key, value))
            args.explicit_horizon = opts.explicit_horizon
            args.explicit_refinement_steps = opts.explicit_refinement_steps
            args.cem_population = opts.population
            args.cem_iterations = opts.generations
            # Saved settings may describe the removed dynamics CEM. Use CARD defaults.
            for key, value in dict(cem_elite_fraction=.05, cem_temperature=.1,
                    cem_translation_sigma_mm=2., cem_rotation_sigma=.016,
                    cem_joint_sigma=.05, cem_final_noise=.2).items():
                setattr(args, key, value)
            for key in ('cem_gradient_steps', 'cem_gradient_lr', 'cem_optimize_pose',
                        'cem_sigma', 'cem_min_sigma', 'cem_trust_radius', 'contact_depth_mm',
                        'grasp_penetration_weight', 'grasp_endpoint_weight'):
                if hasattr(args, key):
                    delattr(args, key)
            args.contact_rounds = opts.rounds
            args.hand_model = report.get('hand_model', 'wuji1')
            model, surface, _, _ = build_scene(report['object'], report['scale'], args.hand_model)
            with np.load(old/'grasp.npz', allow_pickle=False) as saved:
                arrays = {k: saved[k].copy() for k in saved.files}
            hand = StaticHand(model, surface, arrays['hand_body_ids'],
                              arrays.get('initial_hand_local_points', arrays['hand_local_points']),
                              arrays.get('initial_hand_local_normals', arrays['hand_local_normals']))
            start = report['selected']['start']
            case = opts.output/method/f'case_{offset+len(prepared):04d}'/row['template']
            case.mkdir(parents=True)
            # Build a new experiment so originals (including runtime_scene.xml) stay untouched.
            from retarget.grasp_initialization import GraspInitialization
            from retarget.grasp_runtime import prepare_experiment
            guess = GraspInitialization(arrays['seed_qpos'].copy(),
                arrays['initial_object_points'].copy(), arrays['initial_object_normals'].copy(),
                {k[len('dexonomy_'):]: v for k,v in arrays.items() if k.startswith('dexonomy_')},
                copy.deepcopy(report['starts'][start]['initialization']))
            runtime = GraspRuntime(prepare_experiment(case/f'input_{start:02d}', hand, guess, args), args)
            runtime.audit_context = (GraspAudit(surface, args.init_device), hand, args)
            prepared.append((method, row, report, runtime, arrays, case, begun))
        plans = solve_nodes([p[3] for p in prepared])
        for (method, row, report, runtime, arrays, case, begun), (plan, snap, qpos) in zip(prepared, plans):
            audit, realized = runtime.device_audit_result
            obj = qpos[runtime.index.object_qpos:runtime.index.object_qpos+7]
            root = qpos[runtime.index.hand_qpos]
            rotation = Rotation.from_quat(obj[3:], scalar_first=True).inv()
            canonical = np.r_[rotation.apply(root[:3]-obj[:3]),
                (rotation*Rotation.from_euler('XYZ', root[3:6])).as_euler('XYZ'), root[6:]]
            arrays.update(qpos=qpos, hand_qpos=canonical, qvel=plan.next_qvel,
                hold_action=plan.action, initial_qpos=runtime.qpos.copy(),
                optimized_initial_qpos=plan.initial_qpos,
                predicted_terminal_qpos=runtime.predicted_terminal_qpos,
                final_object_pose_in_hand=pose_in_hand(canonical),
                object_points=snap['z'][0], dual=snap['dual'][0],
                hand_local_points=snap.get('hand_local_points', arrays['hand_local_points']),
                hand_local_normals=snap.get('hand_local_normals', arrays['hand_local_normals']), **realized)
            np.savez_compressed(case/'grasp.npz', **arrays)
            np.savez_compressed(case/'contacts.npz', **snap)
            (case/'scene.xml').write_text(runtime.scene_xml)
            selected = copy.deepcopy(report['selected'])
            selected.update(audit, seconds=plan.seconds, solver_diagnostics=runtime.optimizer.last_solver_diag,
                rounds=0 if method == 'cem' else opts.rounds,
                output_time_seconds=runtime.output_step*runtime.cfg.simulator.time_grid.mpc_dt,
                prediction_horizon_seconds=getattr(runtime, 'prediction_horizon_seconds',
                    len(runtime.optimizer.backend.steps)*runtime.cfg.simulator.time_grid.mpc_dt),
                termination=runtime.optimizer.last_solver_diag.get('termination', 'single_shared_solver_window'),
                object_shift_mm=float(np.linalg.norm(obj[:3]-runtime.qpos[runtime.index.object_qpos:runtime.index.object_qpos+3])*1000),
                predicted_terminal_object_shift_mm=float(np.linalg.norm(
                    runtime.predicted_terminal_qpos[runtime.index.object_qpos:runtime.index.object_qpos+3]
                    -runtime.qpos[runtime.index.object_qpos:runtime.index.object_qpos+3])*1000))
            timing = dict(total_seconds=perf_counter()-begun, setup_seconds=runtime.setup_seconds,
                solve_seconds=plan.seconds, graph_prepare_seconds=runtime.optimizer.last_solver_diag['graph_prepare_seconds'])
            settings = {k: str(v) if isinstance(v, Path) else v for k,v in vars(runtime.args).items()}
            updated = dict(report, selected=selected, starts=[selected], settings=settings, timings=timing,
                solver=runtime.optimizer.last_solver_diag['solver'], paired_input_source=row['case'],
                output_pose=(('static grasp configuration; explicit model prediction saved separately' if args.explicit_pose
                              else 'static CARD configuration; no execution or prediction') if method == 'cem'
                             else 'next executed interval endpoint'),
                generated_at=datetime.now().isoformat())
            (case/'result.json').write_text(json.dumps(updated, indent=2))
            new = dict(row, case=str(case.resolve()), **{k: v for k,v in selected.items() if k not in ('case',)}, timings=timing)
            results[method].append(new)
            temporary = opts.output/method/'summary.pending.json'
            temporary.write_text(json.dumps(results[method], indent=2))
            temporary.replace(opts.output/method/'summary.json')
            print(json.dumps(dict(method=method, template=row['template'], primitive=row['primitive'],
                penetration_mm=audit['max_object_penetration_mm'], self_mm=audit['max_self_penetration_mm'],
                force_closure=audit['force_closure']['force_closure'], feasible=audit['feasible'],
                gpu_seconds=plan.seconds, case=str(case))), flush=True)
        save_status('running')
        # Explicitly release each heterogeneous group's graphs and workspaces.
        del prepared, plans, runtime, plan
        import gc
        gc.collect()
    aggregate = {method: dict(cases=len(items), feasible=sum(r['feasible'] for r in items),
        closure=sum(r['force_closure']['force_closure'] for r in items),
        penetration_over_2mm=sum(r['max_object_penetration_mm'] > 2 for r in items),
        self_over_2mm=sum(r['max_self_penetration_mm'] > 2 for r in items),
        mean_solve_seconds=float(np.mean([r['seconds'] for r in items]))) for method,items in results.items()}
    (opts.output/'metrics.json').write_text(json.dumps(aggregate, indent=2))
    if opts.audit:
        from scripts.validate_grasp_matrix import audit_batch
        save_status('auditing')
        for method in opts.methods:
            audited = audit_batch(opts.output/method)
            (opts.output/method/'cases.json').write_text(json.dumps(audited, indent=2))
            validation = dict(cases=len(audited), valid_exports=sum(r.get('saved_data_valid', False) for r in audited),
                closure_disagreements=sum(not r.get('closure_agrees', False) for r in audited))
            (opts.output/method/'validation.json').write_text(json.dumps(validation, indent=2))
    save_status('complete')
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if '--output' in sys.argv:
            path = Path(sys.argv[sys.argv.index('--output')+1])/'status.json'
            if path.exists():
                status = json.loads(path.read_text())
                status.update(status='failed', error=repr(error))
                path.write_text(json.dumps(status, indent=2))
        raise
