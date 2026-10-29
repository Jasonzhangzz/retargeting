#!/usr/bin/env python3
"""Solve selected rigid-primitive grasps and persist background progress.

Run this in its own process/session. The numerical solver batches independent
CUDA scene branches; initialization and file preparation remain host work.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import threading
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpu-nodes', type=int, default=4)
    parser.add_argument('--hand-model', choices=['wuji2', 'wuji1'], default='wuji2')
    parser.add_argument('--templates', nargs='+', default=['all'], help='Grasp type IDs, full names, or all')
    parser.add_argument('--primary-shape-only', action='store_true', help='One primitive family per type')
    parser.add_argument('--population', type=int, default=1024)
    parser.add_argument('--iterations', type=int, default=32,
                        help='Extra generations allow feasibility restoration before contact refinement')
    parser.add_argument('--self-clearance-mm', type=float, default=.5)
    parser.add_argument('--explicit-pose', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--explicit-horizon', type=int, default=4)
    parser.add_argument('--explicit-refinement-steps', type=int, default=8)
    parser.add_argument('--radius-scales', nargs='+', type=float, default=[.75, 1., 1.25])
    parser.add_argument('--seed', type=int, default=0)
    opts = parser.parse_args()
    run_dir = opts.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    status_file = run_dir/'status.json'
    if status_file.exists() or (run_dir/'results').exists():
        parser.error('Use a new run directory; existing results are never overwritten')
    argv = ['--hand-model', opts.hand_model, '--primitives', '--templates', *opts.templates, '--radius-scales',
        *map(str, opts.radius_scales), '--height-scales', '1', '--gpu-nodes', str(opts.gpu_nodes),
        '--grasp-optimizer', 'cem', '--cem-population', str(opts.population),
        '--cem-iterations', str(opts.iterations), '--cem-self-clearance-mm', str(opts.self_clearance_mm),
        '--seed', str(opts.seed), '--no-viewer', '--no-render', '--output', str(run_dir), '--name', 'results']
    argv += ['--explicit-pose' if opts.explicit_pose else '--no-explicit-pose',
             '--explicit-horizon', str(opts.explicit_horizon),
             '--explicit-refinement-steps', str(opts.explicit_refinement_steps)]
    if opts.primary_shape_only:
        argv.append('--primary-shape-only')
    started = time.monotonic()
    lock, stop = threading.Lock(), threading.Event()
    status = dict(status='initializing', pid=os.getpid(),
        started_at=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
        command=[sys.executable, str(ROOT/'run_grasp.py'), *argv],
        completed=0, total=None, hand_model=opts.hand_model, gpu_nodes=opts.gpu_nodes, population=opts.population,
        iterations=opts.iterations, self_clearance_mm=opts.self_clearance_mm,
        explicit_pose=opts.explicit_pose, explicit_horizon=opts.explicit_horizon,
        explicit_refinement_steps=opts.explicit_refinement_steps,
        templates=opts.templates, primary_shape_only=opts.primary_shape_only,
        radius_scales=opts.radius_scales)
    cache = {}

    def update(**fields):
        with lock:
            status.update(fields, elapsed_seconds=time.monotonic()-started,
                          updated_at=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat())
            temporary = run_dir/'status.pending.json'
            temporary.write_text(json.dumps(status, indent=2))
            temporary.replace(status_file)

    def progress():
        rows = {}
        compact = run_dir/'results/grasps.npy'
        if compact.is_file():
            from retarget.grasp_compact import read_rows
            rows = {r['case']: r for r in read_rows(compact.parent)}
        for path in sorted((run_dir/'results').glob('node_*/summary.json')):
            try:
                stamp = path.stat().st_mtime_ns
                if path not in cache or cache[path][0] != stamp:
                    cache[path] = stamp, json.loads(path.read_text())
                for row in cache[path][1]:
                    rows[row['case']] = row
            except (OSError, ValueError):
                # A case writer may be between truncation and completing JSON.
                continue
        accepted = [r for r in rows.values() if r.get('optimizer_success', False)]
        update(completed=len(rows), optimizer_successes=len(accepted),
            feasible=sum(bool(r['feasible']) for r in rows.values()),
            max_successful_self_penetration_mm=max(
                (r['max_self_penetration_mm'] for r in accepted), default=None))
        return list(rows.values())

    def monitor():
        while not stop.wait(5.):
            progress()

    update()
    watcher = threading.Thread(target=monitor, name='grasp-progress', daemon=True)
    watcher.start()
    try:
        from retarget.grasp_primitives import catalog, CATALOG
        entries = catalog(ROOT/'third_party/Dexonomy')
        if opts.templates != ['all']:
            requested = set(opts.templates)
            entries = [r for r in entries if r['template'] in requested or str(r['type_id']) in requested]
            matched = {v for v in requested if any(v in (r['template'], str(r['type_id'])) for r in entries)}
            if matched != requested:
                raise ValueError(f'Unknown grasp templates: {sorted(requested-matched)}')
        supported = [r for r in entries if r['supported']]
        total = sum((1 if opts.primary_shape_only else len(CATALOG[r['type_id']]))
                    *len(opts.radius_scales) for r in supported)
        update(status='solving', total=total, supported_types=len(supported),
            skipped_types=[r['type_id'] for r in entries if not r['supported']],
            skipped_reason='Types 19 and 21 require articulated/multi-object assets')
        from run_grasp import main as solve
        output = solve(argv)
        progress()
        update(status='auditing')
        from scripts.validate_grasp_matrix import audit_batch
        audited = audit_batch(output)
        (run_dir/'independent_audit.json').write_text(json.dumps(audited, indent=2))
        validation = dict(cases=len(audited),
            valid_exports=sum(r.get('saved_data_valid', False) for r in audited),
            closure_disagreements=sum(not r.get('closure_agrees', False) for r in audited))
        (run_dir/'validation.json').write_text(json.dumps(validation, indent=2))
        rows = progress()
        if len(rows) != total:
            raise RuntimeError(f'Only {len(rows)}/{total} cases were saved')
        accepted = [r for r in rows if r.get('optimizer_success', False)]
        # The legacy capsule audit and optimizer geometry are separate checks.
        # Hand 2 can certify complete mesh separation through its enclosing
        # OBBs while the legacy, oversized capsule approximations still overlap.
        validation['optimizer_successes_with_legacy_capsule_overlap'] = sum(
            r['max_self_penetration_mm'] > 1.e-3 for r in accepted)
        validation['optimizer_successes_failing_legacy_self_tolerance'] = sum(
            r['max_self_penetration_mm'] > 2. for r in accepted)
        (run_dir/'validation.json').write_text(json.dumps(validation, indent=2))
        if validation['valid_exports'] != len(audited) or validation['closure_disagreements']:
            raise RuntimeError('Independent export/closure validation failed; inspect independent_audit.json')
        update(status='complete', output=str(output), validation=validation)
    except BaseException as error:
        progress()
        update(status='failed', error=repr(error))
        raise
    finally:
        stop.set()
        watcher.join(timeout=2.)


if __name__ == '__main__':
    main()
