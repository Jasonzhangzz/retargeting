#!/usr/bin/env python3
"""Resume remaining grasp cases, validate exports and render when finished."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def key(row):
    p = row['primitive']
    return row['template'], p['kind'], round(p['radius_m'], 9), round(p['height_m'], 9)


def snapshot_scene(case):
    """Repair historical process-global XML exports from the case's own input."""
    import mujoco
    from comfree_warp.gaussian_loader import _preprocess_xml
    source = json.loads((case/'result.json').read_text())
    experiment = case/f"input_{source['selected']['start']:02d}"
    prepared = _preprocess_xml(str(experiment/'sequence/scene/wuji_right/primitive_sdf.xml'))
    model = mujoco.MjModel.from_xml_string(prepared.model_xml, assets=prepared.assets)
    model.opt.timestep = source['selected']['output_time_seconds']
    mujoco.mj_saveLastXML(str(case/'scene.xml'), model)
    source['scene_export'] = 'case-owned prepared model XML; independent of global last-loaded scene'
    (case/'result.json').write_text(json.dumps(source, indent=2))


def main():
    import numpy as np
    import retarget.grasp as grasp
    from retarget.grasp_primitives import jobs
    from run_grasp import main as cli
    from scripts.validate_grasp_matrix import audit_batch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--started', type=float, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    report_root = source.parent
    combined = report_root/'background_results'
    combined.mkdir(exist_ok=True)
    status_path = report_root/'background_status.json'
    status = dict(status='starting', original_started=args.started, supervisor_pid=os.getpid())

    def update(**values):
        status.update(values, wall_seconds=time.time()-args.started, updated=time.time())
        status_path.write_text(json.dumps(status, indent=2))

    try:
        if args.pid:
            try:
                os.kill(args.pid, signal.SIGINT)
                for _ in range(100):
                    if not Path(f'/proc/{args.pid}').exists():
                        break
                    time.sleep(.1)
            except ProcessLookupError:
                pass
        completed = []
        for p in sorted(source.glob('node_*/summary.json')):
            completed.extend(json.loads(p.read_text()))
        for row in completed:
            snapshot_scene(Path(row['case']))
        (combined/'summary.json').write_text(json.dumps(completed, indent=2))
        with patch.object(grasp, 'run', lambda settings: settings):
            settings = cli(['--primitives', '--templates', 'all', '--radius-scales', '.75', '1', '1.25',
                '--gpu-nodes', '8', '--no-viewer', '--no-render', '--output', str(report_root),
                '--name', 'gpu_remaining_'+str(int(time.time()))])
        all_tasks, _ = jobs(report_root/'resume_meshes', settings.templates, settings)
        done = {key(row) for row in completed}
        rng = np.random.default_rng(settings.seed)
        seeds = [settings.seed]+[int(rng.integers(0, 2**31-1)) for _ in range(len(all_tasks)-1)]
        settings._tasks, settings._task_seeds = [], []
        for index, task in enumerate(all_tasks):
            mesh, template, primitive = task
            if key(dict(template=template, primitive=primitive)) not in done:
                settings._tasks.append(task)
                settings._task_seeds.append(seeds[index])
        remaining = len(settings._tasks)
        elapsed = time.time()-args.started
        estimate = elapsed/max(len(completed), 1)*remaining
        update(status='running', completed_at_restart=len(completed), remaining=remaining,
               expected_cases=len(all_tasks), estimated_remaining_seconds=estimate,
               remaining_output=str(settings.output/settings.name))
        if remaining:
            output = grasp.run(settings)
            completed.extend(json.loads((output/'summary.json').read_text()))
        unique = {key(row): row for row in completed}
        completed = sorted(unique.values(), key=lambda r: (int(r['template'].split('_')[0]),
            r['primitive']['kind'], r['primitive']['radius_m'], r['primitive']['height_m']))
        (combined/'summary.json').write_text(json.dumps(completed, indent=2))
        update(status='auditing', saved_cases=len(completed), estimated_remaining_seconds=None)
        audited = audit_batch(combined)
        (combined/'cases.json').write_text(json.dumps(audited, indent=2))
        if audited:
            with (combined/'cases.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in audited))))
                writer.writeheader(); writer.writerows(audited)
        result = dict(saved_cases=len(audited), expected_cases=len(all_tasks),
            types=len({r['type_id'] for r in audited}),
            force_closure=sum(r['force_closure'] for r in audited), feasible=sum(r['feasible'] for r in audited),
            valid_exports=sum(r.get('saved_data_valid', False) for r in audited),
            closure_disagreements=sum(not r.get('closure_agrees', False) for r in audited),
            wall_seconds=time.time()-args.started)
        (combined/'validation.json').write_text(json.dumps(result, indent=2))
        update(status='rendering', **result)
        subprocess.run([sys.executable, str(ROOT/'scripts/render_grasp_comparison.py'),
            '--legacy', str(report_root/'matrix'), '--gpu', str(combined/'summary.json'),
            '--output', str(report_root/'grasp_comparison_complete.png')], check=True, cwd=ROOT)
        update(status='complete', **result, estimated_remaining_seconds=0.)
    except BaseException as error:
        update(status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
