#!/usr/bin/env python3
"""Run the existing grasp CLI in isolated processes and audit saved results.

This is a validation launcher, not a change to run_grasp.py's serial scheduler.
Physical retention is deliberately excluded from this static closure sweep.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(argv):
    # Record actual rollout allocation, rather than inferring it from a flag.
    from retarget.grasp_runtime import GraspRuntime
    from run_grasp import main
    original = GraspRuntime.solve

    def observed(self):
        result = original(self)
        static = self.grasp_optimizer == 'cem'
        shape = ((self.args.cem_population, self.model.nq) if static else
                 self.optimizer.backend.steps[0].recorded.result.qpos.shape)
        print('[validation_runtime] ' + json.dumps(dict(
            pid=os.getpid(), device=str(self.optimizer.backend.device),
            qpos_shape=list(shape), worlds=shape[0],
            search_starts=shape[0] if static else self.optimizer.search_worlds,
            static_configuration=static)), flush=True)
        return result

    GraspRuntime.solve = observed
    main(argv)


def audit_batch(batch):
    if (Path(batch)/'grasps.npy').is_file():
        from retarget.grasp_compact import audit_records
        return audit_records(batch)
    import mujoco
    import numpy as np
    from retarget.grasp_closure import force_closure
    summary = batch / 'summary.json'
    rows = []
    if not summary.exists():
        return rows
    for item in json.loads(summary.read_text()):
        case = Path(item['case'])
        report = json.loads((case / 'result.json').read_text())
        row = dict(type_id=int(item['template'].split('_')[0]),
                   template=item['template'], **item['primitive'], case=str(case),
                   force_closure=item['force_closure']['force_closure'],
                   feasible=item['feasible'], geometry_feasible=item['geometry_feasible'],
                   active_contacts=item['active_contacts'], gap_max_mm=item['gap_max_mm'],
                   penetration_mm=item['max_object_penetration_mm'],
                   self_penetration_mm=item['max_self_penetration_mm'],
                   initialization_valid=item['initialization'].get('initialization_valid'))
        try:
            model = mujoco.MjModel.from_xml_path(str(case / 'scene.xml'))
            with np.load(case / 'grasp.npz', allow_pickle=False) as saved:
                required = ['qpos', 'qvel', 'hold_action', 'initial_qpos',
                            'realized_object_points', 'realized_object_normals',
                            'realized_contact_mask', 'hand_qpos']
                assert all(key in saved for key in required), 'missing pose/contact arrays'
                assert saved['qpos'].shape == (model.nq,)
                assert saved['qvel'].shape == (model.nv,)
                assert all(np.isfinite(saved[key]).all() for key in saved.files), 'nonfinite data'
                state = mujoco.MjData(model)
                state.qpos[:] = saved['qpos']
                mujoco.mj_kinematics(model, state)
                bodies = saved['hand_body_ids']
                world = state.xpos[bodies]+np.einsum('nij,nj->ni',
                    state.xmat[bodies].reshape(-1, 3, 3), saved['hand_local_points'])
                obj = model.body('grasp_object').id
                independent_points = (world-state.xpos[obj])@state.xmat[obj].reshape(3, 3)
                np.testing.assert_allclose(independent_points, saved['hand_points'], atol=2.e-6, rtol=1.e-5)
                row['saved_fk_max_error_m'] = float(np.abs(independent_points-saved['hand_points']).max())
                active = saved['realized_contact_mask'].astype(bool)
                # Match production's object bounding radius, independently reconstructed.
                mesh_path = Path(report['object'])
                from retarget.grasp import build_scene
                _, surface, _, _ = build_scene(mesh_path, report['scale'], report.get('hand_model','wuji1'))
                from retarget.contact_exploration import ObjectMeshSurface
                from scipy.spatial import cKDTree
                exported = ObjectMeshSurface.from_model(model, model.body('grasp_object').id)
                expected_vertices = surface.triangles.reshape(-1, 3)
                saved_vertices = exported.triangles.reshape(-1, 3)
                mesh_error = max(cKDTree(expected_vertices).query(saved_vertices)[0].max(),
                                 cKDTree(saved_vertices).query(expected_vertices)[0].max())
                assert mesh_error < 2.e-7, f'XML references a different object mesh: error={mesh_error:g} m'
                row['saved_mesh_max_error_m'] = float(mesh_error)
                radius = max(float(np.linalg.norm(surface.triangles.reshape(-1, 3), axis=1).max()), .001)
                oracle = force_closure(saved['realized_object_points'][active],
                    saved['realized_object_normals'][active],
                    friction=report['settings']['friction'], radius=radius,
                    edges=report['settings']['cone_edges'])
                row.update(cpu_force_closure=oracle['force_closure'],
                           cpu_rank=oracle['rank'], cpu_margin=oracle['positive_margin'],
                           cpu_residual=oracle['equilibrium_residual'],
                           closure_agrees=oracle['force_closure'] == row['force_closure'])
            with np.load(case / 'contacts.npz', allow_pickle=False) as saved:
                assert saved.files and all(np.isfinite(saved[key]).all() for key in saved.files)
            row['saved_data_valid'] = True
        except Exception as error:
            row.update(saved_data_valid=False, data_error=repr(error))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--templates', nargs='+', type=int)
    parser.add_argument('--radius-scales', nargs='+', type=float, default=[.75, 1., 1.25])
    parser.add_argument('--height-scales', nargs='+', type=float, default=[1.])
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    from retarget.grasp_primitives import catalog, CATALOG
    rows = [r for r in catalog(ROOT / 'third_party/Dexonomy') if r['supported']]
    if args.templates:
        rows = [r for r in rows if r['type_id'] in args.templates]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='egl')

    def launch(row):
        tag = f"type_{row['type_id']:02d}"
        command = [sys.executable, str(Path(__file__).resolve()), '--worker',
            '--primitives', '--templates', str(row['type_id']),
            '--radius-scales', *map(str, args.radius_scales),
            '--height-scales', *map(str, args.height_scales),
            '--no-viewer', '--no-render', '--no-drop-test',
            '--legacy-serial',
            '--output', str(output), '--name', tag]
        begun = time.time()
        with (output / f'{tag}.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            code = process.wait()
        result = dict(type_id=row['type_id'], pid=process.pid, command=command,
                      start=begun, end=time.time(), returncode=code,
                      expected_cases=sum(len(args.radius_scales) *
                          (1 if kind == 'sphere' else len(args.height_scales))
                          for kind, _, _ in CATALOG[row['type_id']]))
        print(json.dumps(result), flush=True)
        return result

    jobs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(launch, row) for row in rows]
        for future in as_completed(futures):
            jobs.append(future.result())
            (output / 'jobs.json').write_text(json.dumps(jobs, indent=2))
    cases = []
    for job in sorted(jobs, key=lambda r: r['type_id']):
        cases.extend(audit_batch(output / f"type_{job['type_id']:02d}"))
    (output / 'cases.json').write_text(json.dumps(cases, indent=2))
    if cases:
        columns = sorted(set().union(*(row.keys() for row in cases)))
        with (output / 'cases.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(cases)
    events = sorted([(r['start'], 1) for r in jobs] + [(r['end'], -1) for r in jobs])
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    result = dict(python=sys.executable, workers=args.workers, peak_overlapping_processes=peak,
                  requested_types=len(rows), expected_cases=sum(r['expected_cases'] for r in jobs),
                  finished_types=sum(r['returncode'] == 0 for r in jobs), saved_cases=len(cases),
                  force_closure=sum(r['force_closure'] for r in cases),
                  feasible=sum(r['feasible'] for r in cases),
                  saved_data_valid=sum(r.get('saved_data_valid', False) for r in cases),
                  closure_disagreements=sum(not r.get('closure_agrees', False) for r in cases),
                  wall_seconds=max(r['end'] for r in jobs)-min(r['start'] for r in jobs))
    (output / 'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(sys.argv[2:])
    else:
        main()
