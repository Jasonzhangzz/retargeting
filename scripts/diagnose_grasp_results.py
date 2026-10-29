#!/usr/bin/env python3
"""Diagnose saved static grasp outcomes and independently check convex objects."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), median=float(np.median(a)), p90=float(np.quantile(a, .9)),
                minimum=float(a.min()), maximum=float(a.max()))


def stage_depths(case):
    from retarget.grasp_hand import query_path
    model = mujoco.MjModel.from_xml_path(str(case / 'scene.xml'))
    data = mujoco.MjData(model)
    obj = model.body('grasp_object').id
    geom = model.geom('object_visual').id
    mesh = model.geom_dataid[geom]
    va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    local = (model.mesh_vert[va:va + vn]
             @ Rotation.from_quat(model.geom_quat[geom], scalar_first=True).as_matrix().T
             + model.geom_pos[geom])
    # All benchmark objects are convex. For an interior point, the nearest
    # supporting plane gives the exact depth to the polyhedral surface.
    equations = np.unique(np.round(ConvexHull(local).equations, 10), axis=0)
    normals, offsets = equations[:, :3], equations[:, 3]
    source_ids, source_local = [], []
    for body in ['right_palm_link'] + [f'right_finger{f}_link{k}' for f in range(1, 6) for k in range(1, 5)]:
        with np.load(query_path(model, body)) as asset:
            points = asset['local_pos'].copy()
        source_ids.extend([model.body(body).id] * len(points))
        source_local.extend(points)
    source_ids, source_local = np.asarray(source_ids), np.asarray(source_local)

    def depth(points):
        # Chunk to bound temporary memory for the faceted spheres.
        deepest = 0.
        for chunk in np.array_split(points, max(1, int(np.ceil(len(points) / 2048)))):
            deepest = max(deepest, float(np.maximum(-(chunk @ normals.T + offsets).max(-1), 0).max()))
        return deepest * 1000

    output = {}
    with np.load(case / 'grasp.npz') as saved:
        for stage, key in [('initial', 'initial_qpos'), ('output', 'qpos'), ('prediction', 'predicted_terminal_qpos')]:
            data.qpos[:] = saved[key]
            mujoco.mj_kinematics(model, data)
            def object_frame(points):
                return (points - data.xpos[obj]) @ data.xmat[obj].reshape(3, 3)
            queries = data.xpos[source_ids] + np.einsum('nij,nj->ni',
                data.xmat[source_ids].reshape(-1, 3, 3), source_local)
            vertices = []
            for g in range(model.ngeom):
                if model.geom_group[g] != 1 or model.geom_bodyid[g] == obj:
                    continue
                mid = model.geom_dataid[g]
                start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                vertices.append(model.mesh_vert[start:start + count]
                                @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
            output[stage + '_query_penetration_mm'] = depth(object_frame(queries))
            output[stage + '_visual_vertex_penetration_mm'] = depth(object_frame(np.concatenate(vertices)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/grasp_validation_20261028')
    args = parser.parse_args()
    rows = json.loads((args.report / 'background_results/summary.json').read_text())
    cases = []
    for row in rows:
        cases.append(dict(template=row['template'], shape=row['primitive']['kind'],
            scale=row['primitive']['radial_scale'], case=row['case'],
            force_closure=row['force_closure']['force_closure'], feasible=row['feasible'],
            required_contacts=row['required_groups_satisfied'],
            reported_penetration_mm=row['max_object_penetration_mm'],
            reported_self_penetration_mm=row['max_self_penetration_mm'],
            **stage_depths(Path(row['case']))))
    obj = np.array([r['max_object_penetration_mm'] for r in rows])
    finger = np.array([r['max_self_penetration_mm'] for r in rows])
    required = np.array([r['required_groups_satisfied'] for r in rows])
    fc = np.array([r['force_closure']['force_closure'] for r in rows])
    combo = Counter(('object' if o > 2 else '') + (' self' if f > 2 else '')
                    + (' missing_contacts' if not r else '') + (' no_closure' if not c else '')
                    or 'pass' for o, f, r, c in zip(obj, finger, required, fc))
    timing = {key: stats([r['timings'][key] for r in rows]) for key in rows[0]['timings']}
    timing['graph_prepare_seconds'] = stats([r['solver_diagnostics']['graph_prepare_seconds'] for r in rows])
    independent_error = max(abs(c['output_query_penetration_mm'] - c['reported_penetration_mm']) for c in cases)
    status = json.loads((args.report / 'background_status.json').read_text())
    summary = dict(cases=len(rows), failed=sum(not r['feasible'] for r in rows),
        object_penetration_over_2mm=int(sum(obj > 2)), self_penetration_over_2mm=int(sum(finger > 2)),
        either_penetration_over_2mm=int(sum(np.maximum(obj, finger) > 2)),
        required_contacts_missing=int(sum(~required)), force_closure_failed=int(sum(~fc)),
        failure_combinations=dict(combo), timings=timing,
        pipeline_wall_seconds=status['wall_seconds'],
        pipeline_amortized_seconds_per_case=status['wall_seconds'] / len(rows),
        independent_query_penetration_max_difference_mm=independent_error,
        visual_vertex_over_2mm=sum(c['output_visual_vertex_penetration_mm'] > 2 for c in cases),
        visual_vertex_over_2mm_among_passed=sum(c['feasible'] and c['output_visual_vertex_penetration_mm'] > 2 for c in cases),
        no_valid_initialization=sum(not r['initialization']['initialization_valid'] for r in rows),
        object_penetration_transitions=dict(Counter(
            f"initial_{c['initial_query_penetration_mm'] > 2}_output_{c['output_query_penetration_mm'] > 2}"
            for c in cases)))
    folder = args.report / 'diagnosis'
    folder.mkdir(exist_ok=True)
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    (folder / 'cases.json').write_text(json.dumps(cases, indent=2))
    with (folder / 'cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
