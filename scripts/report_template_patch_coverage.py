"""Recompute final-pose coverage only; preserve saved q, costs and forces.

Create a new manifest with diagnostic sidecars. The source's model, trajectory,
template, optimization and ComFree artifacts remain byte-for-byte unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import warp as wp

from retarget.common.results import write_json
from retarget.env.geometry import ObjectMeshSurface
from retarget.env.grasp.audit import GraspAudit
from retarget.env.grasp.template_mppi import TemplateMPPISettings
from retarget.env.grasp.template_mppi_inputs import TemplateContacts
from retarget.env.grasp import template_mppi_patch_coverage as metric


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report(source, output, device='cuda:0'):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.parent != ROOT/'output':
        raise ValueError('Use a new direct child of output/')
    parent_audit = source/'independent_result_audit.json'
    if not read(parent_audit)['passed'] or not read(parent_audit).get('complete'):
        raise ValueError('Source batch must have a completed independent audit')
    started = time.perf_counter()
    output.mkdir()
    manifest = read(source/'manifest.json')
    summary = read(source/'summary.json')
    rows, entries, checks = [], [], []
    all_source_hashes = {}
    wp.init()
    for index, row in enumerate(manifest['cases']):
        entry = dict(row)
        for field in ('model', 'trajectory', 'summary'):
            entry[field] = str((source/entry[field]).resolve())
        original_summary = Path(entry['summary'])
        case = Path(entry['trajectory']).parent
        paths = [Path(entry['model']), Path(entry['trajectory']), original_summary]
        paths += [case/name for name in ('template_contacts.json', 'projection_force_closure.json',
                                        'grasp.npz', 'initialization.json', 'optimization.json')]
        before = {str(path): sha(path) for path in paths}
        all_source_hashes.update(before)
        original = read(original_summary)
        template = TemplateContacts(**read(case/'template_contacts.json'))
        model = mujoco.MjModel.from_xml_path(entry['model'])
        with np.load(entry['trajectory'], allow_pickle=False) as trajectory:
            q = trajectory['qpos'][-1].copy()
        settings = TemplateMPPISettings(**original['objective_settings'])
        surface = ObjectMeshSurface.from_model(model, model.body('grasp_object').id)
        side_bodies = tuple(dict.fromkeys(template.body_names[i] for i in
                            original['geometry_diagnostics'].get('side_contact_slots', [])))
        coverage = metric.diagnose_patch_coverage(model, template, q, GraspAudit(surface, device),
                                                  settings, device=device, side_bodies=side_bodies)
        target = output/'cases'/entry['name']
        patch_file = target/'patch_contact_coverage.json'
        coverage['source_artifact_sha256'] = before
        write_json(patch_file, coverage)
        result = dict(original)
        result['geometry_diagnostics'] = metric.attach_patch_coverage(original['geometry_diagnostics'], coverage)
        result['geometry_diagnostics']['patch_coverage_file'] = str(patch_file)
        result['coverage_source_summary'] = str(original_summary)
        write_json(target/'summary.json', result)
        entry.update(summary=str(target/'summary.json'), patch_coverage=str(patch_file),
                     coverage_source_summary=str(original_summary))
        entries.append(entry)
        rows.append(result)
        witness_checks = []
        # Independently verify each witness FK with CPU MuJoCo, without dynamics
        # or rerunning closure. Non-witness queries remain in the saved sidecar.
        data = mujoco.MjData(model)
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        for patch in coverage['patches']:
            witness = patch['witness']
            if witness is None:
                witness_checks.append(not patch['fitted'])
                continue
            body = model.body(patch['body']).id
            world = data.xpos[body] + data.xmat[body].reshape(3, 3) @ witness['local_point']
            agrees = np.max(np.abs(world-witness['hand_point'])) < 1.e-6
            if patch['fitted']:
                agrees &= (-settings.hand_object_allowance_m <= witness['signed_distance_m'] <= settings.contact_tolerance_m
                    and witness['normal_error_deg'] <= settings.normal_tolerance_deg+1.e-5
                    and witness['edge_clearance_m'] >= metric.EDGE_MARGIN_M)
            witness_checks.append(bool(agrees))
        unchanged = all(sha(path) == digest for path, digest in before.items())
        reporting_fields = {'coverage', 'geometry_success', 'patch_coverage_file', 'coverage_definition',
            'covered_patch_count', 'required_patch_count', 'object_surface_regions'}
        same_diagnostics = all(result['geometry_diagnostics'][key] == value for key, value in
            original['geometry_diagnostics'].items() if key not in reporting_fields)
        checks.append(dict(case=entry['name'], sources_unchanged=unchanged,
            cost_force_acceptance_unchanged=same_diagnostics and result['task_success'] == original['task_success'],
            cpu_fk_witnesses=all(witness_checks), empty_patches=coverage['empty_patches']))
        if (index+1)%10 == 0 or index+1 == len(manifest['cases']):
            print(f"[patch coverage] {index+1}/{len(manifest['cases'])}", flush=True)
    manifest['cases'] = entries
    summary.update(cases=rows, coverage_definition=metric.DEFINITION)
    write_json(output/'manifest.json', manifest)
    write_json(output/'summary.json', summary)
    for name in ('resources.json', 'skipped.json', 'template_initialization/manifest.json'):
        write_json(output/name, read(source/name))
    audit = dict(kind='coverage_only_preservation_audit', coverage_definition=metric.DEFINITION,
        cases=len(rows), complete=len(rows) == summary['planned'],
        parent_audit=str(parent_audit), parent_audit_sha256=sha(parent_audit),
        parent_independent_physics_and_force_checks_reused=True,
        optimization_rerun=False, force_solver_rerun=False, native_evaluation_run=False,
        coverage_80_percent=sum(r['geometry_diagnostics']['coverage'] >= .8-1.e-7 for r in rows),
        template_point_coverage_80_percent=sum(r['geometry_diagnostics']['template_point_coverage'] >= .8-1.e-7 for r in rows),
        projected_closed=sum(r['geometry_diagnostics']['projected_closure'] for r in rows),
        seconds=time.perf_counter()-started, cases_detail=checks)
    audit['passed'] = bool(audit['complete'] and all(c['sources_unchanged'] and
        c['cost_force_acceptance_unchanged'] and c['cpu_fk_witnesses'] and not c['empty_patches'] for c in checks))
    write_json(output/'independent_result_audit.json', audit)
    write_json(output/'patch_coverage_audit.json', audit)
    write_json(output/'provenance.json', dict(kind='coverage_only_composition', source_batch=str(source),
        source_manifest_sha256=sha(source/'manifest.json'), parent_audit_sha256=sha(parent_audit),
        metric_source_sha256={str(path): sha(path) for path in (Path(__file__).resolve(), Path(metric.__file__).resolve())},
        pose_cost_force_artifacts_unchanged=True, coverage_definition=metric.DEFINITION))
    write_json(output/'source_artifact_sha256.json', all_source_hashes)
    write_json(output/'status.json', dict(state='completed', finished=len(rows), planned=len(rows), metric_only=True))
    with (output/'results.csv').open('w') as stream:
        fields = ['template', 'diameter_mm', 'height_mm', 'patch_coverage', 'template_point_coverage',
                  'covered_patches', 'required_patches', 'projected_closure', 'optimization_seconds']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            d, p = row['geometry_diagnostics'], row['primitive']
            writer.writerow(dict(template=row['template'], diameter_mm=p['radius_m']*2000,
                height_mm=p['height_m']*1000, patch_coverage=d['coverage'],
                template_point_coverage=d['template_point_coverage'], covered_patches=d['covered_patch_count'],
                required_patches=d['required_patch_count'], projected_closure=d['projected_closure'],
                optimization_seconds=row['optimization_seconds']))
    concise = {key: value for key, value in audit.items() if key != 'cases_detail'}
    write_json(output/'batch_report.json', concise)
    print(json.dumps(concise, indent=2))
    if not audit['passed']:
        raise RuntimeError('Coverage preservation audit failed; inspect patch_coverage_audit.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    report(args.source, args.output, args.device)
