"""Independently audit a completed fixed-template MPPI batch (no resynthesis)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.common.results import write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repository_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def audit(root, *, partial=False, shared_budget_bytes=None):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())['cases']
    summary = json.loads((root / 'summary.json').read_text())
    references = json.loads((root / 'template_initialization/manifest.json').read_text())
    dense_plan = json.loads((root/'continuation_plan.json').read_text()) if (root/'continuation_plan.json').is_file() else None
    dense_nodes = {n['name']: n for n in dense_plan['nodes']} if dense_plan else {}
    rows, failures, reference_poses = [], [], {}
    for entry in manifest:
        case = (root / entry['trajectory']).parent
        record = json.loads((root / entry['summary']).read_text())
        initial = json.loads((case / 'initialization.json').read_text())
        template = json.loads((case / 'template_contacts.json').read_text())
        d = json.loads((case / 'projection_force_closure.json').read_text())
        reference_path = repository_path(initial['reference'])
        ref = json.loads(reference_path.read_text())
        model = mujoco.MjModel.from_xml_path(str(root / entry['model']))
        with np.load(case / 'grasp.npz', allow_pickle=False) as values:
            q = values['qpos'].copy()
        with np.load(root / entry['trajectory'], allow_pickle=False) as values:
            trajectory = values['qpos'].copy()
        checks = {}
        checks['trajectory_final'] = bool(np.array_equal(q, trajectory[-1]) and np.isfinite(trajectory).all())
        checks['reference_hash'] = sha(reference_path) == initial['reference_sha256']
        canonical_q = initial.get('reference_initial_qpos', initial['initial_qpos'])
        checks['standard_initialization'] = canonical_q == ref['initial_qpos']
        key = (record['template'], record['primitive']['kind'])
        reference_poses.setdefault(key, canonical_q)
        checks['shared_across_sizes'] = bool(dense_plan) or reference_poses[key] == canonical_q
        if ref.get('policy') == 'template_adjacent_solved_grasp_v1':
            from retarget.env.grasp.template_mppi_continuation import pose_change
            continuation = json.loads((root / entry['continuation']).read_text())
            parent = continuation['parent_entry']
            parent_case = repository_path(parent['trajectory']).parent
            parent_q = np.load(parent_case/'grasp.npz')['qpos']
            parent_d = json.loads((parent_case/'projection_force_closure.json').read_text())
            evidence = ref['canonical_grasp_reference']
            checks['adjacent_parent_hash'] = (sha(repository_path(evidence['source'])) == evidence['sha256']
                and sha(repository_path(evidence['qpos_source'])) == evidence['qpos_sha256'])
            checks['adjacent_solved_seed'] = bool(np.array_equal(parent_q, ref['initial_qpos'])
                and ref['object_points'] == parent_d['projected_points']
                and ref['object_outward_normals'] == parent_d['object_normals']
                and ref['canonical_primitive'] == parent['primitive'])
            checks['adjacent_dependency'] = bool(dense_plan and dense_nodes[entry['name']]['parent_index']
                == dense_nodes[parent['name']]['index'])
            measured_motion = pose_change(parent_q,q)
            limit = continuation['limits']
            checks['bounded_adjacent_motion'] = (measured_motion['object_translation_m'] <= limit['translation_m']+3.e-7
                and measured_motion['object_rotation_deg'] <= limit['rotation_deg']+1.e-4
                and measured_motion['max_joint_change_deg'] <= limit['joint_deg']+2.e-4)
            # Trace/acos near zero loses angular precision across BLAS platforms.
            # Keep metre/joint checks strict; allow only 0.00001 degree roundoff
            # when comparing the saved SO(3) angle, independently of hard bounds.
            checks['motion_report_agreement'] = all(abs(measured_motion[k]-continuation['motion'][k])
                < (1.e-5 if k == 'object_rotation_deg' else 1.e-8) for k in measured_motion)
            checks['no_large_object_jump'] = (measured_motion['object_translation_m'] <= .0030003
                and measured_motion['object_rotation_deg'] <= 3.0001)
            checks['adjacent_template_identity'] = all(template[k] == json.loads((parent_case/'template_contacts.json').read_text())[k]
                for k in ('body_names','local_points','local_normals','finger_qpos'))
        if initial.get('size_ik_policy'):
            ik = initial['size_ik']
            checks['cuda_size_ik'] = ik['backend'] == 'torch_cuda_batched_bounded_LM'
            checks['size_ik_slots_preserved'] = ik['template_slots_preserved']
            checks['size_ik_trajectory_initial'] = bool(np.allclose(trajectory[0], initial['initial_qpos'], atol=1.e-6))
            checks['size_ik_descent'] = bool(np.all(np.diff(np.array(ik['lm_objective_history']), axis=0) <= 1.e-6))
            checks['size_ik_primitive_preserved'] = ik.get('target_primitive') == record.get('primitive')
            primitive = record['primitive']
            standard = ref['canonical_primitive']
            transported = np.array(ref['object_points'], float)
            if primitive and standard:
                transported *= np.array([primitive['radius_m']/standard['radius_m']]*2+
                                        [primitive['height_m']/standard['height_m']])
                if ik.get('side_contact_slots'):
                    slots = ik['side_contact_slots']
                    margin = min(.001, primitive['height_m']*.2)
                    transported[slots, 2] = np.clip(transported[slots, 2],
                        -primitive['height_m']/2+margin, primitive['height_m']/2-margin)
            checks['size_contact_transport'] = bool(np.allclose(transported, ik['transported_object_points'], atol=1.e-9, rtol=0))
            checks['pose_prior_disabled'] = record['objective_settings']['pose_weight'] == 0.
            if ref.get('object_contact_reference') == 'solved_1x_grasp':
                evidence = ref['canonical_grasp_reference']
                checks['solved_canonical_source_hash'] = sha(repository_path(evidence['source'])) == evidence['sha256']
                checks['solved_canonical_seed'] = bool(ik['canonical_grasp_seed_used'] and
                    np.array_equal(ik['source_initial_qpos'], ref['canonical_grasp_qpos']))
        checks['no_translation_scaling'] = initial['translation_scaled'] is False
        checks['gpu_candidate_fk'] = d['fk_backend'] == 'mujoco_warp_cuda_batched'
        checks['gpu_template_fk'] = initial['canonical_fk']['backend'] == 'torch_cuda_batched_WujiContactKinematics'
        checks['batched_template_fk'] = (initial['canonical_fk']['batch_size'] >= 1
            and initial['canonical_fk']['batch_size'] == ref['matcher']['canonical_fk']['batch_size'])
        checks['exact_1x'] = (ref['canonical_primitive']['size_scale'] == ref['canonical_size_scale']
            if ref.get('policy') == 'template_adjacent_solved_grasp_v1' else ref['canonical_primitive']['size_scale'] == 1.)
        original_template = json.loads(reference_path.with_name('template_contacts.json').read_text())
        checks['fixed_template'] = all(template[k] == original_template[k] for k in (
            'body_names', 'local_points', 'local_normals', 'finger_qpos'))
        if template['provenance'].get('policy') == 'fixed_prismatic_pad_center_v1':
            from retarget.env.grasp.contact_regions import region_coordinates
            uv = [region_coordinates(body,p,n) for body,p,n in zip(template['body_names'],
                      template['local_points'],template['local_normals'])]
            checks['fixed_pad_center_template'] = all(.3 <= float(u) <= .7 and float(facing) > .9 for u,facing,_ in uv)
            checks['canonical_center_template_recomputed'] = ref.get('policy') == 'template_adjacent_solved_grasp_v1' or not ref['canonical_grasp_reference']['from_cache']
        T = np.eye(4)
        from scipy.spatial.transform import Rotation
        T[:3, :3] = Rotation.from_euler('XYZ', np.array(canonical_q)[3:6]).as_matrix()
        T[:3, 3] = canonical_q[:3]
        checks['object_wrist_inverse'] = bool(np.allclose(T @ np.array(ref['object_in_wrist']), np.eye(4), atol=1e-10))
        data = mujoco.MjData(model)
        data.qpos[:26] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        bodies = np.array([model.body(name).id for name in template['body_names']])
        rotations = data.xmat[bodies].reshape(-1, 3, 3)
        hand = np.einsum('nij,nj->ni', rotations, template['local_points']) + data.xpos[bodies]
        normals = np.einsum('nij,nj->ni', rotations, template['local_normals'])
        position_error = float(np.max(np.abs(hand - d['hand_points'])))
        normal_error = float(np.max(np.abs(normals - d['hand_normals'])))
        checks['cpu_fk'] = position_error < 1e-6 and normal_error < 2e-6
        if d.get('articulation_policy'):
            checks['articulation_joint_bounds'] = bool(np.all(q >= np.array(d['optimization_joint_lower_limits'])-1.e-6) and np.all(q <= np.array(d['optimization_joint_upper_limits'])+1.e-6))
            neutral=mujoco.MjData(model);mujoco.mj_kinematics(model,neutral)
            palm=model.body('right_palm_link').id
            axis=neutral.xpos[model.body('right_finger5_link2').id]-neutral.xpos[model.body('right_finger2_link2').id]
            axis=neutral.xmat[palm].reshape(3,3).T@axis;axis/=np.linalg.norm(axis)
            locations=[]
            for finger,label in zip(range(2,6),('index','middle','ring','pinky')):
                a,b=[model.body(f'right_finger{finger}_link{k}').id for k in (3,4)]
                tip=data.xpos[b]+data.xmat[b].reshape(3,3)@model.site(f'right_{label}_tip').pos
                locations.append([data.xpos[a],data.xpos[b],tip])
            points_order=np.array(locations)
            gaps=(points_order[1:]-points_order[:-1])@(data.xmat[palm].reshape(3,3)@axis)
            crossing=max(float(d['articulation_policy']['order_margin_m']-gaps.min()),0.)
            checks['cpu_finger_order_agreement'] = abs(crossing-d['max_finger_order_violation_m']) < 1.e-6
            checks['no_finger_order_reversal'] = crossing <= d['settings']['penetration_tolerance_m']+1.e-7
        posture = d.get('grasp_posture_bounds', {})
        if posture:
            checks['grasp_posture_range'] = bool(np.all(q[posture['joint_indices']] >= np.array(posture['lower_limits_rad'])-1.e-6))
            checks['posture_not_reference_pose_prior'] = posture['reference_pose_used'] is False and d['settings']['pose_weight'] == 0.
        object_id = model.body('grasp_object').id
        cpu_depth = np.zeros(2)
        for contact in data.contact:
            pair = model.geom_bodyid[contact.geom]
            if np.all(pair == object_id):
                continue
            kind = 0 if object_id in pair else 1
            cpu_depth[kind] = max(cpu_depth[kind], -contact.dist)
        gpu_depth = np.array([d['max_hand_object_penetration_m'], d['max_hand_hand_penetration_m']])
        collision_error = float(np.max(np.abs(cpu_depth - gpu_depth)))
        checks['cpu_collision_agreement'] = collision_error < 2e-5
        checks['cpu_self_acceptance_agreement'] = bool(cpu_depth[1] <= d['settings']['penetration_tolerance_m']) == d['hand_hand_penetration_accepted']
        m = d['closure_model']
        points = np.array(d['projected_points'])
        outward = np.array(d['object_normals'])
        force = np.array(d['response_contact_forces_n'])
        spin = np.array(d['response_intrinsic_torques_nm'])
        normal_force = np.array(d['response_normal_forces_n'])
        arms = points - np.array(m['center'])
        wrench = np.concatenate((force.sum(1), (np.cross(arms[None], force) + spin).sum(1) / m['radius_m']), axis=-1)
        loads = np.concatenate((np.zeros((1, 6)), np.eye(6), -np.eye(6))) * m['test_force_n']
        residual = (wrench + loads) / m['test_force_n']
        closure_error = float(np.linalg.norm(residual[1:], axis=-1).max())
        residual_error = float(np.max(np.abs(residual - np.array(d['scenario_residuals']))))
        certified = closure_error <= m['certificate_tolerance'] and closure_error < 1 / np.sqrt(6)
        checks['force_certificate'] = bool(certified) == d['projected_closure'] and residual_error < 1e-4
        if d.get('side_contact_slots'):
            checks['requested_side_normals'] = bool(np.max(np.abs(outward[d['side_contact_slots'], 2])) < 1.e-5)
        if initial.get('size_ik_policy') and int(record['template'].partition('_')[0]) in (6, 7, 9, 10):
            expected_side = [i for i, body in enumerate(template['body_names']) if body.startswith('right_finger')]
            if record['template'].startswith('10_'):
                selected = d.get('side_contact_slots',[])
                expected_fingers = {template['body_names'][i].split('_link')[0] for i in expected_side}
                selected_fingers = {template['body_names'][i].split('_link')[0] for i in selected}
                checks['requested_side_constraints_enabled'] = len(selected)==len(expected_fingers) and selected_fingers==expected_fingers
                checks['canonical_side_identities'] = selected == ref.get('canonical_side_contact_slots',selected)
            else:
                checks['requested_side_constraints_enabled'] = d.get('side_contact_slots') == expected_side
        checks['nonnegative_ray_forces'] = bool(np.min(d['response_row_forces_n']) >= -1e-7)
        inward = -outward / np.linalg.norm(outward, axis=-1, keepdims=True)
        compressive = np.sum(force * inward[None], axis=-1)
        tangent = force - compressive[..., None] * inward[None]
        checks['friction_bounds'] = bool(np.all(np.linalg.norm(tangent, axis=-1) <= m['friction'] * normal_force + 1e-4)
            and np.all(np.abs(np.sum(spin * inward[None], axis=-1)) <= m['torsional_friction_m'] * normal_force + 1e-5))
        preloads = np.array(d['nominal_preloads_n'])
        checks['preload_bounds'] = bool(np.all(preloads >= 0) and np.all(preloads <= m['max_preload_n'] + 1e-5))
        expected = bool(d['valid'] and certified and gpu_depth[1] <= d['settings']['penetration_tolerance_m'])
        checks['latest_acceptance'] = record['success_criterion'] == 'projected_soft_contact_closure_v2' and record['task_success'] == expected
        checks['physical_evaluation_disabled'] = record['native_evaluated'] is False
        if not all(checks.values()):
            failures.append(dict(case=entry['name'], failed=[key for key, value in checks.items() if not value]))
        rows.append(dict(case=entry['name'], template=record['template'], size_scale=record['primitive']['size_scale'],
            success=record['task_success'], closure=d['projected_closure'], coverage=record['geometry_diagnostics']['coverage'],
            hand_object_depth_mm=float(gpu_depth[0] * 1000), hand_hand_depth_mm=float(gpu_depth[1] * 1000),
            cpu_depth_mm=(cpu_depth * 1000).tolist(), collision_error_m=collision_error,
            position_error_m=position_error, normal_error=normal_error, residual_error=residual_error,
            closure_error=closure_error, optimization_seconds=record['optimization_seconds'], checks=checks))
    source = json.loads((root / 'source_sha256_at_start.json').read_text())
    source_changed = [path for path, digest in source.items() if not Path(path).is_file() or sha(path) != digest]
    lineage = None
    if (root/'composition.json').is_file():
        composition = json.loads((root/'composition.json').read_text())
        parents = composition.get('source_audit_sha256', {})
        lineage = bool(parents) and all(repository_path(path).is_file() and sha(repository_path(path))==digest
            and json.loads(repository_path(path).read_text()).get('passed')
            and json.loads(repository_path(path).read_text()).get('complete') for path,digest in parents.items())
    elif dense_plan:
        parents = dense_plan['source_audit_sha256']
        lineage = bool(parents) and not source_changed and all(Path(path).is_file() and sha(path)==digest
            and json.loads(Path(path).read_text()).get('passed')
            and json.loads(Path(path).read_text()).get('complete') for path,digest in parents.items())
    resources = json.loads((root / 'resources.json').read_text())
    budget = resources['budget_bytes'] if shared_budget_bytes is None else shared_budget_bytes
    if shared_budget_bytes is not None:
        shared = json.loads((root/'shared_cuda_budget.json').read_text())
        if shared['shared_budget_bytes'] != shared_budget_bytes or shared_budget_bytes > .8*shared['initial_free_bytes']:
            raise ValueError('Shared device budget must be recorded and within 80% of initial free VRAM')
    result = dict(cases=len(rows), planned=summary['planned'], solver_completed=summary['solver_completed'],
        fits=references['fits'], fixed_standard_poses=len(reference_poses),
        successes=sum(row['success'] for row in rows), projected_closed=sum(row['closure'] for row in rows),
        coverage_80_percent=sum(row['coverage'] >= .8 - 1e-7 for row in rows),
        source_changed=source_changed, memory_within_budget=resources['peak_incremental_bytes'] <= budget,
        composed_source_audit_chain_verified=lineage,
        validator_sha256=sha(__file__),
        source_validation_scope='archived solver source hashes and verified parent audits; final FK, collisions and forces rechecked here' if lineage is not None else 'single solve source snapshot unchanged',
        memory_validation_budget_bytes=budget,
        memory_validation_scope='coordinated shared device budget' if shared_budget_bytes is not None else 'single batch budget',
        wall_seconds=resources.get('wall_seconds'), peak_fraction_of_initial_free=resources['peak_fraction_of_initial_free'],
        failed_checks=failures, cases_detail=rows)
    result['complete'] = len(rows) == summary['planned'] == summary['solver_completed']
    result['passed'] = bool(not failures and (lineage if lineage is not None else not source_changed) and result['memory_within_budget']
        and (partial or (result['complete'] and len(reference_poses) == references['fits'])))
    write_json(root / ('partial_audit.json' if partial else 'independent_result_audit.json'), result)
    print(json.dumps({key: value for key, value in result.items() if key != 'cases_detail'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--partial', action='store_true', help='Audit the completed subset of a running batch')
    parser.add_argument('--shared-budget-bytes', type=int, help='Recorded joint VRAM ceiling for concurrent batches')
    args = parser.parse_args()
    result = audit(args.batch, partial=args.partial, shared_budget_bytes=args.shared_budget_bytes)
    raise SystemExit(0 if result['passed'] else 1)
