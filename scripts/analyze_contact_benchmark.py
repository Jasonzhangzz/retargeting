#!/usr/bin/env python
"""Add executed-trajectory contact/object metrics and aggregate a completed campaign."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from retarget.common.results import write_json
from retarget.task.contact.tasks import Task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    results = json.loads((campaign / 'results.json').read_text())
    groups = defaultdict(list)
    for result in results:
        tracking_flag = result['tracking']
        directory = campaign.parent / result['run']
        summary_path = directory / 'summary.json'
        if not summary_path.exists():
            status = directory / 'status.json'
            result['runtime_error'] = json.loads(status.read_text()).get('error') if status.exists() else 'No run artifacts'
            result['complete'] = False
            groups[(result['task'], result['method'], result['tracking'])].append(result)
            continue
        summary = json.loads(summary_path.read_text())
        task = Task(result['task'], result['seed'], 'cpu')
        with np.load(directory / 'trajectory.npz') as raw:
            q, v, sensor = raw['qpos'], raw['qvel'], raw['sensordata']
            initial_q, initial_v = raw['initial_qpos'], raw['initial_qvel']
            actuator_force = raw['actuator_force'].reshape(-1, task.model.nu)
        joints = task.model.actuator_trnid[:, 0]
        joint_torque = actuator_force * task.model.actuator_gear[:, 0]
        limited = task.model.jnt_actfrclimited[joints].astype(bool)
        joint_torque[:, limited] = np.clip(joint_torque[:, limited],
            task.model.jnt_actfrcrange[joints[limited], 0], task.model.jnt_actfrcrange[joints[limited], 1])
        np.savez_compressed(directory / 'joint_torque.npz', torque_nm=joint_torque,
            joint_names=[task.model.joint(int(j)).name for j in joints])
        contacts = [json.loads(line)['contacts'] for line in (directory / 'contacts.jsonl').read_text().splitlines()]
        events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
        initial_packet = task.packet(dict(qpos=initial_q, qvel=initial_v), 0)
        runtime = json.loads((directory / 'runtime.json').read_text())
        profile = runtime['task']['profile']
        method = result['method']
        overrides = dict(optimizer_backend='original_robokit_SimMPC')
        if method in ('cem', 'contact_cem'):
            overrides = dict(optimizer_backend='Robokit_PopulationSolver/GaussianSampler_CEM_refit',
                scored_proposals_per_iteration=profile['samples'], sample_every=1,
                final_unscored_mean_rollout=False, noise_decay=1.,
                elite_fraction=runtime['defaults']['elite_fraction'],
                refit=runtime['defaults']['refit'],
                min_sigma_fraction=runtime['defaults']['min_sigma_fraction'],
                simulation_num_samples=profile['samples'] + 1,
                simulation_stage='fine original model timestep',
                extra_contact_dynamics_task=True)
            if method == 'contact_cem':
                overrides.update(physics_control_node_steps=1, physics_interpolation='zero',
                    physics_sampler_ctrl_std=0., contact_variable_count=30,
                    discrete_branches='independent original GaussianSampler instances')
        write_json(directory / 'expanded_problem_parameters.json', dict(
            profile=profile, search=runtime['defaults'],
            original_robokit_config=runtime['task'].get('robokit_config'),
            effective_method_overrides=overrides,
            control_std=task.ctrl_std, control_bounds=task.ctrl_range,
            task_cost={k: initial_packet[k] for k in ('robokit_terms', 'weight', 'terminal', 'ctrl_weight',
                'heading', 'heading_weight', 'heading_terminal', 'support_obj', 'support_margin', 'dt')},
            actuator_force_range=task.model.actuator_forcerange, actuator_gear=task.model.actuator_gear,
            joint_limited=task.model.jnt_limited, joint_range=task.model.jnt_range,
            note='Task weights are constant within each task; evolving references are specified by the saved source and frozen reference artifact. Fixed dynamics penalties are preserved in implementation/retarget/env/contact/native.py and contact_terms.py.'))
        iterations = [json.loads(line) for line in (directory / 'iterations.jsonl').read_text().splitlines()]
        validations_path = directory / 'native_validations.jsonl'
        validations = [json.loads(line) for line in validations_path.read_text().splitlines()] if validations_path.exists() else []
        timing_keys = ('reference_and_candidates', 'contact_lm', 'gpu_rollout', 'cpu4_validation', 'search_other')
        timed = [r for r in iterations if 'phase_seconds' in r]
        flat_contacts = [c for step in contacts for c in step]
        normals, distances, object_errors, angle_errors, valid_sequences = [], [], [], [], []
        prediction_status = Counter()
        dt, m = task.model.opt.timestep, task.model
        for window in events:
            clock = window['step']
            if clock >= len(q):
                continue
            state = dict(qpos=q[clock - 1] if clock else initial_q,
                         qvel=v[clock - 1] if clock else initial_v)
            packet = task.packet(state, clock, contacts[clock - 1] if clock else [])
            n = min(task.profile.ctrl_steps, len(q) - clock)
            for pred in window.get('predictions', []):
                prediction_status[pred['status']] += 1
            observed = []
            expected = sorted([e for e in window['events'] if e['start_time'] <= n * dt], key=lambda e: e['start_time'])
            for e in expected:
                first = None
                for i in range(n):
                    match = any({c['geom1'], c['geom2']} == {e['geom1'], e['geom2']} and c['force'][0] > .002 for c in contacts[clock + i])
                    if match:
                        first = (i + 1) * dt; break
                observed.append(first)
            if expected:
                good = all(t is not None and abs(t - e['start_time']) <= max(.01, 2 * dt) for t, e in zip(observed, expected))
                good &= all(a <= b + 1e-12 for a, b in zip(observed, observed[1:]) if a is not None and b is not None)
                valid_sequences.append(bool(good))
            for i in range(n):
                if task.name != 'h1_jog':
                    adr = task.adr['pos']
                    object_errors.append(np.linalg.norm(sensor[clock + i, adr:adr + 3] - packet['ref'][i + 1, adr:adr + 3]))
                    axis = task.adr['axis' if task.name == 'sharpa_pen' else 'normal']
                    actual = sensor[clock + i, axis:axis + 3]
                    target = packet['ref'][i + 1, axis:axis + 3]
                    angle_errors.append(np.arccos(np.clip(actual @ target / max(np.linalg.norm(actual) * np.linalg.norm(target), 1e-12), -1, 1)))
                task.state.qpos[:] = q[clock + i]
                mujoco.mj_kinematics(m, task.state)
                for e in window['events']:
                    t = (i + 1) * dt
                    if not e['start_time'] <= t <= e['start_time'] + e['duration']:
                        continue
                    g1, g2 = e['geom1'], e['geom2']
                    R1, R2 = task.state.geom_xmat[g1].reshape(3, 3), task.state.geom_xmat[g2].reshape(3, 3)
                    p1 = task.state.geom_xpos[g1] + R1 @ e['link_local_point']
                    p2 = task.state.geom_xpos[g2] + R2 @ e['object_local_point']
                    n1, n2 = R1 @ e['link_local_normal'], R2 @ e['object_local_normal']
                    distances.append(float(np.linalg.norm(p1 - p2)))
                    normals.append(float(np.arccos(np.clip(-n1 @ n2, -1, 1))))
        audit = dict(contact_point_rms_m=float(np.sqrt(np.mean(np.square(distances)))) if distances else None,
            native_constraint_record_count=len(flat_contacts),
            force_bearing_positive_geometric_gap_records=sum(c['force'][0] > .002 and c['geometric_distance'] > 0 for c in flat_contacts),
            force_bearing_geometrically_touching_pair_records=sum(c['force'][0] > .002 and c['geometric_distance'] <= 0 for c in flat_contacts),
            contact_label_definition='Observed native normal constraint force >2mN. Geometric pair touching uses mj_geomDistance <=0; positive gaps can carry margin or speculative constraint force.',
            peak_normal_contact_force_n=max((c['force'][0] for c in flat_contacts), default=0.),
            peak_friction_cone_violation_n=max((c['friction_violation'] for c in flat_contacts), default=0.),
            mean_contact_slip_m_s=float(np.mean([c['slip'] for c in flat_contacts])) if flat_contacts else None,
            peak_abs_joint_actuator_torque_nm=float(np.max(np.abs(joint_torque))) if len(joint_torque) else None,
            contact_normal_mean_error_rad=float(np.mean(normals)) if normals else None,
            evaluated_contact_points=len(distances), evaluated_contact_sequences=len(valid_sequences),
            contact_sequence_valid_rate=float(np.mean(valid_sequences)) if valid_sequences else None,
            object_position_rms_m=float(np.sqrt(np.mean(np.square(object_errors)))) if object_errors else None,
            object_axis_rms_rad=float(np.sqrt(np.mean(np.square(angle_errors)))) if angle_errors else None,
            chosen_forecast_status_counts=dict(prediction_status),
            sequence_definition='Within executed windows only: expected pair force >2mN within max(10ms,2dt), preserving order; future-only events excluded',
            object_reference_definition='Adaptive source task reference at each executed post-state time; pen rotation about its longitudinal axis is not constrained')
        audit['planning_phase_mean_seconds'] = {k: float(np.mean([r['phase_seconds'][k] for r in timed])) for k in timing_keys} if timed else None
        audit['planning_phase_timed_windows'] = len(timed)
        audit['planning_phase_total_windows'] = len(iterations)
        native_timed = [v['phase_seconds'] for v in validations if 'phase_seconds' in v]
        audit['cpu4_candidate_phase_mean_seconds'] = {k: float(np.mean([v[k] for v in native_timed]))
            for k in ('command_forecast', 'dynamics_and_cost', 'realized_prediction')} if native_timed else None
        audit['timing_definition'] = 'Host elapsed time, including synchronization inside existing calls; CPU candidate phases run concurrently and must not be added to validation wall time. Older snapshots without phase instrumentation are explicitly null.'
        if tracking_flag:
            config = json.loads((directory / 'config.json').read_text())
            reference_path = Path(config['task']['options']['physics_reference'])
            if not reference_path.exists():
                reference_path = campaign.parent / reference_path.parent.name / reference_path.name
            with np.load(reference_path, allow_pickle=False) as reference:
                q0 = reference['qpos'][0]
                quaternion_dot = np.clip(abs(initial_q[3:7] @ q0[3:7]), 0., 1.)
                audit['initial_reference_perturbation'] = dict(
                    translation_m=(initial_q[:3] - q0[:3]).tolist(),
                    rotation_deg=float(np.degrees(2 * np.arccos(quaternion_dot))),
                    joint_delta_min_rad=float(np.min(initial_q[7:] - q0[7:])),
                    joint_delta_max_rad=float(np.max(initial_q[7:] - q0[7:])))
        # Original SimMPC has separate single-world moved-mean validation.
        audit['gpu_world_evaluations_including_padding'] = result.get('gpu_rollouts', 0)
        audit['gpu_work_note'] = 'Original SimMPC uses its separate single-world validation; both CEM methods score equal unpadded proposal populations. Graph warmup rollouts are separately recorded in summary.cold_gpu_rollouts.'
        windows = (profile['steps'] + profile['ctrl_steps'] - 1) // profile['ctrl_steps']
        expected = profile['samples'] * ((profile['initial_iterations'] or profile['iterations'])
            + max(0, windows - 1) * profile['iterations'])
        audit['rollout_budget'] = dict(scored_proposals_per_iteration=profile['samples'],
            complete_cem_expected_gpu_rollouts=expected if method in ('cem', 'contact_cem') else None,
            complete_cem_budget_matches=(result.get('gpu_rollouts', 0) == expected)
                if method in ('cem', 'contact_cem') and result.get('complete') else None,
            cpu_shortlist_limit=runtime['defaults']['cpu_candidates'], previous_plan_additional_slots=1,
            cold_gpu_rollouts=summary.get('cold_gpu_rollouts'),
            auxiliary_work='Contact IK/LM and command/realized prediction are timed separately from scored rollout counts')

        write_json(directory / 'trajectory_audit.json', audit)
        result.update(summary)
        if isinstance(result.get('tracking'), dict):
            result['tracking_metrics'] = result['tracking']
        result['tracking'] = tracking_flag
        result['audit'] = audit
        groups[(result['task'], result['method'], result['tracking'])].append(result)
    aggregate = []
    for (task, method, tracking), rows in groups.items():
        success = sum(r['task_success'] for r in rows)
        gpu = sum(r.get('gpu_rollouts', 0) for r in rows)
        cpu = sum(r.get('cpu4_rollouts', 0) for r in rows)
        aggregate.append(dict(task=task, method=method, tracking=tracking, seeds=[r['seed'] for r in rows],
            successes=success, runs=len(rows), completed=sum(r.get('complete', False) for r in rows),
            success_rate=success / len(rows), gpu_rollouts=gpu, cpu4_rollouts=cpu,
            rollouts_per_success=(gpu + cpu) / success if success else None,
            total_wall_seconds=sum(r['wall_seconds'] for r in rows),
            cpu_workers=sorted({r.get('cpu_workers', 1) for r in rows}),
            cold_start_to_first_control_mean_seconds=float(np.mean([r['cold_start_to_first_control_seconds']
                for r in rows if r.get('cold_start_to_first_control_seconds') is not None]))
                if any(r.get('cold_start_to_first_control_seconds') is not None for r in rows) else None,
            steady_solve_mean_seconds=float(np.mean([r['steady_solve_mean_seconds'] for r in rows if r.get('steady_solve_mean_seconds') is not None]))
                if any(r.get('steady_solve_mean_seconds') is not None for r in rows) else None))
    expected_cases = {(task, method, seed, False)
        for task, seeds in [('h1_jog', [0, 1, 2]), ('sharpa_pen', [0, 1, 2]), ('sharpa_coin', [0, 1, 2, 10000])]
        for method in ('source_sampling', 'cem', 'contact_cem') for seed in seeds}
    expected_cases |= {('h1_jog', method, seed, True)
        for method in ('source_sampling', 'cem', 'contact_cem') for seed in (0, 1, 2)}
    recorded_cases = {(r['task'], r['method'], r['seed'], r['tracking']) for r in results}
    write_json(campaign / 'verification.json', dict(expected_cases=len(expected_cases),
        recorded_cases=len(results), matrix_complete=recorded_cases == expected_cases and len(results) == len(expected_cases),
        missing_cases=sorted(expected_cases - recorded_cases),
        unexpected_cases=sorted(recorded_cases - expected_cases),
        full_cem_budget_checks=[dict(run=r['run'], **r['audit']['rollout_budget'])
            for r in results if 'audit' in r and r['method'] in ('cem', 'contact_cem')]))
    write_json(campaign / 'aggregate.json', aggregate)
    write_json(campaign / 'audited_results.json', results)
    text = ['# Contact-space CEM benchmark', '',
        f'Recorded cases: {len(results)}/39. This report covers only recorded cases; the fixed matrix is complete only at 39 distinct expected cases.', '',
        '[Implementation/source map](IMPLEMENTATION.md) · [Matrix and budget verification](verification.json) · [Robokit source hashes](contact_robokit_source_integrity.json) · [Integration checks](contact_robokit_final_checks.log) · [CPU4 geometry checks](contact_robokit_cpu4_geometry.log)', '',
        'All controllers extend the original vendored Robokit library. Source commit and per-file hashes are recorded in third_party/robokit/SOURCE.json; migrated example factory changes are recorded separately. Earlier rewritten-controller experiments are diagnostic only and excluded.', '',
        'GPU: MuJoCo 3.10.0 / mujoco-warp 3.10.0.3 / warp-lang 1.16.0, discrete contacts. Execution and final validation: native MuJoCo 4.0.0 speculative contacts.', '',
        'Full-task pass criteria are [fixed in the implementation documentation](../../docs/contact_cem.md). Partial execution and runtime errors never pass. Source-sampling calls the vendored original Robokit SimMPC under the same CPU4 review and execution boundary.', '',
        '## Aggregate', '', '| Task | Method | Tracking | Full runs | Pass | GPU rollouts | CPU4 rollouts | Cold first control s | Steady solve s |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in aggregate:
        steady = f"{r['steady_solve_mean_seconds']:.3f}" if r['steady_solve_mean_seconds'] is not None else '—'
        cold = f"{r['cold_start_to_first_control_mean_seconds']:.3f}" if r['cold_start_to_first_control_mean_seconds'] is not None else '—'
        text.append(f"| {r['task']} | {r['method']} | {r['tracking']} | {r['completed']}/{r['runs']} | {r['successes']}/{r['runs']} | {r['gpu_rollouts']} | {r['cpu4_rollouts']} | {cold} | {steady} |")
    text += ['', '## Individual runs', '', '| Run | Full | Pass | Result |', '|---|---|---|---|']
    for r in results:
        if r['task'] == 'h1_jog':
            detail = f"velocity error={r.get('velocity_tracking_error')} ({r.get('velocity_tracking_target', '2m_s_command_ramp')}), fell={r.get('fell')}"
            if r.get('tracking'):
                detail += f"; original command error={r.get('source_command_velocity_error')}"
        elif r['task'] == 'sharpa_pen':
            detail = f"turns={r.get('turns')}, dropped={r.get('dropped')}"
        else:
            detail = f"goals={r.get('goals')}, first drop={r.get('first_drop_frame')}, jitter={r.get('jitter_rad_s2')}"
            detail += f"; first-six flips={r.get('flip_angles_deg', [])[:6]}; penetration={r.get('penetration_m')} m, self-overlap={r.get('self_overlap_m')} m"
        if r.get('runtime_error'):
            detail += '; runtime failure (see failure.txt)'
        if (campaign.parent / r['run'] / 'video.mp4').exists():
            detail += f"; [video](../{r['run']}/video.mp4)"
        artifact = 'summary.json' if (campaign.parent / r['run'] / 'summary.json').exists() else 'status.json'
        text.append(f"| [{r['run']}](../{r['run']}/{artifact}) | {r.get('complete', False)} | {r['task_success']} | {detail} |")
    text += ['', '## Interpretation', '',
        'Task generation, tracking an existing robot trajectory and human-to-robot retargeting are different claims. This experiment does not implement human skeleton mapping.', '',
        'Tracking compares the frozen robot motion, while jog generation compares the original 2 m/s command. Both errors are reported for tracking. Reproducing a source trajectory that missed its command does not establish original-command success or human-to-robot retargeting.', '',
        'Contact/object trajectory diagnostics are in each trajectory_audit.json. Forecasts are kinematic proposals and realized-path diagnostics, while native speculative constraints use pre-step velocity at every actual dynamics step.', '',
        'GPU/CPU physics differ; final costs and acceptance use CPU4. This campaign alone does not isolate the benefit of speculative contacts, and a small seed set cannot establish general superiority.', '']
    text += ['The original Robokit SimMPC baseline retains its moved-mean validation and coarse schedule. The two CEM methods use the same scored proposal budget and CPU4 shortlist limits; startup and auxiliary LM/prediction work are recorded separately.', '']
    if (campaign / 'native_speculative_check.json').exists():
        text += ['The independent [thin-wall on/off check](native_speculative_check.json) records native constraints and geometric versus effective distances. It validates the engine feature separately from task-level controller success.', '']
    text += ['Per-window LM, GPU dynamics and CPU4 validation timings, plus per-candidate forecast/dynamics timings, are recorded in trajectory_audit.json. Final-matrix runs use the same nine-worker CPU4 service, capped by CPU availability. Wall times are observations in the shared development environment; excluded older diagnostics are not included in these comparisons.', '']
    diagnostics = campaign / 'diagnostic_results_before_native_fallback.json'
    if diagnostics.exists():
        text += [f'[Earlier diagnostic results]({diagnostics.name}) retain runs superseded after removing an overly strict rejection of uncertain commanded-path forecasts. Their individual artifact directories are retained; they are excluded from the final matrix. UNKNOWN remains recorded and native speculative dynamics supplies the authoritative evaluation.', '']
    (campaign / 'REPORT.md').write_text('\n'.join(text))
    print(campaign / 'REPORT.md')


if __name__ == '__main__':
    main()
