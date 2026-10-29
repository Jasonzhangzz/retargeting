#!/usr/bin/env python3
"""Summarize executed humanoid trajectories without treating finite states as success."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import mujoco

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def summarize(case):
    case = Path(case)
    config = json.loads((case/'config.json').read_text())
    records = [json.loads(line) for line in (case/'iterations.jsonl').read_text().splitlines()]
    with np.load(case/'rollout.npz', allow_pickle=False) as result:
        mask = result['contact_mask'].astype(bool)
        distance = np.linalg.norm(result['x']-result['z'], axis=-1)
        frames = result['reference_frames']
        if len(frames) == 0:
            raise ValueError(f'No executed transitions in {case}')
        with np.load(config['reference'], allow_pickle=False) as arrays:
            reference = arrays['qpos'][np.r_[frames[0]-1, frames]]
        settings = config['settings']
        primal_tol, dual_tol = settings['primal_tolerance'], settings['dual_tolerance']
        contact_max = float(distance[mask].max()) if mask.any() else 0.
        contact_rms = float(np.sqrt(np.mean(distance[mask]**2))) if mask.any() else 0.
        steps = [step for row in records for step in row['iterations']]
        # Evaluate tracking on the actually executed states, independently of
        # the optimizer's predicted costs and its selected future trajectory.
        model=mujoco.MjModel.from_xml_path(config['scene'])
        data=mujoco.MjData(model)
        body_names=('pelvis','torso_link','left_shoulder_pitch_link','right_shoulder_pitch_link',
                    'left_elbow_link','right_elbow_link','left_wrist_yaw_link','right_wrist_yaw_link',
                    'left_hip_pitch_link','right_hip_pitch_link','left_knee_link','right_knee_link',
                    'left_ankle_roll_link','right_ankle_roll_link')
        bodies=[model.body(name).id for name in body_names]
        keypoint_error=[];relative_keypoint_error=[];rotation_error=[];torso_tilt=[]
        for q,ref in zip(result['qpos'],reference):
            data.qpos[:]=q;mujoco.mj_forward(model,data)
            points=data.xpos[bodies].copy();pelvis=data.xpos[model.body('pelvis').id].copy()
            up=data.xmat[model.body('torso_link').id].reshape(3,3)[:,2].copy()
            data.qpos[:]=ref;mujoco.mj_forward(model,data)
            keypoint_error.append(np.sum((points-data.xpos[bodies])**2,axis=1))
            relative_keypoint_error.append(np.sum(((points-pelvis)-(data.xpos[bodies]-data.xpos[model.body('pelvis').id]))**2,axis=1))
            up_ref=data.xmat[model.body('torso_link').id].reshape(3,3)[:,2]
            torso_tilt.append(float(np.arccos(np.clip(up@up_ref,-1,1))))
            error=np.empty(3);mujoco.mju_subQuat(error,q[3:7],ref[3:7]);rotation_error.append(error@error)
        failure=(result['qpos'][:,2]<reference[:,2]-.18)|(np.array(torso_tilt)>np.pi/3)
        failure_ids=np.flatnonzero(failure)
        reference_frames=np.r_[frames[0]-1,frames]
        limited=np.flatnonzero(model.jnt_limited)
        actual_joint=result['qpos'][:,model.jnt_qposadr[limited]]
        violation=np.maximum(model.jnt_range[limited,0]-actual_joint,actual_joint-model.jnt_range[limited,1])
        from retarget.humanoid import HumanoidProblem
        problem=HumanoidProblem(config['scene'],config['reference'],ref_dt=config['ref_dt'])
        physical_gaps=[]
        for q,v in zip(result['qpos'][1:],result['qvel'][1:]):
            problem.data.qpos[:]=q;problem.data.qvel[:]=v
            mujoco.mj_forward(problem.model,problem.data)
            contacts=problem.features(problem.data)['contacts']
            gap=np.zeros(4)
            if problem.surface:
                gap[:2]=np.linalg.norm(contacts[:2]-problem.surface.project(contacts[:2]),axis=1)
            gap[2:]=np.abs(contacts[2:,2]-problem.ground_height)
            physical_gaps.append(gap)
        physical_gaps=np.asarray(physical_gaps)
        return dict(
            case=case.name, frames=len(frames), settings=settings, weights=config['weights'],
            seconds=config['seconds'],
            action_solver=config.get('action_solver','legacy_scipy_TRF'),
            gradient=config.get('gradient','legacy_finite_difference'),
            completed_requested_frames=config['completed_requested_frames'],
            executed_contact_max_mm=contact_max*1000,
            executed_contact_rms_mm=contact_rms*1000,
            actual_geometry_gap_rms_mm=float(np.sqrt(np.mean(physical_gaps[mask]**2)))*1000 if mask.any() else 0.,
            actual_geometry_gap_max_mm=float(physical_gaps[mask].max())*1000 if mask.any() else 0.,
            contact_consensus_within_tolerance=contact_max < primal_tol,
            root_position_rms_m=float(np.sqrt(np.mean(np.sum(
                (result['qpos'][:, :3]-reference[:, :3])**2, axis=1)))),
            root_height_min_m=float(result['qpos'][:, 2].min()),
            keypoint_position_rms_m=float(np.sqrt(np.mean(keypoint_error))),
            keypoint_relative_position_rms_m=float(np.sqrt(np.mean(relative_keypoint_error))),
            torso_tilt_max_deg=float(np.degrees(max(torso_tilt))),
            height_deficit_max_m=float(np.max(reference[:,2]-result['qpos'][:,2])),
            joint_limit_violation_max_rad=float(max(violation.max(),0.)),
            balance_audit_failure_frames=int(failure.sum()),
            first_balance_audit_failure_frame=int(reference_frames[failure_ids[0]]) if len(failure_ids) else None,
            no_balance_failure_under_audit=not bool(failure.any()),
            balance_audit_rule='reference-relative root height deficit > 0.18m OR torso up-axis tilt >60deg; diagnostic threshold, not a universal stability certificate',
            root_rotation_rms_rad=float(np.sqrt(np.mean(rotation_error))),
            finite_states=bool(np.isfinite(result['qpos']).all() and np.isfinite(result['qvel']).all()),
            contact_mask_counts=mask.sum(axis=0).tolist(),
            converged_contact_windows=sum(
                row['iterations'][-1]['primal_max_m'] < primal_tol and
                row['iterations'][-1]['dual_max_rho_m'] < dual_tol for row in records),
            windows_without_active_contacts=int((~mask.any(axis=1)).sum()),
            action_descent_violations=sum(
                step['action_cost_after'] > step['action_cost_before']+1e-8 for step in steps),
            contact_descent_violations=sum(
                step['contact_cost_after'] > step['contact_cost_before']+1e-8 for step in steps
                if 'contact_cost_after' in step),
            note='executed_contact measures material-point ADMM consensus; actual_geometry_gap independently measures the nearest collision surface/lowest foot support gap. Neither alone proves stable tracking.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    summaries = [summarize(case) for case in args.cases]
    content = json.dumps(summaries, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    for item in summaries:
        print(f'{item["case"]}: frames={item["frames"]}, '
              f'contact RMS/max={item["executed_contact_rms_mm"]:.2f}/{item["executed_contact_max_mm"]:.2f} mm, '
              f'root RMS={item["root_position_rms_m"]:.3f} m, '
              f'contact feasible={item["contact_consensus_within_tolerance"]}')
