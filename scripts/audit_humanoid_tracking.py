#!/usr/bin/env python3
"""Measure hand tracking and balance on saved executed states, frame by frame."""
import argparse
import csv
from dataclasses import fields
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.humanoid import HumanoidProblem, Settings, Weights, rotation_error


def tracking_errors(problem, actual, reference):
    """Both MjData inputs must have current forward kinematics/subtree velocity."""
    p, s, w = problem, problem.settings, problem.weights
    wrist_ids = [p.model.body(side + '_wrist_yaw_link').id for side in ('left', 'right')]
    root_delta = actual.xpos[p.root] - reference.xpos[p.root]
    wrist_delta = actual.xpos[wrist_ids] - reference.xpos[wrist_ids]
    result = dict(root_error_m=float(np.linalg.norm(root_delta)),
                  root_height_m=float(actual.qpos[2]), reference_root_height_m=float(reference.qpos[2]),
                  height_deficit_m=float(reference.qpos[2] - actual.qpos[2]))
    for i, side in enumerate(('left', 'right')):
        result[side + '_wrist_world_error_m'] = float(np.linalg.norm(wrist_delta[i]))
        result[side + '_wrist_pelvis_error_m'] = float(np.linalg.norm(wrist_delta[i] - root_delta))
        result[side + '_wrist_rotation_error_deg'] = float(np.degrees(np.linalg.norm(rotation_error(
            actual.xmat[wrist_ids[i]], reference.xmat[wrist_ids[i]]))))
    relative = (actual.xpos[p.bodies] - actual.xpos[p.root]) - (reference.xpos[p.bodies] - reference.xpos[p.root])
    result['keypoint_position_cost'] = float(w.keypoint_position * np.sum(relative ** 2))
    hand_indices = [list(p.bodies).index(body) for body in wrist_ids]
    result['hand_keypoint_position_cost'] = float(w.keypoint_position * np.sum(relative[hand_indices] ** 2))
    dc = actual.subtree_com[p.root] - reference.subtree_com[p.root]
    dv = actual.subtree_linvel[p.root] - reference.subtree_linvel[p.root]
    dcm = dc + dv / np.sqrt(9.81 / max(reference.subtree_com[p.root, 2] - p.ground_height, .2))
    up = actual.xmat[p.torso].reshape(3, 3)[:, 2]
    up_ref = reference.xmat[p.torso].reshape(3, 3)[:, 2]
    result.update(com_xy_error_m=float(np.linalg.norm(dc[:2])), com_height_error_m=float(dc[2]),
                  com_velocity_error_m_s=float(np.linalg.norm(dv)), dcm_xy_error_m=float(np.linalg.norm(dcm[:2])),
                  torso_tilt_deg=float(np.degrees(np.arctan2(np.linalg.norm(np.cross(up, up_ref)), up @ up_ref))),
                  upright_cost=float(w.upright * np.sum((up - up_ref) ** 2) / s.upright_scale ** 2))
    parts = dict(com_xy=np.sum(dc[:2] ** 2) / s.balance_xy_scale ** 2,
                 com_height=dc[2] ** 2 / s.balance_z_scale ** 2,
                 dcm=.5 * np.sum(dcm[:2] ** 2) / s.balance_xy_scale ** 2,
                 vertical_velocity=.25 * dv[2] ** 2 / s.balance_velocity_scale ** 2,
                 collapse=(max(result['height_deficit_m'] - s.balance_height_margin, 0) / s.balance_height_scale) ** 2)
    result.update({'balance_' + key + '_cost': float(w.balance * value) for key, value in parts.items()})
    result['balance_cost'] = float(w.balance * sum(parts.values()))
    result['balance_failure'] = bool(result['height_deficit_m'] > .18 or result['torso_tilt_deg'] > 60)
    if p.surface:
        geom = p.object_geom
        actual_rot = actual.geom_xmat[geom].reshape(3, 3)
        ref_rot = reference.geom_xmat[geom].reshape(3, 3)
        local = (actual.geom_xpos[p.hand_geoms] - actual.geom_xpos[geom]) @ actual_rot
        local_ref = (reference.geom_xpos[p.hand_geoms] - reference.geom_xpos[geom]) @ ref_rot
        projected = p.surface.project(local)
        normals = p.surface.normal(local, projected)
        signed_gap = np.sum((local - projected) * normals, axis=1) - p.model.geom_size[p.hand_geoms, 0]
        ref_anchor=p.surface.project(p.features(reference)['contacts'][:2])
        ref_normal=p.surface.normal(ref_anchor,ref_anchor)
        task_center=(ref_anchor+p.model.geom_size[p.hand_geoms,:1]*ref_normal)@ref_rot.T+reference.geom_xpos[geom]
        for i, side in enumerate(('left', 'right')):
            result[side + '_hand_world_error_m'] = float(np.linalg.norm(
                actual.geom_xpos[p.hand_geoms[i]] - reference.geom_xpos[p.hand_geoms[i]]))
            result[side + '_hand_box_error_m'] = float(np.linalg.norm(local[i] - local_ref[i]))
            result[side + '_hand_box_gap_m'] = float(signed_gap[i])
            result[side + '_hand_task_error_m'] = float(np.linalg.norm(actual.geom_xpos[p.hand_geoms[i]]-task_center[i]))
        result.update(box_position_error_m=float(np.linalg.norm(actual.geom_xpos[geom] - reference.geom_xpos[geom])),
                      box_height_m=float(actual.geom_xpos[geom, 2]),
                      reference_box_height_m=float(reference.geom_xpos[geom, 2]))
        for axis, name in enumerate(('x', 'y', 'z')):
            result['box_' + name + '_m'] = float(actual.geom_xpos[geom, axis])
            result['reference_box_' + name + '_m'] = float(reference.geom_xpos[geom, axis])
    return result


def collect(case):
    case = Path(case)
    config = json.loads((case / 'config.json').read_text())
    settings = Settings(**{key: value for key, value in config['settings'].items()
                           if key in {f.name for f in fields(Settings)}})
    p = HumanoidProblem(config['scene'], config['reference'], ref_dt=config['ref_dt'],
                        settings=settings, weights=Weights(**config['weights']))
    with np.load(case / 'rollout.npz', allow_pickle=False) as arrays:
        arrays = {key: arrays[key] for key in arrays.files}
    logs = [json.loads(line) for line in (case / 'iterations.jsonl').read_text().splitlines()]
    frame_ids = np.r_[arrays['reference_frames'][0] - 1, arrays['reference_frames']]
    actual, ref = mujoco.MjData(p.model), mujoco.MjData(p.model)
    records = []
    for index, frame in enumerate(frame_ids):
        actual.qpos[:], actual.qvel[:] = arrays['qpos'][index], arrays['qvel'][index]
        ref.qpos[:], ref.qvel[:] = p.qref[frame], p.tracking_vref[frame]
        for state in (actual, ref):
            mujoco.mj_forward(p.model, state)
            mujoco.mj_subtreeVel(p.model, state)
        row = dict(frame=int(frame), time_s=float(index * p.ref_dt), **tracking_errors(p, actual, ref))
        if index:
            for role, side in enumerate(('left', 'right')):
                row[side + '_hand_contact_active'] = bool(arrays['contact_mask'][index - 1, role])
                row[side + '_hand_consensus_m'] = float(np.linalg.norm(
                    arrays['x'][index - 1, role] - arrays['z'][index - 1, role]))
            last = logs[index - 1]['iterations'][-1]
            row.update({'predicted_' + name + '_cost': value for name, value in last['costs'].items()})
            row['robot_support_ratio'] = last['executed_support_ratio']
        else:
            for side in ('left', 'right'):
                row[side + '_hand_contact_active'] = bool(p.mask[frame, ('left', 'right').index(side)])
        records.append(row)
    def stats(key, selected=records):
        values = np.array([row[key] for row in selected])
        return dict(rms=float(np.sqrt(np.mean(values ** 2))), mean=float(values.mean()),
                    min=float(values.min()), max=float(values.max()), final=float(values[-1]))
    scalar_keys = [key for key, value in records[-1].items()
                   if isinstance(value, (int, float)) and not isinstance(value, bool)
                   and key not in ('frame', 'time_s') and not key.startswith('predicted_')]
    failures = [row['frame'] for row in records if row['balance_failure']]
    summary = dict(case=str(case), executed_frames=len(records) - 1, settings=config['settings'],
                   completed_requested_frames=config['completed_requested_frames'],
                   balance_failure_frames=len(failures), first_balance_failure_frame=failures[0] if failures else None,
                   metrics={key: stats(key, [row for row in records if key in row]) for key in scalar_keys},
                   notes=['Hand world errors use the real lh/rh sphere centers; wrist errors also report the exact optimized body keypoint.',
                          'Hand box errors use each state\'s box geom frame; signed gap is positive for separation.',
                          'Balance uses exact kinematic COM velocity. The optimizer uses adjacent 60 Hz COM differences; these instantaneous costs are diagnostics, not the logged horizon cost.',
                          'Predicted costs are the selected candidate\'s full horizon terms before the dual update.',
                          'Failure diagnostic: reference-relative pelvis height deficit >0.18m or torso tilt >60deg.'])
    for side in ('left', 'right'):
        active = [row for row in records[1:] if row[side + '_hand_contact_active']]
        summary[side + '_active_hand'] = dict(frames=len(active), first_frame=active[0]['frame'] if active else None,
            **({key: stats(side + '_' + key, active) for key in ('hand_world_error_m', 'hand_task_error_m', 'hand_box_error_m', 'hand_box_gap_m', 'hand_consensus_m')} if active else {}),
            near_surface_frames=sum(abs(row[side + '_hand_box_gap_m']) < .002 for row in active))
    if p.surface:
        summary['box_lift_m'] = max(row['box_height_m'] for row in records) - records[0]['box_height_m']
        summary['reference_box_lift_m'] = max(row['reference_box_height_m'] for row in records) - records[0]['reference_box_height_m']
    return records, summary


def save(records, summary, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'frames.json').write_text(json.dumps(records, indent=2) + '\n')
    with (output / 'frames.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in records for key in row)))
        writer.writeheader()
        writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    time = [row['time_s'] for row in records]
    groups = [(['left_wrist_world_error_m', 'right_wrist_world_error_m', 'left_wrist_pelvis_error_m', 'right_wrist_pelvis_error_m'], 'Wrist tracking [m]'),
              (['left_hand_box_error_m', 'right_hand_box_error_m', 'left_hand_box_gap_m', 'right_hand_box_gap_m'], 'Hand vs actual box [m]'),
              (['com_xy_error_m', 'com_height_error_m', 'dcm_xy_error_m'], 'COM / DCM error [m]'),
              (['torso_tilt_deg'], 'Torso tilt vs reference [deg]'),
              (['balance_cost', 'upright_cost', 'keypoint_position_cost', 'hand_keypoint_position_cost'], 'Instant executed cost'),
              (['box_height_m', 'reference_box_height_m', 'root_height_m', 'reference_root_height_m'], 'Height [m]')]
    for ax, (keys, label) in zip(axes.flat, groups):
        for key in keys:
            if key in records[0]:
                ax.plot(time, [row[key] for row in records], label=key, linewidth=1.2)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel('Executed simulation time [s]')
    fig.suptitle(Path(summary['case']).name)
    fig.tight_layout()
    fig.savefig(output / 'tracking_balance.png', dpi=160)
    plt.close(fig)
    if 'box_x_m' in records[0]:
        fig, ax = plt.subplots(figsize=(6, 5))
        for prefix, label in (('', 'Executed box'), ('reference_', 'Reference box')):
            ax.plot([row[prefix + 'box_x_m'] for row in records],
                    [row[prefix + 'box_y_m'] for row in records], label=label)
        for index in (0, min(40, len(records) - 1), min(80, len(records) - 1), len(records) - 1):
            row = records[index]
            ax.scatter(row['box_x_m'], row['box_y_m'], s=18, color='tab:blue')
            ax.annotate(str(row['frame']), (row['box_x_m'], row['box_y_m']))
        ax.set(xlabel='World x [m]', ylabel='World y [m]', title='Actual box trajectory')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / 'box_xy.png', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records, summary = collect(args.case)
    save(records, summary, args.output)
    print(json.dumps(summary, indent=2))
