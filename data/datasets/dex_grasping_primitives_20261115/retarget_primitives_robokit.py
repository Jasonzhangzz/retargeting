#!/usr/bin/env python3
"""Batch independent frozen MANO primitive poses to calibrated glove or Wuji2.

Uses Robokit's population/LM hand retargeter, not the anatomical-angle exporter.
The source poses, object frames and original acceptance flags are preserved.
This is kinematic retargeting; no new force-closure or grasp-success claim is made.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
CHAINS = tuple(tuple(f'{f}_{j}' for j in
                     (('cmc', 'mcp', 'ip', 'tip') if f == 'thumb' else
                      ('mcp', 'pip', 'dip', 'tip'))) for f in FINGERS)
POINT_NAMES = ('wrist',) + tuple(n for chain in CHAINS for n in chain)
TIP_IDS = np.array([4, 8, 12, 16, 20])
PALM_IDS = np.array([5, 9, 13, 17])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def numbers(a):
    return ' '.join(format(float(x), '.17g') for x in np.asarray(a).reshape(-1))


def glove_xml(candidate):
    """Export the frozen personalized FK exactly, including serial MCP twists."""
    definition = candidate['bone_extrinsics']
    rig = definition['rig']
    root = ET.Element('mujoco', model='calibrated_moxian_glove_29')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true', fusestatic='false')
    world = ET.SubElement(root, 'worldbody')
    wrist = rig['wrist_body']
    nodes = {wrist: ET.SubElement(world, 'body', name=rig['body_names'][wrist])}
    scales = np.exp(definition['length_log_scales'])
    for body in range(wrist + 1, len(rig['parents'])):
        parent = rig['parents'][body]
        if parent not in nodes:
            raise ValueError('Glove body is outside the wrist subtree')
        quat = Rotation.from_matrix(rig['body_r0'][body]).as_quat()[[3, 0, 1, 2]]
        pos = np.array(rig['body_p0'][body]) * scales[rig['body_bone'][parent]]
        joints = rig['body_joints'][body]
        # Robokit accepts one hinge per MJCF body. Serial identity children
        # preserve the exact ordered pivot rotations of a multi-hinge body.
        node = ET.SubElement(nodes[parent], 'body',
                             name=rig['body_names'][body] + ('_serial_0' if len(joints)>1 else ''),
                             pos=numbers(pos), quat=numbers(quat))
        ET.SubElement(node, 'inertial', pos='0 0 0', mass='.001', diaginertia='1e-7 1e-7 1e-7')
        for index, j in enumerate(joints):
            if index:
                node = ET.SubElement(node, 'body', name=rig['body_names'][body])
                ET.SubElement(node, 'inertial', pos='0 0 0', mass='.001', diaginertia='1e-7 1e-7 1e-7')
            ET.SubElement(node, 'joint', name=rig['joint_names'][j], type='hinge',
                          pos=numbers(rig['joint_p0'][j]), axis=numbers(rig['joint_axis0'][j]),
                          range=numbers(rig['limits'][j]))
        nodes[body] = node
    links = {'wrist': rig['body_names'][wrist]}
    for i, finger in enumerate(FINGERS):
        original = 'little' if finger == 'pinky' else finger
        parts = ('metacarpal', 'proximal', 'distal', 'tip') if i == 0 else (
            'proximal', 'intermediate', 'distal', 'tip')
        links.update({key: f'right_{original}_{part}' for key, part in zip(CHAINS[i], parts)})
    return root, links, rig['joint_names'], np.array(definition['rest_angles_rad'])


def wuji_xml():
    import mujoco
    from retarget.env.grasp.hand import hand_tree
    root = hand_tree('wuji2')
    native = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    links = {'wrist': 'right_palm_link'}
    for i, chain in enumerate(CHAINS):
        for k, name in enumerate(chain[:3], 2):
            links[name] = f'right_finger{i+1}_link{k}'
        tip = f'right_{FINGERS[i]}_robokit_tip'
        site = native.site(f'right_{FINGERS[i]}_tip')
        body = root.find(f".//body[@name='right_finger{i+1}_link4']")
        ET.SubElement(body, 'body', name=tip, pos=numbers(site.pos))
        links[chain[-1]] = tip
    names = [native.joint(i).name for i in range(6, 26)]
    return root, links, names, np.array(native.qpos0[6:26]), native


def palm_basis(points):
    """Right-handed palm basis: index-to-pinky lateral, wrist-to-MCP forward."""
    x = points[..., 5, :] - points[..., 17, :]
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    forward = points[..., PALM_IDS, :].mean(-2) - points[..., 0, :]
    z = np.cross(x, forward)
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    return np.stack((x, y, z), axis=-1)


def packed_fk(robot, packed, link_ids, device):
    import warp as wp
    from robokit.utils.warp_utils import wp_vec7
    state = robot.state(q=wp.from_numpy(np.ascontiguousarray(packed[:, 7:]), dtype=wp.float32, device=device),
                        T_world_base=wp.from_numpy(np.ascontiguousarray(packed[:, :7]),
                                                  dtype=wp_vec7, device=device))
    robot.forward_kinematics(state)
    frames = state.T_world_link.numpy()[:, link_ids]
    return frames[..., :3], state


def validate_fk(robot, target, native_names, links, candidate, native, device):
    """Compare Robokit with an independent frozen FK, at random legal poses."""
    import mujoco
    names = list(robot.spec.actuated_joint_names)
    to_native = np.array([names.index(n) for n in native_names])
    rng = np.random.default_rng(20261115)
    limits = robot.spec.actuated_joint_limits
    q = rng.uniform(limits[:, 0], limits[:, 1], (24, len(names))).astype(np.float32)
    packed = np.zeros((len(q), 7 + len(names)), np.float32)
    packed[:, 3] = 1
    packed[:, 7:] = q
    link_ids = [robot.spec.link_names.index(links[n]) for n in POINT_NAMES]
    actual, _ = packed_fk(robot, packed, link_ids, device)
    expected = []
    if target == 'glove':
        from moxian_web.bone_extrinsics import BoneKinematics
        fk = BoneKinematics(candidate['bone_extrinsics'])
        ids = [fk.body_names.index(links[n]) for n in POINT_NAMES]
        for row in q[:, to_native]:
            expected.append(fk.forward(row)[1][ids])
    else:
        data = mujoco.MjData(native)
        for row in q[:, to_native]:
            data.qpos[:] = native.qpos0
            data.qpos[6:26] = row
            mujoco.mj_forward(native, data)
            points = [data.xpos[native.body('right_palm_link').id].copy()]
            for i, chain in enumerate(CHAINS):
                points.extend(data.xpos[native.body(links[n]).id].copy() for n in chain[:3])
                points.append(data.site_xpos[native.site(f'right_{FINGERS[i]}_tip').id].copy())
            expected.append(points)
    error = float(np.max(np.linalg.norm(actual - np.array(expected), axis=-1)))
    if error > 3e-6:
        raise ValueError(f'{target} FK mismatch: {error} m')
    return dict(random_poses=len(q), max_error_m=error), link_ids, to_native


def load_inputs(dataset):
    manifest = json.loads((dataset/'manifest.json').read_text())
    entries, points, origins = [], [], []
    for item in manifest['entries']:
        path = dataset/item['file']
        if sha256(path) != item['sha256']:
            raise ValueError(f'Source hash mismatch: {path}')
        with np.load(path, allow_pickle=False) as data:
            j = data['mano_joints'].astype(np.float32)
            if j.shape[1:] != (21, 3) or not np.isfinite(j).all():
                raise ValueError(f'Invalid MANO points: {path}')
            entries.append((item, len(points), len(j)))
            points.extend(j)
            origins.extend((item['type'], k) for k in range(len(j)))
    return manifest, entries, np.array(points), origins


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--glove-model', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--target', choices=('glove', 'wuji2'), required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--limit', type=int, help='Only for a separate smoke output')
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    import torch
    import warp as wp
    from robokit.robo import Robot
    from robokit.robo.robot_spec import RobotSpec
    from robokit.helpers.hand_retargeting import HandSpec, HandRetargeterOnline, HandRetargeterOnlineConfig
    from robokit.helpers.hand_retargeting.config import _online_solver
    from robokit.utils.ee_coord_utils import HandCoordinateSpec
    from robokit.utils.warp_utils import wp_vec7
    wp.init()
    wp.config.log_level = wp.LOG_WARNING
    manifest, entries, source_points, origins = load_inputs(args.dataset)
    if args.limit:
        source_points = source_points[:args.limit]
    candidate = None
    native = None
    if args.target == 'glove':
        candidate = json.loads(args.glove_model.read_text())
        tree, links, native_names, rest = glove_xml(candidate)
    else:
        tree, links, native_names, rest, native = wuji_xml()
    xml = ET.tostring(tree, encoding='unicode')
    (args.output/'model.xml').write_text(xml)
    if candidate:
        write_json(args.output/'glove-model.json', candidate)
    robot = Robot(RobotSpec.parse(xml, base_link_name=links['wrist'], load_meshes=False))
    names = list(robot.spec.actuated_joint_names)
    if set(names) != set(native_names):
        raise ValueError('Actuated joints differ from the native target')
    verification, link_ids, to_native = validate_fk(
        robot, args.target, native_names, links, candidate, native, args.device)
    print('FK_VERIFIED', json.dumps(verification), flush=True)
    name_to_rest = dict(zip(native_names, rest))
    # Only position/direction tasks are used; points are already in source world coordinates.
    spec = HandSpec(floating_base=True, target_names=POINT_NAMES, target_link_names=links,
                    target_chains=CHAINS, root_target_name='wrist',
                    contact_target_names=tuple(POINT_NAMES[i] for i in TIP_IDS),
                    root_link_coord_spec=HandCoordinateSpec('y', '-z'),
                    rest_q_by_name=name_to_rest)
    config = HandRetargeterOnlineConfig(reference_origin='wrist', base_init_sample_range=(0., 0., 0.))
    config.solver = _online_solver(iters=(12, 20, 12), widths=(16, 4, 1))
    config.task('positions').weight = {n: (24. if n.endswith('_tip') else 8. if n == 'wrist' else 4.) for n in POINT_NAMES}
    config.task('directions').weight = {(links[a], links[b], a, b): 10.
                                        for chain in CHAINS for a, b in zip(chain, chain[1:])}
    config.task('rest').weight = .001
    config.task('smoothness').weight = 0.
    config.task('self_collision').weight = 0.
    helper = HandRetargeterOnline(config, robot, spec, device=args.device)
    rest_packed = np.zeros((1, 7+len(names)), np.float32)
    rest_packed[:, 3] = 1
    rest_packed[:, 7:] = np.array([name_to_rest[n] for n in names])
    rest_points = packed_fk(robot, rest_packed, link_ids, args.device)[0][0]
    basis = palm_basis(rest_points)
    batch = min(args.batch_size, len(source_points))
    helper.warmup(batch)
    solved, initial_points, final_points, term_costs = [], [], [], {}
    began = time.perf_counter()
    for start in range(0, len(source_points), batch):
        points = source_points[start:start+batch]
        n = len(points)
        if n < batch:
            points = np.concatenate((points, np.repeat(points[-1:], batch-n, axis=0)))
        # Every row starts independently; sizes are not treated as a temporal trajectory.
        rotation = palm_basis(points) @ basis.T
        base = np.concatenate((points[:, 0], Rotation.from_matrix(rotation).as_quat()[:, [3, 0, 1, 2]]), axis=1).astype(np.float32)
        # Keep both state buffers explicitly on the selected device.
        initial = robot.state(q=wp.from_numpy(np.repeat(rest_packed[:, 7:], batch, axis=0), dtype=wp.float32, device=args.device),
                              T_world_base=wp.from_numpy(base, dtype=wp_vec7, device=args.device))
        init_packed = np.concatenate((base, np.repeat(rest_packed[:, 7:], batch, axis=0)), axis=1)
        ip = packed_fk(robot, init_packed, link_ids, args.device)[0]
        helper.reset()
        out = helper.solve_numpy(points, init_state=initial)
        fp, state = packed_fk(robot, out, link_ids, args.device)
        costs = helper.compute_costs(state)
        for name, value in costs.items():
            term_costs.setdefault(name, []).append(value.numpy()[:n].copy())
        solved.append(out[:n]); initial_points.append(ip[:n]); final_points.append(fp[:n])
        print(f'BATCH {min(start+n,len(source_points))}/{len(source_points)} {time.perf_counter()-began:.2f}s', flush=True)
    packed = np.concatenate(solved)
    final_points = np.concatenate(final_points)
    initial_points = np.concatenate(initial_points)
    q = packed[:, 7:][:, to_native]
    limits = robot.spec.actuated_joint_limits[to_native]
    finite = np.isfinite(packed).all(1) & np.isfinite(final_points).all((1, 2))
    within = ((q >= limits[:, 0]-1e-5) & (q <= limits[:, 1]+1e-5)).all(1)
    quat_ok = np.abs(np.linalg.norm(packed[:, 3:7], axis=1)-1) < 1e-5
    error = np.linalg.norm(final_points-source_points, axis=-1)
    baseline = np.linalg.norm(initial_points-source_points, axis=-1)
    result_ok = finite & within & quat_ok
    config_record = dict(
        method='robokit.HandRetargeterOnline', source='frozen MANO 21 joints',
        joint_names=native_names, keypoint_names=POINT_NAMES, keypoint_links=links,
        coordinate_frame='same world frame as source MANO and source object_pose_world',
        base_pose_format='xyz + quaternion_wxyz', joint_units='radians', position_units='metres',
        target_scale=1., source_units='metres',
        solver_iterations=[12,20,12], solver_candidates=[16,4,1],
        position_weights=config.task('positions').weight,
        direction_weight=10., rest_weight=.001, independent_rows=True,
        temporal_smoothing=False, collision_objective=False,
        force_closure_tested=False, physical_stability_tested=False,
        glove_model_hash=candidate['model_hash'] if candidate else None,
        glove_calibration_accepted=candidate.get('report',{}).get('accepted') if candidate else None,
        source_manifest_sha256=sha256(args.dataset/'manifest.json'),
        script_sha256=sha256(__file__), robokit_version=__import__('robokit').__version__,
        model_xml_sha256=sha256(args.output/'model.xml'), fk_verification=verification,
    )
    write_json(args.output/'config.json', config_record)
    (args.output/'grids').mkdir()
    sizes = []
    written = []
    for entry, offset, count in entries:
        count = min(count, len(source_points)-offset)
        if count <= 0:
            continue
        sl = slice(offset, offset+count)
        with np.load(args.dataset/entry['file'], allow_pickle=False) as data:
            payload = {k: data[k][:count] for k in (
                'size_scale','radius_m','height_m','object_pose_world','mano_joints','mano_vertices',
                'mano_angles','mano_variables','wuji_qpos','source_knot','valid',
                'patch_joint_accepted','grasp_passed','rollout_valid','rollout_stable')}
            payload['source_wuji_qpos_reference'] = payload.pop('wuji_qpos')
        payload.update(
            source_row_index=np.arange(count), target_qpos=q[sl], target_base_pose=packed[sl,:7],
            target_keypoints_world=final_points[sl], initial_keypoints_world=initial_points[sl],
            target_keypoint_error_m=error[sl], initial_keypoint_error_m=baseline[sl],
            retarget_numerically_valid=result_ok[sl], joint_limits_rad=limits,
            joint_names=np.array(native_names), keypoint_names=np.array(POINT_NAMES),
            target_kind=np.array(args.target),
        )
        for name, arrays in term_costs.items():
            payload['robokit_cost_'+name] = np.concatenate(arrays)[sl]
        path = args.output/'grids'/Path(entry['file']).name
        np.savez_compressed(path, **payload)
        written.append(dict(type=entry['type'], file=str(path.relative_to(args.output)),
                            samples=count, sha256=sha256(path)))
        for i in range(count):
            sizes.append(dict(type=entry['type'], kind=entry['kind'], row=i,
                              size_scale=float(payload['size_scale'][i]),
                              radius_m=float(payload['radius_m'][i]), height_m=float(payload['height_m'][i]),
                              source_grasp_passed=bool(payload['grasp_passed'][i]),
                              source_patch_joint_accepted=bool(payload['patch_joint_accepted'][i]),
                              retarget_numerically_valid=bool(result_ok[offset+i]),
                              fingertip_rmse_mm=float(np.sqrt(np.mean(error[offset+i,TIP_IDS]**2))*1000)))
    with (args.output/'size-index.csv').open('w') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(sizes[0]));writer.writeheader();writer.writerows(sizes)
    summary = dict(
        target=args.target, source_types=len(written), samples=len(source_points),
        numerically_valid=int(result_ok.sum()), joint_limit_violations=int((~within).sum()),
        source_contact_accepted=sum(int(row['source_patch_joint_accepted']) for row in sizes),
        source_grasp_passed=sum(int(row['source_grasp_passed']) for row in sizes),
        keypoint_rmse_mm=float(np.sqrt(np.mean(error**2))*1000),
        fingertip_rmse_mm=float(np.sqrt(np.mean(error[:,TIP_IDS]**2))*1000),
        initial_fingertip_rmse_mm=float(np.sqrt(np.mean(baseline[:,TIP_IDS]**2))*1000),
        fingertip_error_p50_mm=float(np.median(error[:,TIP_IDS])*1000),
        fingertip_error_p95_mm=float(np.quantile(error[:,TIP_IDS],.95)*1000),
        kinematic_only=True, physical_grasp_success=None,
        processing_seconds=time.perf_counter()-began, entries=written,
    )
    write_json(args.output/'summary.json', summary)
    print('COMPLETE',json.dumps({k:v for k,v in summary.items() if k!='entries'}),flush=True)


if __name__ == '__main__':
    main()
