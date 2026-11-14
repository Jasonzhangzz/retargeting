"""Native Robokit contact-aware trajectory retargeting on the execution hand."""
from copy import deepcopy
from pathlib import Path
import json
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from retarget.env.grasp.hand import hand_tree


@wp.kernel
def project_contacts(mesh: wp.uint64, points: wp.array(dtype=wp.vec3),
                     closest: wp.array(dtype=wp.vec3), distances: wp.array(dtype=float)):
    i = wp.tid()
    query = wp.mesh_query_point_no_sign(mesh, points[i], 10.)
    if query.result:
        p = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        closest[i] = p
        distances[i] = wp.length(points[i] - p)
    else:
        closest[i] = points[i]
        distances[i] = 1.e6


def build_retargeter(device='cuda:0'):
    """Preserve all 26 native coordinates; add only fixed fingertip frames."""
    from robokit.robo import Robot
    from robokit.helpers.hand_retargeting import HandRetargeterOffline, HandSpec, contact_config
    from robokit.helpers.hand_retargeting.presets.wuji_hand_2_2 import offline
    from robokit.helpers.hand_retargeting.presets._common import TARGET_NAMES, TARGET_CHAINS, FIVE_FINGERTIPS
    from robokit.utils.ee_coord_utils import HandCoordinateSpec
    if not str(device).startswith('cuda'):
        raise ValueError('Native retargeting requires CUDA')
    tree = hand_tree('wuji2')
    links = {'wrist': 'right_palm_link'}
    for f, chain in enumerate(TARGET_CHAINS, 1):
        for name, k in zip(chain[:3], (2, 3, 4)):
            links[name] = f'right_finger{f}_link{k}'
        site = tree.find(f".//site[@name='right_{chain[-1].split('_')[0]}_tip']")
        parent = next(b for b in tree.iter('body') if site in list(b))
        name = f'retarget_tip_{f}'
        ET.SubElement(parent, 'body', name=name, pos=site.get('pos', '0 0 0'), quat=site.get('quat', '1 0 0 0'))
        links[chain[-1]] = name
    xml = ET.tostring(tree, encoding='unicode')
    robot = Robot.load(xml, load_meshes=False)
    spec = HandSpec(floating_base=False, target_names=TARGET_NAMES, target_link_names=links,
                    target_chains=TARGET_CHAINS, root_target_name='wrist',
                    contact_target_names=FIVE_FINGERTIPS,
                    root_link_coord_spec=HandCoordinateSpec(palm_forward_axis='y', four_fingers_up_axis='-z'))
    # Native Wuji preset and native contact terms/LM unchanged. The canonical
    # hand represents wrist SE(3) with six named joints rather than a free base.
    config = contact_config(deepcopy(offline))
    config.optimize_base = False
    solver = HandRetargeterOffline(config, robot, spec, device=device)
    return robot, solver, xml


def _palm_frame(points):
    wrist, roots = points[..., 0, :], points[..., [1, 5, 9, 13, 17], :]
    origin = (wrist + roots.sum(-2)) / 6
    x = roots[..., 1, :] - roots[..., 4, :]
    x /= np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1.e-9)
    forward = roots[..., 1:, :].mean(-2) - wrist
    z = np.cross(x, forward)
    z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1.e-9)
    y = np.cross(z, x)
    return origin, np.stack([x, y, z], axis=-1)


def verify_kinematics(model, robot, solver, *, device='cuda:0'):
    """Check named joint/semantic point mappings against the actual MJWarp hand."""
    from robokit.sim import MjWarpSim
    rng = np.random.default_rng(42)
    q = np.repeat(model.qpos0[None], 3, axis=0)
    q[:, :26] += rng.normal(size=(3, 26)) * np.r_[np.full(3, .1), np.full(3, .3), np.full(20, .2)]
    q[:, :26] = np.clip(q[:, :26], *model.actuator_ctrlrange.T)
    order = [robot.spec.actuated_joint_names.index(model.joint(j).name) for j in model.actuator_trnid[:, 0]]
    native = np.zeros((3, 26), np.float32)
    native[:, order] = q[:, :26]
    with wp.ScopedDevice(device):
        state = robot.state(q=wp.array(native, dtype=float, device=device))
        robot.forward_kinematics(state)
        indices = [robot.spec.link_names.index(solver.spec.target_link_names[k]) for k in solver.spec.target_names]
        actual = state.T_world_link.numpy()[:, indices, :3]
        sim = MjWarpSim(model, 3, mujoco.MjData(model), nconmax=1, njmax=1, device=device)
        with wp.ScopedStream(sim.stream):
            sim.data.qpos.assign(q.astype(np.float32))
            sim.kinematics()
            wp.synchronize_stream(sim.stream)
            body, site = sim._d.xpos.numpy(), sim._d.site_xpos.numpy()
    expected = [body[:, model.body('right_palm_link').id]]
    for f, name in enumerate(('thumb', 'index', 'middle', 'ring', 'pinky'), 1):
        expected.extend(body[:, model.body(f'right_finger{f}_link{k}').id] for k in (2, 3, 4))
        expected.append(site[:, model.site(f'right_{name}_tip').id])
    error = float(np.max(np.abs(actual - np.stack(expected, axis=1))))
    if error > 1.e-6:
        raise ValueError(f'Robokit and execution MJWarp FK differ by {error} m')
    return dict(max_coordinate_difference_m=error, poses=3, points_per_pose=21, device=device)


def retarget_motion(motion, mesh_path, mano_path, output, *, device='cuda:0'):
    """No CA trajectory/state enters this function; targets come from HO3D."""
    import trimesh
    from retarget.task.ho3d.source import vertex_finger_owners
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    robot, solver, xml = build_retargeter(device)
    (out/'retarget_hand.xml').write_text(xml)
    points = np.asarray(motion['mano_joints'], np.float32)
    n = len(points)
    object_pose = motion['object_pose']
    rotation = Rotation.from_quat(object_pose[:, 3:], scalar_first=True).as_matrix()
    tip = points[:, [4, 8, 12, 16, 20]]
    tip_object = np.einsum('nji,ncj->nci', rotation, tip - object_pose[:, None, :3]).astype(np.float32)
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    with wp.ScopedDevice(device):
        gpu_mesh = wp.Mesh(points=wp.array(np.asarray(mesh.vertices, np.float32), dtype=wp.vec3, device=device),
                           indices=wp.array(np.asarray(mesh.faces, np.int32).reshape(-1), dtype=int, device=device))
        closest = wp.zeros(n*5, dtype=wp.vec3, device=device)
        distance = wp.zeros(n*5, dtype=float, device=device)
        wp.launch(project_contacts, dim=n*5, inputs=[gpu_mesh.id,
                  wp.array(tip_object.reshape(-1, 3), dtype=wp.vec3, device=device), closest, distance], device=device)
        closest_object = closest.numpy().reshape(n, 5, 3)
        distances = distance.numpy().reshape(n, 5)
        owner = vertex_finger_owners(str(mano_path))
        labelled = np.asarray(motion['vertex_contact_valid'], bool)
        counts = np.stack([motion['vertex_contact'][:, owner == f].sum(-1) for f in range(5)], axis=1)
        mask = (distances < .020) & (counts >= 3) & labelled[:, None]
        targets = object_pose[:, None, :3] + np.einsum('nij,ncj->nci', rotation, closest_object)
        # A closed-form anatomical alignment provides a source-only initial
        # wrist. All trajectory optimization is performed by native Robokit LM.
        neutral = np.clip(np.zeros(robot.spec.num_actuated_joints), *robot.spec.actuated_joint_limits.T).astype(np.float32)
        state = robot.state(q=wp.array(neutral[None], dtype=float, device=device))
        robot.forward_kinematics(state)
        links = [robot.spec.link_names.index(solver.spec.target_link_names[k]) for k in solver.spec.target_names]
        positions = state.T_world_link.numpy()[0, links, :3]
        origin, basis = _palm_frame(positions)
        human_origin, human_basis = _palm_frame(points)
        turn = human_basis @ basis.T
        translation = human_origin - np.einsum('nij,j->ni', turn, origin)
        q = np.repeat(solver._q_init_np[None], n, axis=0)
        canonical = mujoco.MjModel.from_xml_string(ET.tostring(hand_tree('wuji2'), encoding='unicode'))
        order = [robot.spec.actuated_joint_names.index(canonical.joint(j).name) for j in canonical.actuator_trnid[:, 0]]
        q[:, order[:3]] = translation
        q[:, order[3:6]] = Rotation.from_matrix(turn).as_euler('XYZ')
        q = np.clip(q, *robot.spec.actuated_joint_limits.T).astype(np.float32)
        init = robot.state(q=wp.array(q[None], dtype=float, device=device))
        solver.warmup(1, n)
        wp.synchronize_device(device)
        warmup = time.perf_counter() - started
        before = time.perf_counter()
        packed = solver.solve_with_contact_numpy(points, targets.astype(np.float32), mask, init_state=init)
        wp.synchronize_device(device)
        solve_seconds = time.perf_counter() - before
    raw = packed[:, 7:][:, order]
    if raw.shape != (n, 26) or not np.isfinite(raw).all():
        raise RuntimeError('Invalid native Robokit retargeting solution')
    clipped = np.clip(raw, *canonical.actuator_ctrlrange.T)
    np.save(out/'hand_reference.npy', clipped)
    np.savez_compressed(out/'contact_targets.npz', contact_mask=mask, contact_pos_obj=closest_object,
                        distance_m=distances, labelled_vertices=counts)
    report = dict(backend='robokit.HandRetargeterOffline + contact_config(Wuji2 offline preset)',
                  source='Recovered HO3D MANO-21 and source vertex contact labels; no ContactAware input',
                  wrist='Six native wrist joints, fixed optimizer base', frames=n,
                  warmup_seconds=warmup, solve_seconds=solve_seconds, total_seconds=time.perf_counter()-started,
                  solver_iterations=solver.config.solver.max_iter, contact_weight=solver.config.task('contact').weight,
                  contact_margin_m=solver.config.task('contact').contact_margin,
                  contact_detection_distance_m=.020, active_contacts_per_frame=mask.sum(-1).tolist(),
                  clipped_max_rad_or_m=float(np.max(np.abs(raw-clipped))), joint_order=canonical.joint_names if hasattr(canonical,'joint_names') else [canonical.joint(j).name for j in canonical.actuator_trnid[:,0]])
    (out/'timing.json').write_text(json.dumps(report, indent=2))
    return clipped, report
