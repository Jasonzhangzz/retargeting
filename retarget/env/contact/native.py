"""Original Robokit CPU cost bridge, complete named state transfer and contact diagnostics."""
import mujoco
import numpy as np
from .prediction import distance_query
from .objective import force_penalty_parameters
from retarget.solver.contact_defaults import DEFAULTS

STATE_FIELDS = ('qpos', 'qvel', 'act', 'history', 'qacc_warmstart', 'ctrl', 'qfrc_applied',
                'xfrc_applied', 'eq_active', 'mocap_pos', 'mocap_quat', 'userdata')


def snapshot(data: mujoco.MjData) -> dict:
    return dict(time=float(data.time), **{k: np.array(getattr(data, k), copy=True) for k in STATE_FIELDS if hasattr(data, k)})


def restore(model: mujoco.MjModel, data: mujoco.MjData, state: dict) -> None:
    # Reset *all* persistent state before applying a named snapshot. Never reuse
    # candidate-derived warm starts, controls, histories, or applied forces.
    mujoco.mj_resetData(model, data)
    data.time = float(state.get('time', 0.))
    for k, v in state.items():
        if k == 'time':
            continue
        if not hasattr(data, k) or np.shape(getattr(data, k)) != np.shape(v):
            if np.size(v):
                raise ValueError(f'Incompatible integration state field: {k}')
        else:
            getattr(data, k)[:] = v
    mujoco.mj_forward(model, data)


def contact_records(model: mujoco.MjModel, data: mujoco.MjData) -> list:
    records = []
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 < 0 or c.geom2 < 0:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        dist, _, _, _ = distance_query(model, data, int(c.geom1), int(c.geom2))
        jac1, jac2 = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jac1, None, c.pos, int(model.geom_bodyid[c.geom1]))
        mujoco.mj_jac(model, data, jac2, None, c.pos, int(model.geom_bodyid[c.geom2]))
        velocity = c.frame.reshape(3, 3) @ ((jac2 - jac1) @ data.qvel)
        friction = max(0., float(np.linalg.norm(force[1:3]) - min(c.friction[:2]) * max(force[0], 0.)))
        records.append(dict(geom1=int(c.geom1), geom2=int(c.geom2), position=c.pos.copy(),
            normal=c.frame[:3].copy(), force=force, geometric_distance=dist,
            effective_distance=float(c.dist), normal_speed=float(velocity[0]),
            slip=float(np.linalg.norm(velocity[1:])), friction_violation=friction,
            constraint_address=int(c.efc_address)))
    return records


def step_cost(model: mujoco.MjModel, data: mujoco.MjData, packet: dict, step: int,
              events: list = (), dynamics_cost: bool = True, terminal: bool = False) -> tuple:
    H = len(packet['ref']) - 1
    k = min(step, H)
    from .robokit_cost import source_cost
    records = contact_records(model, data)
    original = source_cost(model, data, packet, k, terminal, records)
    parts = dict(**original, contact=0., timing=0., velocity=0., force=0., slip=0., friction=0.,
                 unexpected=0., joint=0., smooth=0., retarget=0.)
    if dynamics_cost:
        dt = packet['dt']
        force_scale, force_weight = force_penalty_parameters(model)
        for e in events:
            time = k * dt
            if e['start_time'] <= time <= e['start_time'] + e['duration']:
                g1, g2 = e['geom1'], e['geom2']
                p1 = data.geom_xpos[g1] + data.geom_xmat[g1].reshape(3, 3) @ e['link_local_point']
                p2 = data.geom_xpos[g2] + data.geom_xmat[g2].reshape(3, 3) @ e['object_local_point']
                n1 = data.geom_xmat[g1].reshape(3, 3) @ e['link_local_normal']
                n2 = data.geom_xmat[g2].reshape(3, 3) @ e['object_local_normal']
                delta = p1 - p2
                gap = np.linalg.norm(delta) if e['mode'] == 'stick' else abs(delta @ n2)
                parts['contact'] += 100 * max(gap - .002, 0.)**2 + .01 * (1 + n1 @ n2)**2
                if e['mode'] != 'stick':
                    parts['contact'] += .1 * np.sum((delta - (delta @ n2) * n2)**2)
                jp1, jp2 = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jp1, None, p1, int(model.geom_bodyid[g1]))
                mujoco.mj_jac(model, data, jp2, None, p2, int(model.geom_bodyid[g2]))
                relative = (jp1 - jp2) @ data.qvel
                vn = relative @ n2
                tangent = relative - vn * n2
                vt = e['tangential_velocity'] if e['mode'] == 'slide' else np.zeros(3)
                vt = np.asarray(vt) - (np.asarray(vt) @ n2) * n2
                parts['velocity'] += .01 * ((vn - e['normal_velocity'])**2 + np.sum((tangent - vt)**2))
        for c in records:
            fn = c['force'][0]
            parts['force'] += force_weight * max(fn / force_scale - 1., 0.)**2
            parts['friction'] += 1e-4 * c['friction_violation']**2
            expected = c['geom1'] in packet.get('target_geoms', ()) or c['geom2'] in packet.get('target_geoms', ())
            if expected:
                parts['slip'] += .001 * c['slip']**2
            else:
                parts['unexpected'] += 1000 * max(-c['geometric_distance'] - .001, 0.)**2
        for j in range(model.njnt):
            if model.jnt_limited[j] and model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                q = data.qpos[model.jnt_qposadr[j]]
                parts['joint'] += 10 * max(model.jnt_range[j, 0] - q, q - model.jnt_range[j, 1], 0.)**2
        limits = np.maximum(np.max(np.abs(model.actuator_forcerange), axis=1), 1.)
        parts['joint'] += 1e-5 * np.sum((data.actuator_force / limits)**2)
        parts['smooth'] += 1e-8 * np.sum(data.qacc**2)
        if 'qpos_ref' in packet:
            error = np.zeros(model.nv)
            mujoco.mj_differentiatePos(model, error, 1., packet['qpos_ref'][k], data.qpos)
            velocity = data.qvel - packet['qvel_ref'][k]
            parts['retarget'] = (np.dot(DEFAULTS.tracking_root_position_weight, error[:3]**2)
                + DEFAULTS.tracking_root_rotation_weight * np.sum(error[3:6]**2)
                + DEFAULTS.tracking_joint_position_weight * np.sum(error[6:]**2)
                + DEFAULTS.tracking_root_velocity_weight * np.sum(velocity[:6]**2)
                + DEFAULTS.tracking_joint_velocity_weight * np.sum(velocity[6:]**2))
    valid = (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
             and not any(w.number for w in data.warning))
    return float(sum(parts.values())) if valid else np.inf, parts, records


def forecast_contacts(model: mujoco.MjModel, state: dict, controls: np.ndarray,
                      packet: dict, events: list) -> list:
    """Query the planner's commanded joint path before its dynamics validation.

    This is a kinematic proposal, not a promise of actuator tracking. Free joints
    follow the measured twist; dynamics validation below replaces that forecast
    with a controlled rollout, including acceleration and contact response.
    """
    from scipy.spatial.transform import Rotation
    from .prediction import predict_pair
    if not events:
        return []
    initial, scratch = mujoco.MjData(model), mujoco.MjData(model)
    restore(model, initial, state)
    actuator_joints = model.actuator_trnid[:, 0]
    addresses = model.jnt_qposadr[actuator_joints]
    pairs = sorted({(e['geom1'], e['geom2']) for e in events})
    positions, quats = [], []
    for k in range(len(controls) + 1):
        scratch.qpos[:] = initial.qpos
        if k:
            mujoco.mj_integratePos(model, scratch.qpos, initial.qvel, k * packet['dt'])
            scratch.qpos[addresses] = controls[k - 1]
        mujoco.mj_kinematics(model, scratch)
        positions.append(scratch.geom_xpos.copy())
        quats.append(scratch.geom_xmat.copy())
    results = []
    for g1, g2 in pairs:
        gids = [g1, g2]
        pos = np.asarray(positions)[:, gids]
        quat = Rotation.from_matrix(np.asarray(quats)[:, gids].reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]].reshape(-1, 2, 4)
        result = predict_pair(model, initial, g1, g2, np.arange(len(pos)) * packet['dt'], pos, quat,
                              clearance=float(model.geom_margin[gids].sum())).as_dict()
        result['trajectory'] = 'commanded_joint_path_measured_free_twist'
        results.append(result)
    return results
