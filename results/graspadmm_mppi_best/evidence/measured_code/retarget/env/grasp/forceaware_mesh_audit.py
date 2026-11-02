"""Actual MuJoCo mesh-contact forces from an unchanged official FA replay.

These observations apply only to the independent mesh replay, not native GS
predictions or hardware. Contact forces are read after each original mj_step;
there is no extra forward call or change to the physics/solver/controller.
"""

import json
from pathlib import Path

import mujoco
import numpy as np

from .forceaware_validation import FINGER_NAMES, finger_body_labels, object_motion


class MeshReplayMismatch(RuntimeError):
    """The instrumented replay did not reproduce its saved official trajectory."""


def control_physics_indices(action_count, substeps):
    if action_count < 1 or substeps < 1 or int(action_count) != action_count or int(substeps) != substeps:
        raise ValueError("Mesh replay requires positive integer action and substep counts")
    return np.repeat(np.arange(action_count, dtype=np.int32), int(substeps))


def _object_subtree(model, body):
    selected = np.zeros(model.nbody, bool)
    selected[body] = True
    for child in range(body + 1, model.nbody):
        selected[child] = selected[int(model.body_parentid[child])]
    return selected


def mesh_contact_snapshot(model, data, *, object_body, body_finger=None):
    """Read real contact wrenches, including rigid pad/sensor descendants."""
    body_finger = finger_body_labels(model) if body_finger is None else np.asarray(body_finger)
    object_bodies = _object_subtree(model, object_body)
    count = int(data.ncon)
    geom = np.empty((count, 2), np.int32)
    distance, position, frame = np.empty(count), np.empty((count, 3)), np.empty((count, 3, 3))
    wrench, force_on_geom2 = np.empty((count, 6)), np.empty((count, 3))
    finger_ids = np.full(count, -1, np.int32)
    object_sign = np.zeros(count)
    normal = np.zeros(5)
    finger_world, object_world = np.zeros((5, 3)), np.zeros(3)
    for contact_id in range(count):
        contact = data.contact[contact_id]
        geom[contact_id] = contact.geom
        if np.any(geom[contact_id] < 0):
            raise ValueError("Mesh contact audit supports rigid geom contacts, not flex contacts")
        distance[contact_id], position[contact_id] = contact.dist, contact.pos
        frame[contact_id] = contact.frame.reshape(3, 3)
        mujoco.mj_contactForce(model, data, contact_id, wrench[contact_id])
        force_on_geom2[contact_id] = frame[contact_id].T @ wrench[contact_id, :3]
        first, second = model.geom_bodyid[geom[contact_id]]
        if object_bodies[first] and not object_bodies[second]:
            hand, sign = second, -1.
        elif object_bodies[second] and not object_bodies[first]:
            hand, sign = first, 1.
        else:
            continue
        object_sign[contact_id] = sign
        object_force = sign * force_on_geom2[contact_id]
        object_world += object_force
        finger = int(body_finger[hand])
        finger_ids[contact_id] = finger
        if finger >= 0:
            normal[finger] += wrench[contact_id, 0]
            finger_world[finger] += object_force
    return dict(contact_geom_ids=geom, contact_distance_m=distance, contact_position_world_m=position,
        contact_frame=frame, contact_wrench_local=wrench, contact_force_on_geom2_world_n=force_on_geom2,
        contact_object_force_sign=object_sign, contact_finger_id=finger_ids,
        finger_normal_force_n=normal, finger_force_on_object_world_n=finger_world,
        object_contact_force_world_n=object_world)


def compare_saved_replay(sampled, saved):
    """Exact official float32 action-grid comparison; never hide a discrepancy."""
    comparison = {}
    for name in ("qpos", "qvel", "time", "ctrl"):
        observed, reference = np.asarray(getattr(sampled, name)), np.asarray(saved[name])
        shape_equal = observed.shape == reference.shape
        comparison[name] = dict(audit_shape=list(observed.shape), saved_shape=list(reference.shape),
            exact_equal=bool(shape_equal and np.array_equal(observed, reference)),
            max_abs_difference=float(np.max(np.abs(observed.astype(np.float64)-reference.astype(np.float64)), initial=0.)) if shape_equal else None)
    comparison["all_exact_equal"] = all(comparison[name]["exact_equal"] for name in ("qpos", "qvel", "time", "ctrl"))
    return comparison


def audit_mesh_controls(cfg, rollout, output_dir):
    """Return/save a force report, raising after saving if official replay differs.

    ``output_dir`` is a separate audit directory. The saved official replay at
    rollout.parent/replay/mujoco_cpu is required. Its sim_dt is used verbatim;
    if replay.json is absent, the original default 2ms step is used.
    """
    from retarget.dynamics.dexforge_upstream import activate_upstream, require_environment, verify_upstream
    require_environment(); activate_upstream()
    from forceaware.replay import (DEFAULT_REPLAY_SIM_DT, action_to_ctrl, load_rollout,
        object_qadr, replay_errors, replay_from_logs, replay_substeps, seed_mujoco_data, stats, validate_model)
    from tools.render import scene_mesh_xml_for
    from retarget.common.results import write_json

    rollout = Path(rollout).expanduser().resolve()
    official_dir = rollout.parent / "replay/mujoco_cpu"
    saved_path, replay_report_path = official_dir / "replay.npz", official_dir / "replay.json"
    if not saved_path.is_file():
        raise ValueError(f"Saved official mesh replay is required: {saved_path}")
    replay_report = json.loads(replay_report_path.read_text()) if replay_report_path.is_file() else {}
    sim_dt = float(replay_report.get("sim_dt", DEFAULT_REPLAY_SIM_DT))
    if not np.isfinite(sim_dt) or sim_dt <= 0:
        raise ValueError("Saved official replay sim_dt must be finite and positive")
    scene = scene_mesh_xml_for(Path(cfg.sequence.xml).resolve())
    if "scene_xml" in replay_report and Path(replay_report["scene_xml"]).resolve() != scene:
        raise ValueError("Official mesh replay scene does not match this config")
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = sim_dt
    source = load_rollout(rollout, scene, model)
    validate_model(model, source)
    substeps = replay_substeps(source.action_dt, sim_dt)
    control_index = control_physics_indices(len(source.ctrl), substeps)
    object_qpos = object_qadr(model)
    object_joint = model.joint("obj_joint").id
    object_body = int(model.jnt_bodyid[object_joint])
    labels = finger_body_labels(model)
    data = seed_mujoco_data(model, source)
    sampled_q, sampled_v, sampled_ctrl = [data.qpos.copy()], [data.qvel.copy()], []
    physics_q, physics_v, physics_time = [data.qpos.copy()], [data.qvel.copy()], [float(data.time)]
    force_log, finger_world_log, object_world_log = [], [], []
    flat_names = ("contact_geom_ids", "contact_distance_m", "contact_position_world_m", "contact_frame",
        "contact_wrench_local", "contact_force_on_geom2_world_n", "contact_object_force_sign", "contact_finger_id")
    flat = {name: [] for name in flat_names}
    offsets = [0]
    physics_ctrl = []
    for action in source.ctrl:
        ctrl = action_to_ctrl(model, action)
        for _ in range(substeps):
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            snapshot = mesh_contact_snapshot(model, data, object_body=object_body, body_finger=labels)
            for name in flat_names:
                flat[name].append(snapshot[name])
            offsets.append(offsets[-1]+len(snapshot["contact_geom_ids"]))
            force_log.append(snapshot["finger_normal_force_n"])
            finger_world_log.append(snapshot["finger_force_on_object_world_n"])
            object_world_log.append(snapshot["object_contact_force_world_n"])
            physics_q.append(data.qpos.copy()); physics_v.append(data.qvel.copy())
            physics_time.append(float(data.time)); physics_ctrl.append(ctrl.copy())
        sampled_q.append(data.qpos.copy()); sampled_v.append(data.qvel.copy()); sampled_ctrl.append(ctrl.copy())
    sampled = replay_from_logs(sampled_q, sampled_v, sampled_ctrl, source.action_dt)
    with np.load(saved_path, allow_pickle=False) as saved:
        comparison = compare_saved_replay(sampled, saved)
    arrays = {name: np.concatenate(value, axis=0) for name, value in flat.items()}
    arrays.update(contact_offsets=np.asarray(offsets, np.int64), control_index=control_index,
        physics_qpos=np.asarray(physics_q), physics_qvel=np.asarray(physics_v), physics_time=np.asarray(physics_time),
        physics_control=np.asarray(physics_ctrl), finger_normal_force_n=np.asarray(force_log),
        finger_force_on_object_world_n=np.asarray(finger_world_log), object_contact_force_world_n=np.asarray(object_world_log),
        sampled_qpos=sampled.qpos, sampled_qvel=sampled.qvel, sampled_time=sampled.time, sampled_control=sampled.ctrl,
        body_finger_id=labels, geom_body_id=model.geom_bodyid.copy())
    arrays.update(object_motion(arrays["physics_qpos"], object_qpos))
    finite = all(np.isfinite(arrays[name]).all() for name in ("physics_qpos", "physics_qvel", "contact_wrench_local", "finger_normal_force_n"))
    normal = arrays["finger_normal_force_n"]
    threshold = 1e-6
    zero = normal <= threshold
    errors = replay_errors(source, sampled.qpos, object_qpos)
    report = dict(scope="independent MuJoCo mesh replay only; not native GS predictions or hardware",
        hardware_tested=False, force_source="mujoco.mj_contactForce called after each original mj_step",
        physical_parameters_changed=False, solver_changed=False, controls_changed=False,
        initial_state_changed=False, extra_forward_calls=False, upstream_source_modified=False,
        upstream_commit=verify_upstream()["commit"], scene_xml=str(scene), input_rollout=str(rollout),
        official_replay_npz=str(saved_path), official_replay_json=str(replay_report_path),
        sim_dt=sim_dt, action_dt=source.action_dt, substeps_per_control=substeps,
        physics_steps=len(control_index), duration_seconds=len(control_index)*sim_dt,
        integrator=mujoco.mjtIntegrator(model.opt.integrator).name, finite=bool(finite),
        official_replay_comparison=comparison, trajectory_matches_official_replay=comparison["all_exact_equal"],
        force_conclusions_apply_to_saved_replay=bool(finite and comparison["all_exact_equal"]),
        contact_force_convention="wrench contact-frame X is signed normal force; frame.T rotates geom1-to-geom2 force into world; object-pair orientation determines sign",
        force_stage="solver contact state immediately after mj_step, without recomputing contacts; logged beside post-step qpos/qvel",
        finger_names=FINGER_NAMES, positive_load_threshold_n=threshold,
        finger_normal_force_min_n=normal.min(0).tolist(), finger_normal_force_mean_n=normal.mean(0).tolist(),
        finger_normal_force_max_n=normal.max(0).tolist(), finger_normal_force_terminal_n=normal[-1].tolist(),
        finger_zero_load_steps=zero.sum(0).tolist(), finger_positive_load_fraction=(~zero).mean(0).tolist(),
        finger_continuously_positive=(~zero).all(0).tolist(), all_fingers_continuously_positive=bool((~zero).all()),
        first_zero_load_step=[None if not np.any(zero[:,k]) else int(np.flatnonzero(zero[:,k])[0]+1) for k in range(5)],
        object_displacement_max_mm=float(arrays["object_displacement_mm"].max()),
        object_rotation_max_rad=float(arrays["object_rotation_rad"].max()),
        official_sample_grid_errors={name: stats(values) for name, values in errors.items()},
        object_contact_force_world_mean_n=arrays["object_contact_force_world_n"].mean(0).tolist(),
        mujoco_warning_counts=[int(item.number) for item in data.warning])
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir == official_dir:
        raise ValueError("Mesh audit output must not overwrite the official replay directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "mesh_contact_audit.npz", **arrays)
    write_json(output_dir / "mesh_contact_audit.json", report)
    if not comparison["all_exact_equal"]:
        raise MeshReplayMismatch(f"Instrumented mesh replay differs from saved official replay; inspect {output_dir / 'mesh_contact_audit.json'}")
    if not finite:
        raise RuntimeError(f"Nonfinite independent mesh replay audit; inspect {output_dir / 'mesh_contact_audit.json'}")
    return report
