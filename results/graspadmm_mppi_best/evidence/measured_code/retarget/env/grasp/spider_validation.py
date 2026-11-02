"""Independent full-mesh replay of SPIDER MPPI actuator controls.

This module uses MuJoCo and NumPy only. It never activates DexForge, replaces
the contact solver, changes controls, or inserts a forward call into replay.
Material-point geometry is evaluated on separate MjData instances.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from retarget.common.results import write_json
from .forceaware_mesh_audit import mesh_contact_snapshot
from .forceaware_validation import FINGER_NAMES, finger_body_labels


LOAD_THRESHOLD_N = 1e-6
POSITION_LIMIT_MM = 2.
ROTATION_LIMIT_DEG = 5.


def object_errors(qpos, reference_qpos, object_qpos=26):
    """Reference-relative free-object errors with WXYZ sign invariance."""
    values, reference = np.asarray(qpos, np.float64), np.asarray(reference_qpos, np.float64)
    if values.ndim != 2 or values.shape[1] < object_qpos+7:
        raise ValueError("Object errors require a full state matrix with a free object")
    if reference.shape == (values.shape[1],):
        reference = np.broadcast_to(reference, values.shape)
    if reference.shape != values.shape or not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("Reference must be finite static qpos or one qpos per state")
    actual = values[:, object_qpos:object_qpos+7]
    target = reference[:, object_qpos:object_qpos+7]
    norm = np.linalg.norm(actual[:, 3:], axis=1)
    target_norm = np.linalg.norm(target[:, 3:], axis=1)
    if np.any(norm <= 1e-12) or np.any(target_norm <= 1e-12):
        raise ValueError("Zero object quaternion is not a valid reference or state")
    dot = np.abs(np.sum(actual[:, 3:]*target[:, 3:], axis=1)/(norm*target_norm))
    return dict(position_mm=np.linalg.norm(actual[:, :3]-target[:, :3], axis=1)*1000.,
                rotation_deg=np.rad2deg(2*np.arccos(np.clip(dot, 0., 1.))))


def _stats(values, *, exclude_initial=False):
    values = np.asarray(values, np.float64)
    mean_values = values[1:] if exclude_initial else values
    return dict(mean=float(mean_values.mean()), max=float(values.max()), terminal=float(values[-1]))


def _load_metrics(values):
    values = np.asarray(values, np.float64)
    if values.ndim != 2 or values.shape[1] != 5 or not len(values):
        raise ValueError("Measured finger loads must be [physics_steps,5]")
    finite = bool(np.isfinite(values).all())
    positive = np.isfinite(values) & (values > LOAD_THRESHOLD_N)
    return dict(available=True, finite=finite, finger_names=FINGER_NAMES,
        positive_load_threshold_n=LOAD_THRESHOLD_N,
        finger_normal_force_min_n=values.min(0).tolist(),
        finger_normal_force_mean_n=values.mean(0).tolist(),
        finger_normal_force_max_n=values.max(0).tolist(),
        finger_zero_load_steps=(~positive).sum(0).tolist(),
        finger_positive_load_fraction=positive.mean(0).tolist(),
        finger_continuously_positive=positive.all(0).tolist(),
        all_fingers_continuously_positive=bool(finite and positive.all()))


def _tracking_metrics(errors):
    return dict(position_mm=_stats(errors["position_mm"], exclude_initial=True),
                rotation_deg=_stats(errors["rotation_deg"], exclude_initial=True),
                mean_excludes_initial_state=True)


def _tracking_pass(errors):
    return bool(np.isfinite(errors["position_mm"]).all() and np.isfinite(errors["rotation_deg"]).all()
                and errors["position_mm"].max() <= POSITION_LIMIT_MM
                and errors["rotation_deg"].max() <= ROTATION_LIMIT_DEG)


def _contact_geometry(model, qpos, reference, body_ids, local, object_local, object_body):
    """Evaluate post-state CA gaps without changing a replay solver workspace."""
    body_ids = np.asarray(body_ids)
    local, anchors = np.asarray(local, np.float64), np.asarray(object_local, np.float64)
    if (body_ids.ndim != 1 or not np.issubdtype(body_ids.dtype, np.integer)
            or np.any(body_ids < 1) or np.any(body_ids >= model.nbody)
            or not len(body_ids) or local.shape != (len(body_ids), 3)
            or anchors.shape != local.shape or not np.isfinite(local).all()
            or not np.isfinite(anchors).all()):
        raise ValueError("Contact geometry requires body_ids[C], hand local[C,3] and object local[C,3]")
    actual_data, reference_data = mujoco.MjData(model), mujoco.MjData(model)
    points, targets = [], []
    for actual, target in zip(qpos, reference):
        actual_data.qpos[:] = actual
        reference_data.qpos[:] = target
        mujoco.mj_kinematics(model, actual_data)
        mujoco.mj_kinematics(model, reference_data)
        points.append(actual_data.xpos[body_ids]+np.einsum("nij,nj->ni",
            actual_data.xmat[body_ids].reshape(-1, 3, 3), local))
        targets.append(anchors @ reference_data.xmat[object_body].reshape(3, 3).T
                       + reference_data.xpos[object_body])
    points, targets = np.asarray(points), np.asarray(targets)
    return points, targets, np.linalg.norm(points-targets, axis=-1)*1000.


def audit_replay(model_xml, trajectory_npz, output_dir, *, reference_qpos,
                 contact_body_ids=None, contact_local=None, contact_object_local=None,
                 required_contact_finger_mask=None):
    """Replay every recorded executor step and report real five-finger loads.

    Required trajectory keys are qpos[T+1,33], qvel[T+1,32], ctrl[T,26], and
    time[T+1]. Optional ``finger_normal_force_n[T,5]`` must be actual GPU force
    measurements; absent measurements remain unavailable rather than inferred.
    CA anchors, when supplied, are transformed by the reference object pose,
    matching the original SPIDER world-space contact-target convention.
    """
    model_xml, trajectory_npz = Path(model_xml).resolve(), Path(trajectory_npz).resolve()
    output_dir = Path(output_dir).resolve()
    if trajectory_npz == output_dir/"replay.npz":
        raise ValueError("Replay audit must not overwrite its input trajectory")
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    required = np.ones(5, bool) if required_contact_finger_mask is None else np.asarray(required_contact_finger_mask)
    if required.shape != (5,) or not np.isin(required, (0, 1)).all():
        raise ValueError("Required contact finger mask must contain five binary entries")
    required = required.astype(bool)
    if (model.nq, model.nv, model.nu) != (33, 32, 26):
        raise ValueError("SPIDER WUJI replay requires nq/nv/nu=33/32/26")
    joint = model.joint("obj_joint").id
    if int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("SPIDER object must retain its free joint")
    if np.any(model.actuator_trnid[:, 0] == joint):
        raise ValueError("Physical hold audit cannot use an actuated object")
    object_qpos, object_dof, object_body = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint]), int(model.jnt_bodyid[joint])
    with np.load(trajectory_npz, allow_pickle=False) as raw:
        arrays = {name: np.asarray(raw[name]).copy() for name in ("qpos", "qvel", "ctrl", "time")}
        gpu_force = np.asarray(raw["finger_normal_force_n"], np.float64).copy() if "finger_normal_force_n" in raw else None
        initial_warmstart = np.asarray(raw["initial_qacc_warmstart"], np.float64).copy() if "initial_qacc_warmstart" in raw else None
    steps = len(arrays["ctrl"])
    shapes = dict(qpos=(steps+1, 33), qvel=(steps+1, 32), ctrl=(steps, 26), time=(steps+1,))
    if steps < 1 or any(arrays[name].shape != shape for name, shape in shapes.items()):
        raise ValueError("Trajectory must cover every executor control and full initial/final state")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Replay input states, controls and times must be finite")
    differences = np.diff(arrays["time"].astype(np.float64))
    dt = float(differences.mean())
    if dt <= 0 or np.any(differences <= 0) or not np.allclose(differences, dt, rtol=1e-4, atol=1e-8):
        raise ValueError("Replay time must be increasing on a uniform executor grid")
    model_dt = float(model.opt.timestep)
    if np.allclose(differences, model_dt, rtol=1e-4, atol=1e-8):
        dt = model_dt
    model.opt.timestep = dt
    reference = np.asarray(reference_qpos, np.float64)
    if reference.shape == (33,):
        reference = np.broadcast_to(reference, (steps+1, 33))
    gpu_errors = object_errors(arrays["qpos"], reference, object_qpos)
    gpu_load = _load_metrics(gpu_force) if gpu_force is not None else dict(available=False,
        reason="No measured GPU finger_normal_force_n array was recorded", all_fingers_continuously_positive=None)
    if gpu_force is not None and gpu_force.shape != (steps, 5):
        raise ValueError("GPU pressure measurements must cover exactly every executor step")
    geometry_inputs = (contact_body_ids, contact_local, contact_object_local)
    if any(value is not None for value in geometry_inputs) and not all(value is not None for value in geometry_inputs):
        raise ValueError("True CA gaps need body IDs, hand-local points and object-local anchors together")
    data = mujoco.MjData(model)
    data.qpos[:] = arrays["qpos"][0]
    data.qvel[:] = arrays["qvel"][0]
    data.time = float(arrays["time"][0])
    if initial_warmstart is not None:
        if initial_warmstart.shape != (32,) or not np.isfinite(initial_warmstart).all():
            raise ValueError("initial_qacc_warmstart must contain all 32 finite DOFs")
        data.qacc_warmstart[:] = initial_warmstart
    labels = finger_body_labels(model)
    qpos, qvel, times = [data.qpos.copy()], [data.qvel.copy()], [float(data.time)]
    normal, finger_force, object_force, constraint = [], [], [], []
    flat_names = ("contact_geom_ids", "contact_distance_m", "contact_position_world_m", "contact_frame",
                  "contact_wrench_local", "contact_force_on_geom2_world_n", "contact_object_force_sign", "contact_finger_id")
    flat = {name: [] for name in flat_names}
    offsets = [0]
    for command in arrays["ctrl"]:
        data.ctrl[:] = command
        mujoco.mj_step(model, data)
        snapshot = mesh_contact_snapshot(model, data, object_body=object_body, body_finger=labels)
        for name in flat_names:
            flat[name].append(snapshot[name])
        offsets.append(offsets[-1]+len(snapshot["contact_geom_ids"]))
        normal.append(snapshot["finger_normal_force_n"])
        finger_force.append(snapshot["finger_force_on_object_world_n"])
        object_force.append(snapshot["object_contact_force_world_n"])
        constraint.append(data.qfrc_constraint.copy())
        qpos.append(data.qpos.copy()); qvel.append(data.qvel.copy()); times.append(float(data.time))
    qpos, qvel, normal = np.asarray(qpos), np.asarray(qvel), np.asarray(normal)
    cpu_state_finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())
    if cpu_state_finite:
        cpu_errors = object_errors(qpos, reference, object_qpos)
        model_difference = object_errors(qpos, arrays["qpos"], object_qpos)
    else:
        # Preserve the failing raw trace and explicit failed acceptance rather
        # than replacing invalid states or throwing away the replay evidence.
        cpu_errors = dict(position_mm=np.full(steps+1, np.nan), rotation_deg=np.full(steps+1, np.nan))
        model_difference = {name: value.copy() for name, value in cpu_errors.items()}
    cpu_load = _load_metrics(normal)
    warnings = [int(item.number) for item in data.warning]
    warnings_zero = not any(warnings)
    trace = {f"cpu_{name}": np.concatenate(values, axis=0) for name, values in flat.items()}
    trace.update(gpu_qpos=arrays["qpos"], gpu_qvel=arrays["qvel"], cpu_qpos=qpos, cpu_qvel=qvel,
        ctrl=arrays["ctrl"], time_gpu=arrays["time"], time_cpu=np.asarray(times), reference_qpos=reference,
        cpu_contact_offsets=np.asarray(offsets, np.int64), cpu_finger_normal_force_n=normal,
        cpu_finger_force_on_object_world_n=np.asarray(finger_force),
        cpu_object_contact_force_world_n=np.asarray(object_force), cpu_generalized_constraint_force=np.asarray(constraint),
        gpu_object_position_error_mm=gpu_errors["position_mm"], gpu_object_rotation_error_deg=gpu_errors["rotation_deg"],
        cpu_object_position_error_mm=cpu_errors["position_mm"], cpu_object_rotation_error_deg=cpu_errors["rotation_deg"])
    if gpu_force is not None:
        trace["gpu_finger_normal_force_n"] = gpu_force
    contact_report = dict(available=False)
    if contact_body_ids is not None and cpu_state_finite:
        gpu_points, targets, gpu_gaps = _contact_geometry(model, arrays["qpos"], reference,
            contact_body_ids, contact_local, contact_object_local, object_body)
        cpu_points, _, cpu_gaps = _contact_geometry(model, qpos, reference,
            contact_body_ids, contact_local, contact_object_local, object_body)
        trace.update(gpu_contact_points_world_m=gpu_points, cpu_contact_points_world_m=cpu_points,
                     contact_target_world_m=targets, gpu_contact_target_gap_mm=gpu_gaps, cpu_contact_target_gap_mm=cpu_gaps)
        contact_report = dict(available=True, target_frame="reference object pose times accepted CA object-local anchor",
            measurement="post-state material FK on separate MjData; no mutation of replay workspace",
            gpu_mean_gap_mm=float(gpu_gaps[1:].mean()), gpu_max_gap_mm=float(gpu_gaps.max()),
            cpu_mean_gap_mm=float(cpu_gaps[1:].mean()), cpu_max_gap_mm=float(cpu_gaps.max()),
            gpu_per_contact_max_gap_mm=gpu_gaps.max(0).tolist(), cpu_per_contact_max_gap_mm=cpu_gaps.max(0).tolist())
    cpu_force_finite = bool(all(np.isfinite(trace[name]).all() for name in
        ("cpu_contact_wrench_local", "cpu_finger_normal_force_n", "cpu_finger_force_on_object_world_n",
         "cpu_object_contact_force_world_n", "cpu_generalized_constraint_force")))
    finite = bool(cpu_state_finite and cpu_force_finite and (gpu_force is None or gpu_load["finite"]))
    gpu_hold, cpu_hold = _tracking_pass(gpu_errors), bool(_tracking_pass(cpu_errors) and warnings_zero)
    gpu_five = gpu_load["all_fingers_continuously_positive"]
    cpu_five = bool(cpu_force_finite and cpu_load["all_fingers_continuously_positive"] and warnings_zero)
    force_error = np.asarray(object_force)-np.asarray(constraint)[:, object_dof:object_dof+3]
    gpu_required = None if gpu_force is None else bool((gpu_force[:, required] > LOAD_THRESHOLD_N).all())
    cpu_required = bool((normal[:, required] > LOAD_THRESHOLD_N).all())
    report = dict(method="spider_mppi", finite=finite, hardware_tested=False,
        input_model=str(model_xml), input_trajectory=str(trajectory_npz),
        model_sha256=hashlib.sha256(model_xml.read_bytes()).hexdigest(),
        trajectory_sha256=hashlib.sha256(trajectory_npz.read_bytes()).hexdigest(),
        physics_steps=steps, dt=dt, duration_seconds=steps*dt,
        original_model_timestep=model_dt, timestep_matched_to_recorded_executor_grid=dt != model_dt,
        integrator=mujoco.mjtIntegrator(model.opt.integrator).name, object_actuated=False,
        initial_qpos_qvel_unchanged=True, controls_changed=False, extra_forward_calls=False,
        initial_warmstart_source="trajectory initial_qacc_warmstart" if initial_warmstart is not None else "MuJoCo default zero",
        force_stage="actual mj_contactForce immediately after mj_step; solver contacts precede logged post-step qpos",
        force_source="CPU: mujoco.mj_contactForce; GPU: optional caller-recorded actual finger_normal_force_n",
        gpu=dict(**_tracking_metrics(gpu_errors), **gpu_load),
        cpu=dict(**_tracking_metrics(cpu_errors), **cpu_load),
        gpu_cpu_object_difference=_tracking_metrics(model_difference),
        gpu_cpu_qpos_max_abs_difference=float(np.max(np.abs(qpos-arrays["qpos"]))),
        gpu_cpu_qvel_max_abs_difference=float(np.max(np.abs(qvel-arrays["qvel"]))),
        gpu_cpu_comparison_scope="independent solvers; differences reported, no exact-trajectory requirement",
        cpu_object_contact_force_constraint_max_error_n=float(np.max(np.abs(force_error))),
        mujoco_warning_counts=warnings, mujoco_warnings_zero=warnings_zero,
        acceptance=dict(max_object_position_error_mm=POSITION_LIMIT_MM, max_object_rotation_error_deg=ROTATION_LIMIT_DEG,
                        minimum_each_finger_normal_force_n=LOAD_THRESHOLD_N, every_executor_step_required=True,
                        required_contact_finger_mask=required.tolist(),
                        contact_scope="CA-designated fingers; all five when no explicit mask supplied"),
        native_hold_success=gpu_hold, independent_mujoco_hold_success=cpu_hold,
        five_finger_loaded_contact_retained=gpu_five,
        independent_mujoco_five_finger_contact_retained=cpu_five,
        required_finger_contact_retained=gpu_required,
        independent_mujoco_required_finger_contact_retained=cpu_required,
        task_success=bool(finite and gpu_hold and cpu_hold and gpu_required is True and cpu_required),
        object_drift_max_mm=float(gpu_errors["position_mm"].max()),
        object_rotation_max_deg=float(gpu_errors["rotation_deg"].max()), contact_geometry=contact_report)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir/"replay.npz", **trace)
    write_json(output_dir/"replay.json", report)
    return report
