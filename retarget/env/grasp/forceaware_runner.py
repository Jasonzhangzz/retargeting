"""Run the byte-verified original DexForge ForceAware workflow on WUJI2.

The optimizer and objective stay upstream. Assets, topology, accepted stationary
ContactAware inputs and explicitly selected native integration compatibility
are adapted. Each native job uses a fresh process.
"""

from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from retarget.common.results import write_json


def _options(config):
    options = dict(config.environment.get("dexforge", {}))
    allowed = {"duration_seconds", "sphere_radius_mm", "sphere_fit", "gradient_check", "gradient_horizon",
               "gradient_directions", "case_indices", "virtual_wrist_rotation_pd", "finger_servo_scale", "servo_integration", "control_preload", "fixed_initial_state", "contactaware_refinement"}
    if set(options) - allowed:
        raise ValueError(f"Unknown grasp ForceAware options: {sorted(set(options)-allowed)}")
    for key, default in (("duration_seconds", .4), ("sphere_radius_mm", 1.)):
        value = options.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"environment.dexforge.{key} must be finite and positive")
    if not isinstance(options.setdefault("gradient_check", True), bool):
        raise ValueError("gradient_check must be boolean")
    if not isinstance(options.setdefault("fixed_initial_state", False), bool):
        raise ValueError("fixed_initial_state must be boolean")
    preload = options.setdefault("control_preload", False)
    if isinstance(preload, dict):
        if set(preload) - {"minimum_per_contact", "minimum_total", "iterations"}:
            raise ValueError("Unknown static control preload options")
        preload = {"minimum_per_contact": .2, "minimum_total": 6., "iterations": 10000, **preload}
        pressure = preload["minimum_per_contact"]
        if isinstance(pressure, dict):
            if set(pressure) != {"thumb", "index", "middle", "ring", "pinky"}:
                raise ValueError("control_preload.minimum_per_contact must name all five fingers")
            values = list(pressure.values())
        else:
            values = [pressure]
        values.append(preload["minimum_total"])
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Static preload pressure bounds must be finite and positive")
        if isinstance(preload["iterations"], bool) or not isinstance(preload["iterations"], int) or preload["iterations"] < 1:
            raise ValueError("control_preload.iterations must be a positive integer")
        options["control_preload"] = preload
    elif not isinstance(preload, bool):
        raise ValueError("control_preload must be boolean or a static preload options mapping")
    refinement = options.setdefault("contactaware_refinement", False)
    if refinement is True:
        refinement = {}
    if isinstance(refinement, dict):
        if set(refinement)-{"contact_tolerance_mm", "penetration_tolerance_mm", "active_only", "normal_cosine"}:
            raise ValueError("Unknown ContactAware refinement options")
        refinement = {"contact_tolerance_mm": .2, "penetration_tolerance_mm": .5, "active_only": True, **refinement}
        for key in ("contact_tolerance_mm", "penetration_tolerance_mm"):
            value = refinement[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"contactaware_refinement.{key} must be finite and positive")
        if not isinstance(refinement["active_only"], bool):
            raise ValueError("contactaware_refinement.active_only must be boolean")
        cosine = refinement.get("normal_cosine")
        if cosine is not None and (isinstance(cosine, bool) or not isinstance(cosine, (int, float))
                or not np.isfinite(cosine) or not 0 <= cosine < 1):
            raise ValueError("contactaware_refinement.normal_cosine must be in [0, 1)")
        options["contactaware_refinement"] = refinement
    elif refinement is not False:
        raise ValueError("contactaware_refinement must be boolean or an explicit options mapping")
    if options.setdefault("servo_integration", "upstream_euler") not in ("upstream_euler", "implicit_actuator"):
        raise ValueError("servo_integration must be upstream_euler or implicit_actuator")
    if options.setdefault("sphere_fit", "tangent") not in ("tangent", "convex_inside"):
        raise ValueError("sphere_fit must be tangent or convex_inside")
    for key, default in (("gradient_horizon", 10), ("gradient_directions", 1)):
        value = options.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    indices = options.get("case_indices")
    if indices is not None and (not isinstance(indices, list) or not indices or
            any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices) or len(set(indices)) != len(indices)):
        raise ValueError("case_indices must contain distinct nonnegative integers")
    pd = options.get("virtual_wrist_rotation_pd")
    if pd is not None and (not isinstance(pd, dict) or set(pd) != {"kp", "kv"} or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0
            for value in pd.values())):
        raise ValueError("virtual_wrist_rotation_pd requires explicit positive finite kp and kv")
    scale = options.get("finger_servo_scale")
    if scale is not None and (not isinstance(scale, dict) or set(scale) != {"kp_scale", "kv_scale"} or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0
            for value in scale.values())):
        raise ValueError("finger_servo_scale requires explicit positive finite kp_scale and kv_scale")
    if pd is not None and scale is not None:
        raise ValueError("Choose one explicit controller calibration profile per experiment")
    return options


def _solver_overrides(settings):
    """Allow explicit numerical budgets, while keeping the original FA costs."""
    from retarget.solver.dexforge import DexForgeSettings
    defaults, values = asdict(DexForgeSettings()), asdict(settings)
    optimizer = {"horizon", "n_iter", "lr_max", "lr_min", "adam_beta1", "adam_beta2", "adam_epsilon",
                 "grad_clip", "warm_shift_scale", "multistart_count", "multistart_seed",
                 "multistart_init_std", "multistart_ctrl_std", "multistart_init_raw_limit"}
    simulator = {"mpc_dt", "action_dt", "knot_dt", "exec_dt"}
    unsupported = [key for key in values if key not in optimizer | simulator | {"contact_topk"}
                   and values[key] != defaults[key]]
    if unsupported:
        raise ValueError(f"Grasp uses the original ForceAware costs/bounds; unsupported overrides: {unsupported}")
    return ({key: values[key] for key in optimizer if values[key] != defaults[key]},
            {key: values[key] for key in simulator if values[key] != defaults[key]})


def _experiments(data):
    data = Path(data)
    if (data / "card_input.npz").is_file():
        return [data]
    candidates = sorted(path.parent for path in data.glob("**/card_input.npz")
        if (path.parent / "contactaware_reference.json").is_file())
    # A prepared FA case can retain a nested refined-CA input for provenance.
    # Its outer card is the consumable experiment, rather than a second sample.
    return [path for path in candidates if not any((parent / "card_input.npz").is_file()
            for parent in path.parents if parent.is_relative_to(data))]


def validate(config):
    if config.task.get("type") != "grasp" or config.task.get("name") != "primitive" or config.solver.type != "dexforge":
        raise ValueError("Grasp ForceAware requires task grasp/primitive and solver dexforge")
    if config.environment.get("dynamics") != "dexforge" or not str(config.device).startswith("cuda"):
        raise ValueError("Original ForceAware requires dexforge dynamics and CUDA")
    if config.task.get("options", {}) != {"hand_model": "wuji2"}:
        raise ValueError("Grasp ForceAware requires task.options: {hand_model: wuji2}")
    options = _options(config)
    _solver_overrides(config.solver.dexforge)
    from retarget.dynamics.dexforge_upstream import require_environment, verify_upstream
    from .forceaware_reference import _read_ca
    require_environment(); verify_upstream()
    sources = _experiments(config.task["data"])
    if not sources:
        raise ValueError("No accepted ContactAware experiments found")
    indices = options.get("case_indices", list(range(len(sources))))
    if max(indices) >= len(sources):
        raise ValueError(f"case_indices exceeds {len(sources)} available CA experiments")
    selected = [sources[i] for i in indices]
    for source in selected:
        _read_ca(source)
        if not (source / "sequence/scene/wuji_right/primitive_sdf.xml").is_file():
            raise ValueError(f"Missing CA source scene: {source}")
    return selected


def _job(artifacts, config, action, destination, options):
    command = [sys.executable, "-m", __name__, "--action", action,
               "--root", str(artifacts.root), "--config", str(artifacts.config_path),
               "--output", str(destination), "--device", str(config.device),
               "--gradient-horizon", str(options["gradient_horizon"]),
               "--gradient-directions", str(options["gradient_directions"]),
               "--servo-integration", options["servo_integration"],
               "--baseline-duration", str(options["duration_seconds"] if options["control_preload"] else min(.1, options["duration_seconds"]))]
    if options["control_preload"]:
        command.append("--control-preload")
    if options["fixed_initial_state"]:
        command.append("--fixed-initial-state")
    if not options["gradient_check"]:
        command.append("--skip-gradients")
    # Match the original run_example.py worker environment. Small CPU matrix
    # operations should not contend with CUDA submission through BLAS threads.
    environment = dict(os.environ, MUJOCO_GL="egl", FORCEAWARE_DEVICE=str(config.device),
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (artifacts.root / f"{action}.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    if result.returncode:
        raise RuntimeError(f"Original ForceAware {action} exited {result.returncode}; see {artifacts.root / (action+'.log')}")


def _rollout_status(arrays, cfg):
    """Account for every state, control and target on the official solve grid."""
    expected = sum(cfg.simulator.time_grid.execution_schedule(cfg.sequence.hold_seconds))
    shapes = {"qpos_traj": (expected + 1, 33), "qvel_traj": (expected + 1, 32),
              "ctrl_traj": (expected, 26), "drift": (expected,),
              "rot_err_deg": (expected,), "object_reference_qpos": (expected + 1, 7)}
    errors = {name: dict(actual=list(arrays[name].shape), expected=list(shape))
              for name, shape in shapes.items() if arrays[name].shape != shape}
    return dict(finite=bool(all(np.isfinite(value).all() for value in arrays.values())),
                complete=not errors, expected_physics_steps=expected, shape_errors=errors)


def _penetration_acceptance(cfg, maximum_mm):
    """Use the same soft/deep depths as the actual FA contact objective."""
    contact = cfg.contact
    soft_mm, deep_mm = float(contact.soft_penetration_limit) * 1000., float(contact.penetration_limit) * 1000.
    target_mm = float(contact.target_phi) * 1000.
    if (not np.isfinite([soft_mm, deep_mm, target_mm]).all() or soft_mm < 0 or deep_mm < soft_mm
            or not np.isclose(soft_mm, -target_mm, rtol=0, atol=1e-9)):
        raise ValueError("ForceAware penetration acceptance requires consistent configured soft/deep limits")
    finite = bool(np.isfinite(maximum_mm) and maximum_mm >= 0.)
    return dict(target_phi_mm=target_mm, soft_limit_mm=soft_mm, deep_limit_mm=deep_mm,
                maximum_mm=maximum_mm, finite=finite,
                within_soft_limit=bool(finite and maximum_mm <= soft_mm),
                within_deep_limit=bool(finite and maximum_mm <= deep_mm),
                soft_excess_max_mm=max(maximum_mm-soft_mm, 0.),
                deep_excess_max_mm=max(maximum_mm-deep_mm, 0.))


def _paper_tracking_criterion(position_m, rotation_deg):
    """Report Sec.4.2.1's mean-error criterion without changing hold acceptance."""
    position, rotation = np.asarray(position_m, np.float64), np.asarray(rotation_deg, np.float64)
    if position.ndim != 1 or rotation.shape != position.shape:
        raise ValueError("Paper tracking errors must be matching per-step vectors")
    finite = bool(len(position) and np.isfinite(position).all() and np.isfinite(rotation).all())
    mean_position = float(position.mean()) if finite else None
    mean_rotation = float(np.deg2rad(rotation).mean()) if finite else None
    return dict(success=bool(finite and mean_position < .01 and mean_rotation < .1),
        samples=len(position), finite=finite, mean_position_m=mean_position,
        mean_rotation_rad=mean_rotation, position_threshold_m=.01, rotation_threshold_rad=.1,
        source="https://arxiv.org/html/2610.06331v1#S4.SS2.SSS1",
        scope="tracking criterion only on this local task; not a reproduction of the paper benchmark",
        full_paper_benchmark_reproduced=False, replaces_strict_hold_acceptance=False)


def collect(artifacts, cfg, validation):
    from scipy.spatial.transform import Rotation
    native = cfg.sequence.seq_dir / cfg.sequence.output_dir_name
    with np.load(native / "rollout.npz", allow_pickle=False) as raw:
        arrays = {name: raw[name] for name in ("qpos_traj", "qvel_traj", "ctrl_traj", "drift", "rot_err_deg", "object_reference_qpos")}
        window_seconds = np.asarray(raw["mpc_step_sec"], np.float64) if "mpc_step_sec" in raw else None
    qpos, qvel, ctrl = (arrays[name] for name in ("qpos_traj", "qvel_traj", "ctrl_traj"))
    dt = cfg.simulator.time_grid.mpc_dt
    status = _rollout_status(arrays, cfg)
    finite, complete = status["finite"], status["complete"]
    if qpos.ndim != 2 or qpos.shape[1] != 33 or not len(qpos):
        raise ValueError("Official WUJI2 rollout must contain a full initial qpos[33]")
    np.savez_compressed(artifacts.root / "trajectory.npz", qpos=qpos, qvel=qvel, ctrl=ctrl,
                        time=np.arange(len(qpos))*dt, object_reference_qpos=arrays["object_reference_qpos"])
    metrics = json.loads((native / "metrics.json").read_text())
    replay_path = native / "replay/mujoco_cpu/replay.json"
    replay = json.loads(replay_path.read_text())
    drift = float(np.max(arrays["drift"], initial=0.)*1000.)
    rotation = float(np.max(arrays["rot_err_deg"], initial=0.))
    penetration = float(metrics["penetration"]["hand_object"]["max_mm"])
    penetration_status = _penetration_acceptance(cfg, penetration)
    ca = np.load(artifacts.contactaware_dir / "wuji_right_qpos.npy", allow_pickle=False)[0]
    audit_path = artifacts.root / "contact_audit/replay.json"
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    audit["official_trajectory_matched"] = False
    audit_states = artifacts.root / "contact_audit/native_replay.npz"
    if audit_states.is_file():
        stride = cfg.simulator.time_grid.executor_substeps_per_mpc_step
        with np.load(audit_states, allow_pickle=False) as trace:
            trace_q, trace_v = trace["qpos_with_initial"], trace["qvel_with_initial"]
            if trace_q.shape != (len(ctrl)*stride+1, 33) or trace_v.shape != (len(ctrl)*stride+1, 32):
                raise ValueError("Physical contact audit does not cover the official rollout")
            sampled_q, sampled_v = trace_q[::stride], trace_v[::stride]
        if sampled_q.shape != qpos.shape or sampled_v.shape != qvel.shape:
            raise ValueError("Physical contact audit does not cover the official rollout")
        audit.update(official_qpos_max_difference=float(np.max(np.abs(sampled_q-qpos))),
                     official_qvel_max_difference=float(np.max(np.abs(sampled_v-qvel))))
        audit["official_trajectory_matched"] = bool(np.allclose(sampled_q, qpos, rtol=0, atol=2e-6)
            and np.allclose(sampled_v, qvel, rtol=0, atol=2e-5))
    native_physics_drift = float(audit.get("object_displacement_max_mm", drift))
    native_physics_rotation = float(audit.get("object_rotation_max_rad", np.deg2rad(rotation)))*180./np.pi
    native_hold = bool(finite and complete and np.isfinite([native_physics_drift, native_physics_rotation]).all()
                       and max(drift, native_physics_drift) <= 2.
                       and max(rotation, native_physics_rotation) <= 5.
                       and penetration_status["within_soft_limit"])
    contact_retained = audit.get("finger_has_positive_force_all_steps", [])
    all_fingers = bool(audit.get("finite") is True and audit["official_trajectory_matched"]
                       and len(contact_retained) == 5 and all(value is True for value in contact_retained))
    replay_hold = bool(replay["summary"]["position_mm"]["max"] <= 2. and
                       replay["summary"].get("rotation_deg", {}).get("max", float("inf")) <= 5.)
    mesh_audit_path = artifacts.root / "mesh_contact_audit/mesh_contact_audit.json"
    mesh_audit = json.loads(mesh_audit_path.read_text()) if mesh_audit_path.is_file() else {}
    replay_hold = bool(replay_hold
        and float(mesh_audit.get("object_displacement_max_mm", 0.)) <= 2.
        and float(mesh_audit.get("object_rotation_max_rad", 0.))*180./np.pi <= 5.)
    mesh_contacts = bool(mesh_audit.get("finite") is True
        and mesh_audit.get("trajectory_matches_official_replay") is True
        and mesh_audit.get("all_fingers_continuously_positive") is True)
    wrist_delta = Rotation.from_euler("XYZ", ca[3:6]).inv() * Rotation.from_euler("XYZ", qpos[0, 3:6])
    state_policy_path = artifacts.root / "initial_state_policy.json"
    state_policy = json.loads(state_policy_path.read_text()) if state_policy_path.is_file() else dict(initial_state_correction_enabled=True)
    summary = dict(method="dexforge", source="original DexForge ForceAware optimizer/objective with explicit task adapters",
        solver_completed=bool(finite and complete), requested_duration_seconds=cfg.sequence.hold_seconds,
        rollout_status=status,
        paper_tracking_criterion=_paper_tracking_criterion(arrays["drift"], arrays["rot_err_deg"]),
        executed_duration_seconds=len(ctrl)*dt, task_success=bool(native_hold and all_fingers and replay_hold and mesh_contacts),
        native_hold_success=native_hold, five_finger_loaded_contact_retained=all_fingers,
        independent_mujoco_hold_success=replay_hold, physical_contact_audit=audit,
        independent_mujoco_five_finger_contact_retained=mesh_contacts,
        independent_mujoco_contact_audit=mesh_audit,
        acceptance=dict(max_object_drift_mm=2., max_object_rotation_deg=5.,
            max_penetration_mm=penetration_status["soft_limit_mm"],
            soft_penetration_limit_mm=penetration_status["soft_limit_mm"],
            deep_penetration_limit_mm=penetration_status["deep_limit_mm"],
            contact_target_phi_mm=penetration_status["target_phi_mm"],
            penetration_limit_source="cfg.contact.soft_penetration_limit / penetration_limit"),
        object_drift_max_mm=drift, object_rotation_max_deg=rotation, penetration_max_mm=penetration,
        native_physics_step_object_drift_max_mm=native_physics_drift,
        native_physics_step_object_rotation_max_deg=native_physics_rotation,
        penetration=penetration_status,
        original_initial_correction_enabled=state_policy["initial_state_correction_enabled"],
        initial_state_policy=state_policy,
        initial_correction_translation_mm=float(np.linalg.norm(qpos[0, :3]-ca[:3])*1000.),
        initial_correction_wrist_rotation_deg=float(wrist_delta.magnitude()*180/np.pi),
        initial_correction_finger_max_deg=float(np.max(np.abs(qpos[0, 6:26]-ca[6:26]))*180/np.pi),
        native_dynamics_executed=True, native_control_gradient_passed=validation.get("native_control_gradient_passed", False),
        full_objective_gradient_certified=False, independent_mujoco_replay_tested=True,
        original_default_replay_supported=not replay.get("independent_replay_time_grid_adapter", False),
        independent_mujoco_replay=replay["summary"], hardware_tested=False,
        original_outputs=str(native), reference_source="dexforge_contactaware")
    summary["controller_changed"] = (artifacts.root / "controller_variant_manifest.json").is_file()
    adapter_path = artifacts.root / "dynamics_adapter.json"
    summary["dynamics_adapter"] = json.loads(adapter_path.read_text()) if adapter_path.is_file() else dict(mode="upstream_euler", upstream_runtime_modified=False)
    summary["upstream_optimizer_and_objective_unmodified"] = True
    preload_path = artifacts.root / "control_preload.json"
    summary["control_preload"] = json.loads(preload_path.read_text()) if preload_path.is_file() else None
    if window_seconds is not None:
        if (window_seconds.ndim != 1 or not len(window_seconds)
                or not np.isfinite(window_seconds).all() or np.any(window_seconds < 0)):
            raise ValueError("Original FA MPC timing must be a nonempty finite nonnegative vector")
        summary["timing"] = dict(optimization_seconds=float(window_seconds.sum()),
            rolling_windows=len(window_seconds), mean_window_seconds=float(window_seconds.mean()),
            source="original rollout.npz mpc_step_sec; excludes model/graph preparation and output/replay",
            optimizer_iterations=cfg.optimizer.n_iter, multistart_count=cfg.optimizer.multistart_count,
            planning_dt=cfg.simulator.time_grid.mpc_dt, execution_dt=cfg.simulator.time_grid.exec_dt)
    write_json(artifacts.root / "summary.json", summary)
    return summary


def run(config, output):
    from .forceaware_scene import prepare_scene
    from .forceaware_reference import prepare_reference, make_config
    from retarget.dynamics.dexforge_upstream import verify_upstream
    options = _options(config)
    optimizer, simulator = _solver_overrides(config.solver.dexforge)
    cases, summaries = [], []
    def rejected(case, stage, report):
        summary = dict(name=case.name, method="dexforge", solver_completed=False, task_success=False,
            failed_stage=stage, failure_report=str(report), native_dynamics_executed=False,
            hardware_tested=False, upstream_optimizer_and_objective_unmodified=True)
        case.mkdir(parents=True, exist_ok=True)
        write_json(case / "summary.json", summary)
        summaries.append(summary)
        print(f"[grasp FA] rejected {stage}; see {report}", flush=True)
    for number, source in enumerate(validate(config)):
        case_started = time.perf_counter()
        case = Path(output) / f"{number:03d}_{source.parents[1].name}"
        print(f"[grasp FA] preparing {source}", flush=True)
        if options["contactaware_refinement"]:
            from .forceaware_ca_refinement import refine_saved_reference
            from .forceaware_reference import _read_ca
            write_json(Path(output) / "status.json", dict(state="running", stage="contactaware_refinement", case=case.name))
            source = refine_saved_reference(source, case / "ca_input", **options["contactaware_refinement"])
            ca_report = source / "contactaware_reference.json"
            if json.loads(ca_report.read_text()).get("success") is not True:
                rejected(case, "contactaware_refinement", ca_report)
                continue
            _read_ca(source)
        scene = prepare_scene(source / "sequence/scene/wuji_right/primitive_sdf.xml", case / "native_scene",
            timestep=config.solver.dexforge.mpc_dt, sphere_radius=options["sphere_radius_mm"]*.001,
            sphere_fit=options["sphere_fit"])
        native_xml, viewer_xml = scene.xml, scene.viewer_xml
        if options.get("virtual_wrist_rotation_pd") is not None:
            import shutil
            from .forceaware_controller import apply_virtual_wrist_rotation_pd
            native_xml, viewer_xml, controller_manifest = apply_virtual_wrist_rotation_pd(
                native_xml, viewer_xml, case / "controller_scene", **options["virtual_wrist_rotation_pd"])
            shutil.copy2(controller_manifest, case / "controller_variant_manifest.json")
        if options.get("finger_servo_scale") is not None:
            import shutil
            from .forceaware_controller import scale_finger_servos
            native_xml, viewer_xml, controller_manifest = scale_finger_servos(
                native_xml, viewer_xml, case / "controller_scene", **options["finger_servo_scale"])
            shutil.copy2(controller_manifest, case / "controller_variant_manifest.json")
        artifacts = prepare_reference(source, case, native_xml, viewer_xml=viewer_xml)
        cfg = make_config(artifacts, duration_seconds=options["duration_seconds"], optimizer_overrides=optimizer,
                          simulator_overrides=simulator, contact_topk=config.solver.dexforge.contact_topk)
        reference_manifest = json.loads(artifacts.manifest_path.read_text())
        reference_manifest.update(task_fixed_initial_state=options["fixed_initial_state"],
                                  task_control_preload=options["control_preload"],
                                  task_servo_integration=options["servo_integration"])
        write_json(artifacts.manifest_path, reference_manifest)
        if options["control_preload"]:
            from .forceaware_preload import prepare_preload
            preload_options = options["control_preload"] if isinstance(options["control_preload"], dict) else {}
            command, preload = prepare_preload(artifacts, cfg, device=config.device, **preload_options)
            write_json(case / "control_preload.json", preload)
            if preload.get("feasible") is not True:
                rejected(case, "static_actuator_preload", case / "control_preload.json")
                continue
            hand_reference = np.load(artifacts.contactaware_dir / "wuji_right_qpos.npy", allow_pickle=False)[0]
            np.savez_compressed(case / "control_preload.npz", command=command, hand_reference_qpos=hand_reference)
        write_json(case / "upstream_verification.json", verify_upstream())
        print(f"[grasp FA] native validation {case.name}", flush=True)
        write_json(Path(output) / "status.json", dict(state="running", stage="native_validation", case=case.name))
        _job(artifacts, config, "diagnose", case / "validation", options)
        validation = json.loads((case / "validation/validation.json").read_text())
        if options["servo_integration"] == "implicit_actuator":
            mismatches = [test for probe in validation.get("gradient_probes", [])
                          for test in probe.get("directional_tests", [])
                          if test.get("outcome") == "mismatch_on_same_branch"]
            if not validation.get("finite") or mismatches:
                raise RuntimeError(f"Repaired native dynamics prerequisite failed; see {case / 'validation/validation.json'}")
        if not validation.get("finite"):
            print("[grasp FA] constant-control baseline is nonfinite; the original multistart validity checks will decide the optimization result", flush=True)
        print(f"[grasp FA] original optimization {case.name}", flush=True)
        write_json(Path(output) / "status.json", dict(state="running", stage="original_optimization", case=case.name))
        _job(artifacts, config, "solve", case, options)
        _job(artifacts, config, "audit", case / "contact_audit", options)
        summary = collect(artifacts, cfg, validation)
        from .grasp_evaluation import apply_evaluation
        with np.load(artifacts.root / "trajectory.npz", allow_pickle=False) as final_trace:
            final_qpos = final_trace["qpos"][-1].copy()
        apply_evaluation(summary, artifacts.viewer_xml, final_qpos, artifacts.root, method="dexforge",
                         selection=dict(kind="final_executed_state", path=str(artifacts.root / "trajectory.npz")))
        summary.setdefault("timing", {})["case_wall_seconds"] = time.perf_counter()-case_started
        summary["timing"]["case_wall_scope"] = "scene/reference/preload, native validation, original solve/output/replay and independent contact audits"
        write_json(case / "summary.json", summary)
        relative = case.relative_to(output)
        cases.append(dict(task="grasp", name=case.name, hand="wuji2", method="dexforge",
            model=str(artifacts.viewer_xml.relative_to(output)), trajectory=str(relative / "trajectory.npz"),
            summary=str(relative / "summary.json"), dt=cfg.simulator.time_grid.mpc_dt))
        summaries.append(dict(name=case.name, **summary))
    write_json(Path(output) / "summary.json", dict(cases=summaries, evaluation_protocol="graspadmm_native_v1",
               all_solver_completed=bool(summaries) and all(s["solver_completed"] for s in summaries),
               all_task_success=bool(summaries) and all(s["task_success"] for s in summaries)))
    return cases


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("diagnose", "solve", "audit"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gradient-horizon", type=int, default=10)
    parser.add_argument("--gradient-directions", type=int, default=1)
    parser.add_argument("--skip-gradients", action="store_true")
    parser.add_argument("--servo-integration", choices=("upstream_euler", "implicit_actuator"), default="upstream_euler")
    parser.add_argument("--control-preload", action="store_true")
    parser.add_argument("--fixed-initial-state", action="store_true")
    parser.add_argument("--baseline-duration", type=float, default=.1)
    args = parser.parse_args()
    from retarget.dynamics.dexforge_upstream import activate_upstream, require_environment
    from .forceaware_reference import register_wuji2_hand, ReferenceArtifacts
    require_environment(); activate_upstream()
    dynamics_adapter = dict(mode="upstream_euler", upstream_runtime_modified=False)
    if args.servo_integration == "implicit_actuator":
        from retarget.dynamics.dexforge_servo_integration import activate_servo_integration
        dynamics_adapter = dict(activate_servo_integration(), mode=args.servo_integration,
                                upstream_runtime_modified=True)
    write_json(args.root / "dynamics_adapter.json", dynamics_adapter)
    register_wuji2_hand()
    from forceaware.config import load_retarget_config
    from forceaware.runtime import setup_runtime
    cfg = load_retarget_config(args.root / "sequence", args.config)
    os.environ["FORCEAWARE_DEVICE"] = args.device
    setup_runtime(cfg)
    preload_command = None
    if args.control_preload:
        with np.load(args.root / "control_preload.npz", allow_pickle=False) as raw:
            preload_command = raw["command"].copy()
    if args.action == "solve":
        if args.fixed_initial_state:
            from .forceaware_initialization import activate_fixed_initial_state
            write_json(args.root / "initial_state_policy.json", activate_fixed_initial_state())
        if args.control_preload:
            from .forceaware_initialization import activate_control_preload
            write_json(args.root / "control_initialization.json", activate_control_preload(args.root))
        from forceaware.workflow import rolling_mpc_retarget
        try:
            rolling_mpc_retarget(cfg, config_path=args.config)
        except ValueError as error:
            # Official FA writes its complete native result before replay. Its
            # independent mesh replay has a fixed 2ms grid, which cannot express
            # WUJI's smaller control intervals. Keep that failure visible and
            # use the same upstream replay integrator at an explicit finer dt.
            from forceaware.replay import DEFAULT_REPLAY_SIM_DT
            expected = f"action_dt={cfg.simulator.time_grid.mpc_dt:g} must be an integer multiple of replay sim_dt={DEFAULT_REPLAY_SIM_DT:g}"
            native = cfg.sequence.seq_dir / cfg.sequence.output_dir_name
            if str(error) != expected or cfg.simulator.time_grid.mpc_dt >= DEFAULT_REPLAY_SIM_DT or not (native / "rollout.npz").is_file():
                raise
            write_json(args.root / "original_replay_grid_limitation.json", dict(
                original_workflow_error=str(error), original_default_replay_supported=False,
                optimizer_and_native_execution_completed=True, upstream_source_modified=False))
            from .forceaware_replay import replay_mesh
            replay_mesh(cfg, native / "rollout.npz", native / "replay/mujoco_cpu")
    else:
        artifacts = ReferenceArtifacts(args.root, args.root, cfg.sequence.seq_dir, cfg.sequence.contact_guidance_npz.parent,
            cfg.sequence.xml, cfg.sequence.xml.with_name("primitive_mesh.xml"), args.config, args.root / "reference_manifest.json")
        if args.action == "diagnose":
            from .forceaware_validation import diagnose_native
            report = diagnose_native(artifacts, cfg, args.output, horizon=args.gradient_horizon,
                                     directions=args.gradient_directions, skip_gradients=args.skip_gradients,
                                     duration_seconds=args.baseline_duration, control=preload_command)
            report["dynamics_adapter"] = dynamics_adapter
            write_json(args.output / "validation.json", report)
        else:
            from .forceaware_validation import audit_native_controls
            rollout = cfg.sequence.seq_dir / cfg.sequence.output_dir_name / "rollout.npz"
            with np.load(rollout, allow_pickle=False) as raw:
                audit_native_controls(artifacts, cfg, args.output, qpos=raw["qpos_traj"][0], qvel=raw["qvel_traj"][0], controls=raw["ctrl_traj"])
            from .forceaware_mesh_audit import audit_mesh_controls
            audit_mesh_controls(cfg, rollout, args.root / "mesh_contact_audit")


if __name__ == "__main__":
    _main()
