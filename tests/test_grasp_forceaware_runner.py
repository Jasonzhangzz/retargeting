"""Configuration and output accounting for the original WUJI2 FA workflow."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from retarget.env.grasp.forceaware_runner import _options, _solver_overrides, _rollout_status, collect
from retarget.solver.dexforge import DexForgeSettings


def test_original_costs_cannot_be_silently_replaced_by_humanoid_overrides():
    assert _solver_overrides(DexForgeSettings()) == ({}, {})
    with pytest.raises(ValueError, match="original ForceAware costs"):
        _solver_overrides(replace(DexForgeSettings(), joint_ref_weight=2.))
    optimizer, simulator = _solver_overrides(replace(DexForgeSettings(), n_iter=10, mpc_dt=.005, exec_dt=.005))
    assert optimizer == {"n_iter": 10}
    assert simulator == {"mpc_dt": .005, "exec_dt": .005}


@pytest.mark.parametrize("value", [{"case_indices": [1, 1]}, {"duration_seconds": float("nan")},
                                    {"gradient_check": 1}, {"gradient_horizon": 0}, {"servo_integration": "unknown"},
                                    {"fixed_initial_state": 1}, {"control_preload": {"minimum_total": -1}},
                                    {"control_preload": {"iterations": 1.5}}, {"control_preload": {"unknown": 1}},
                                    {"contactaware_refinement": {"normal_cosine": 1.}},
                                    {"contactaware_refinement": {"normal_cosine": True}}])
def test_invalid_runtime_options_are_rejected(value):
    with pytest.raises(ValueError):
        _options(SimpleNamespace(environment={"dexforge": value}))


def test_rejected_ca_inputs_are_recorded_without_entering_physics_or_aborting_batch(tmp_path, monkeypatch):
    from retarget.env.grasp import forceaware_runner as runner
    from retarget.env.grasp import forceaware_ca_refinement as refinement
    from retarget.env.grasp import forceaware_reference as reference
    sources = [tmp_path / f"source{i}/sequence/input_00" for i in range(2)]
    monkeypatch.setattr(runner, "validate", lambda config: sources)
    def reject(source, destination, **kwargs):
        destination.mkdir(parents=True)
        (destination / "contactaware_reference.json").write_text(json.dumps(dict(success=False)))
        return destination
    monkeypatch.setattr(refinement, "refine_saved_reference", reject)
    def cannot_seed(*args, **kwargs):
        raise AssertionError("Rejected CA candidates must never reach FA inputs")
    monkeypatch.setattr(reference, "_read_ca", cannot_seed)
    cfg = SimpleNamespace(environment={"dexforge": {"contactaware_refinement": True}},
        solver=SimpleNamespace(dexforge=DexForgeSettings()))
    destination = tmp_path / "results"
    cases = runner.run(cfg, destination)
    report = json.loads((destination / "summary.json").read_text())
    assert cases == [] and len(report["cases"]) == 2
    assert report["all_task_success"] is False and report["all_solver_completed"] is False
    assert all(case["failed_stage"] == "contactaware_refinement" for case in report["cases"])


def _recorded_case(tmp_path, *, penetration_mm=1., with_audit=False):
    """Format-only saved results test acceptance; no synthetic physics claim."""
    from retarget.dynamics.dexforge_upstream import activate_upstream
    activate_upstream()
    from forceaware.time_grid import TimeGrid

    contact = tmp_path / "contactaware"
    native = tmp_path / "sequence/forceaware"
    replay = native / "replay/mujoco_cpu"
    contact.mkdir(); replay.mkdir(parents=True)
    q = np.zeros((3, 33)); q[:, 29] = 1.; q[0, 0] = .001
    np.save(contact / "wuji_right_qpos.npy", np.zeros((2, 26)))
    arrays = dict(qpos_traj=q, qvel_traj=np.zeros((3, 32)),
        ctrl_traj=np.zeros((2, 26)), drift=np.asarray([.001, .001]), rot_err_deg=np.asarray([1., 2.]),
        object_reference_qpos=q[:, 26:33])
    np.savez(native / "rollout.npz", **arrays)
    (native / "metrics.json").write_text(json.dumps({"penetration": {"hand_object": {"max_mm": penetration_mm}}}))
    (replay / "replay.json").write_text(json.dumps({"summary": {"position_mm": {"max": 1.}, "rotation_deg": {"max": 2.}}}))
    artifacts = SimpleNamespace(root=tmp_path, contactaware_dir=contact)
    cfg = SimpleNamespace(sequence=SimpleNamespace(seq_dir=tmp_path / "sequence", output_dir_name="forceaware", hold_seconds=.02),
        simulator=SimpleNamespace(time_grid=TimeGrid.create(mpc_dt=.01, exec_dt=.01,
            action_dt=.03, knot_dt=.06, ref_dt=1/30)),
        contact=SimpleNamespace(target_phi=-.004, soft_penetration_limit=.004, penetration_limit=.008))
    if with_audit:
        audit = tmp_path / "contact_audit"; audit.mkdir()
        (audit / "replay.json").write_text(json.dumps(dict(finite=True, finger_has_positive_force_all_steps=[True]*5)))
        np.savez(audit / "native_replay.npz", qpos_with_initial=q, qvel_with_initial=arrays["qvel_traj"])
        mesh_audit = tmp_path / "mesh_contact_audit"; mesh_audit.mkdir()
        (mesh_audit / "mesh_contact_audit.json").write_text(json.dumps(dict(finite=True,
            trajectory_matches_official_replay=True, all_fingers_continuously_positive=True)))
    return artifacts, cfg, arrays, native


def test_solver_completion_does_not_imply_stable_hold_or_mesh_transfer(tmp_path):
    artifacts, cfg, arrays, native = _recorded_case(tmp_path)
    arrays["drift"][-1] = .003
    np.savez(native / "rollout.npz", **arrays)
    (native / "replay/mujoco_cpu/replay.json").write_text(json.dumps({"summary": {"position_mm": {"max": 12.}}}))
    summary = collect(artifacts, cfg, {"native_control_gradient_passed": False})
    assert summary["solver_completed"] is True
    assert summary["task_success"] is False
    assert summary["object_drift_max_mm"] == 3.
    assert summary["initial_correction_translation_mm"] == 1.
    assert summary["independent_mujoco_replay"]["position_mm"]["max"] == 12.
    assert summary["full_objective_gradient_certified"] is False
    with np.load(tmp_path / "trajectory.npz") as raw:
        np.testing.assert_allclose(raw["time"], [0., .01, .02])


@pytest.mark.parametrize("audit", [{}, {"finite": True, "trajectory_matches_official_replay": False,
    "all_fingers_continuously_positive": True}, {"finite": True, "trajectory_matches_official_replay": True,
    "all_fingers_continuously_positive": False}])
def test_mesh_drift_alone_cannot_certify_five_finger_hold(tmp_path, audit):
    artifacts, cfg, _, _ = _recorded_case(tmp_path, with_audit=True)
    (tmp_path / "mesh_contact_audit/mesh_contact_audit.json").write_text(json.dumps(audit))
    result = collect(artifacts, cfg, {})
    assert result["native_hold_success"] is True
    assert result["five_finger_loaded_contact_retained"] is True
    assert result["independent_mujoco_hold_success"] is True
    assert result["independent_mujoco_five_finger_contact_retained"] is False
    assert result["task_success"] is False


@pytest.mark.parametrize("engine", ["native", "mesh"])
def test_hold_checks_physics_substeps_between_saved_control_states(tmp_path, engine):
    artifacts, cfg, _, _ = _recorded_case(tmp_path, with_audit=True)
    path = (tmp_path / "contact_audit/replay.json" if engine == "native" else
            tmp_path / "mesh_contact_audit/mesh_contact_audit.json")
    audit = json.loads(path.read_text()); audit["object_displacement_max_mm"] = 2.1
    path.write_text(json.dumps(audit))
    result = collect(artifacts, cfg, {})
    assert result["task_success"] is False
    assert result["native_hold_success" if engine == "native" else "independent_mujoco_hold_success"] is False


def test_original_window_timing_is_not_counted_once_per_executed_state(tmp_path):
    artifacts, cfg, arrays, native = _recorded_case(tmp_path, with_audit=True)
    cfg.optimizer = SimpleNamespace(n_iter=40, multistart_count=64)
    # One solve window executes two states; the original recorder contains
    # both per-window and repeated per-state timing. Only count the former.
    np.savez(native / "rollout.npz", **arrays, mpc_step_sec=np.asarray([.25]),
             multistart_step_sec=np.asarray([.25, .25]))
    result = collect(artifacts, cfg, {})
    assert result["timing"]["optimization_seconds"] == .25
    assert result["timing"]["rolling_windows"] == 1
    assert result["timing"]["optimizer_iterations"] == 40
    assert result["timing"]["multistart_count"] == 64


def test_paper_mean_tracking_and_strict_five_finger_hold_are_distinct(tmp_path):
    artifacts, cfg, arrays, native = _recorded_case(tmp_path)
    arrays["drift"][:] = [.003, .004]
    np.savez(native / "rollout.npz", **arrays)
    result = collect(artifacts, cfg, {})
    assert result["paper_tracking_criterion"]["success"] is True
    assert result["task_success"] is False
    assert result["native_hold_success"] is False
    assert result["paper_tracking_criterion"]["mean_position_m"] == pytest.approx(.0035)
    assert result["paper_tracking_criterion"]["full_paper_benchmark_reproduced"] is False


@pytest.mark.parametrize("position,rotation", [([.01, .01], [0., 0.]),
    ([0., 0.], [12., 12.]), ([np.nan], [0.]), ([], [])])
def test_paper_tracking_criterion_keeps_thresholds_and_rejects_invalid_data(position, rotation):
    from retarget.env.grasp.forceaware_runner import _paper_tracking_criterion
    assert _paper_tracking_criterion(position, rotation)["success"] is False


@pytest.mark.parametrize("depth,soft_ok,deep_ok", [
    (3., True, True), (4., True, True), (4.001, False, True),
    (8., False, True), (8.001, False, False)])
def test_penetration_uses_configured_soft_and_deep_objective_depths(tmp_path, depth, soft_ok, deep_ok):
    artifacts, cfg, _, _ = _recorded_case(tmp_path, penetration_mm=depth, with_audit=True)
    summary = collect(artifacts, cfg, {})
    assert summary["solver_completed"] is True
    assert summary["native_hold_success"] is soft_ok
    assert summary["task_success"] is soft_ok
    assert summary["five_finger_loaded_contact_retained"] is True
    assert summary["acceptance"]["max_penetration_mm"] == 4.
    assert summary["acceptance"]["contact_target_phi_mm"] == -4.
    assert summary["penetration"]["within_soft_limit"] is soft_ok
    assert summary["penetration"]["within_deep_limit"] is deep_ok
    assert summary["penetration"]["soft_excess_max_mm"] == pytest.approx(max(depth-4., 0.))
    assert summary["penetration"]["deep_excess_max_mm"] == pytest.approx(max(depth-8., 0.))


def test_penetration_acceptance_follows_actual_cfg_instead_of_a_fixed_number(tmp_path):
    artifacts, cfg, _, _ = _recorded_case(tmp_path, penetration_mm=3., with_audit=True)
    cfg.contact = SimpleNamespace(target_phi=-.002, soft_penetration_limit=.002, penetration_limit=.006)
    summary = collect(artifacts, cfg, {})
    assert summary["acceptance"]["soft_penetration_limit_mm"] == 2.
    assert summary["acceptance"]["deep_penetration_limit_mm"] == 6.
    assert summary["native_hold_success"] is False
    cfg.contact.soft_penetration_limit = .003
    with pytest.raises(ValueError, match="consistent configured soft/deep"):
        collect(artifacts, cfg, {})


@pytest.mark.parametrize("key", ["qpos_traj", "qvel_traj", "ctrl_traj", "drift", "rot_err_deg", "object_reference_qpos"])
def test_solver_completion_requires_all_saved_states_controls_and_targets(tmp_path, key):
    _, cfg, arrays, _ = _recorded_case(tmp_path)
    arrays[key] = arrays[key][:-1]
    status = _rollout_status(arrays, cfg)
    assert status["finite"] is True
    assert status["complete"] is False
    assert key in status["shape_errors"]


def test_nonfinite_control_cannot_be_called_completed_or_held(tmp_path):
    artifacts, cfg, arrays, native = _recorded_case(tmp_path, with_audit=True)
    arrays["ctrl_traj"][0, 0] = np.nan
    np.savez(native / "rollout.npz", **arrays)
    summary = collect(artifacts, cfg, {})
    assert summary["rollout_status"]["complete"] is True
    assert summary["solver_completed"] is False
    assert summary["native_hold_success"] is False
    assert summary["task_success"] is False


@pytest.mark.parametrize("failure", ["missing_trace", "wrong_trajectory", "missing_finger"])
def test_five_finger_claim_requires_matching_official_audit(tmp_path, failure):
    artifacts, cfg, arrays, _ = _recorded_case(tmp_path, with_audit=True)
    audit = tmp_path / "contact_audit"
    if failure == "missing_trace":
        (audit / "native_replay.npz").unlink()
        # A stale JSON claim is insufficient without its state trace.
        (audit / "replay.json").write_text(json.dumps(dict(finite=True,
            official_trajectory_matched=True, finger_has_positive_force_all_steps=[True]*5)))
    elif failure == "wrong_trajectory":
        q = arrays["qpos_traj"].copy(); q[-1, 0] += .001
        np.savez(audit / "native_replay.npz", qpos_with_initial=q, qvel_with_initial=arrays["qvel_traj"])
    else:
        (audit / "replay.json").write_text(json.dumps(dict(finite=True, finger_has_positive_force_all_steps=[True]*4)))
    summary = collect(artifacts, cfg, {})
    assert summary["native_hold_success"] is True
    assert summary["five_finger_loaded_contact_retained"] is False
    assert summary["task_success"] is False
