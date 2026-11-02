"""Native grasp diagnostics preserve WUJI coordinates and physical forces."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from retarget.env.grasp.forceaware_validation import (
    _native_fd_assessment, constant_native_control, decode_contact_forces, expand_controls,
    finger_body_labels, finger_contact_metrics, object_motion,
)


def test_object_rotation_uses_only_free_object_quaternion_and_accepts_sign_flip():
    q = np.zeros((3, 33))
    q[:, 3:7] = [[.8, -.6, .2, 1.4], [1., .7, -.5, .3], [2., 3., 4., 5.]]
    q[:, 26:29] = [[0., 0., 0.], [.003, .004, 0.], [0., 0., .012]]
    q[:, 29:33] = [[1., 0., 0., 0.], [-1., 0., 0., 0.], [np.cos(.1), 0., np.sin(.1), 0.]]
    original = q.copy()
    motion = object_motion(q, 26)
    np.testing.assert_allclose(motion["object_displacement_mm"], [0., 5., 12.])
    np.testing.assert_allclose(motion["object_rotation_rad"], [0., 0., .2], atol=1e-7)
    np.testing.assert_array_equal(q, original)


def test_native_force_rows_include_friction_and_exclude_joint_limits_and_inactive_contacts():
    frame = np.tile(np.eye(3), (2, 2, 1, 1))
    row_contact, row_edge = np.array([0, 0, 0, 0, 1, -1]), np.array([0, 1, 2, 3, 0, -1])
    friction = np.array([[.5, .25], [1., 1.]])
    rows = np.array([[1., 2., 3., 4., 7., 900.], [1., 2., 3., 4., 7., 900.]])
    active = np.array([[True, False], [False, True]])
    force, pressure = decode_contact_forces(row_contact, row_edge, friction, frame, rows, active)
    np.testing.assert_allclose(force[0], [[10., -.5, -.25], [0., 0., 0.]])
    np.testing.assert_allclose(force[1], [[0., 0., 0.], [7., 7., 0.]])
    np.testing.assert_allclose(pressure, [[10., 0.], [0., 7.]])


def test_finger_force_uses_object_pairs_and_correct_geom_force_sign():
    geom_body = np.array([3, 2, 4, 1])
    # Object body3: pair0 force points into object, pair1 is on hand, pair2 self-contact.
    pairs = np.array([[1, 0], [0, 2], [1, 3]])
    body_finger = np.array([-1, 4, 0, -1, 2])
    active = np.array([[True, True, True], [True, True, True]])
    distance = np.array([[.0002, -.0004, -.1], [.003, .004, -.1]])
    pressure = np.array([[2., 3., 100.], [0., 0., 100.]])
    force = np.array([[[2., 1., 0.], [3., 0., 0.], [100., 0., 0.]],
                      [[0., 0., 0.], [0., 0., 0.], [100., 0., 0.]]])
    result = finger_contact_metrics(pairs, geom_body, body_finger, 3, active, distance, force, pressure)
    np.testing.assert_allclose(result["finger_normal_force_n"], [[2., 0., 3., 0., 0.], [0., 0., 0., 0., 0.]])
    np.testing.assert_allclose(result["object_contact_force_world_n"], [[-1., 1., 0.], [0., 0., 0.]])
    np.testing.assert_allclose(result["finger_min_native_gap_mm"][:, [0, 2]], [[.2, -.4], [3., 4.]])
    np.testing.assert_array_equal(result["finger_loaded_contact_count"][:, [0, 2]], [[1, 1], [0, 0]])


def test_fixed_sensor_descendants_belong_to_their_finger():
    names = ["world", "right_palm_link", "right_finger1_link4", "r_thumb_sensor", "right_finger5_link3", "r_little_pad"]
    model = SimpleNamespace(nbody=len(names), body_parentid=np.array([0, 0, 1, 2, 1, 4]),
                            body=lambda i: SimpleNamespace(name=names[i]))
    np.testing.assert_array_equal(finger_body_labels(model), [-1, -1, 0, 0, 4, 4])


def test_object_motion_rejects_degenerate_quaternion():
    with pytest.raises(ValueError, match="Zero free-object quaternion"):
        object_motion(np.zeros((2, 33)), 26)


def test_replay_preserves_official_action_order_and_physics_substeps():
    actions = np.array([[1., 2.], [3., 4.]])
    expected = np.array([[2., 0., 1.], [2., 0., 1.], [4., 0., 3.], [4., 0., 3.]])
    np.testing.assert_array_equal(expand_controls(actions, [2, 0], 3, control_dt=.01, exec_dt=.005), expected)
    with pytest.raises(ValueError, match="integer multiple"):
        expand_controls(actions, [2, 0], 3, control_dt=.01, exec_dt=.003)


def test_constant_preload_command_preserves_ca_state_and_uses_same_actuator_mapping():
    index = SimpleNamespace(hand_ctrl=np.array([2, 0]), hand_qpos=np.array([0, 1]))
    q0 = np.array([.1, .2, 0., 0., 0., 1.])
    initial = q0.copy()
    np.testing.assert_allclose(constant_native_control(index, q0, 3), [.2, 0., .1])
    np.testing.assert_allclose(constant_native_control(index, q0, 3, [.8, .9]), [.9, 0., .8])
    np.testing.assert_allclose(constant_native_control(index, q0, 3, [.8, .9, .7]), [.8, .9, .7])
    np.testing.assert_array_equal(q0, initial)
    with pytest.raises(ValueError, match="shape"):
        constant_native_control(index, q0, 3, np.zeros((1, 2)))
    with pytest.raises(ValueError, match="finite"):
        constant_native_control(index, q0, 3, [float("nan"), .2])
    with pytest.raises(ValueError, match="finite"):
        constant_native_control(index, q0, 3, [.1, float("inf")])


def test_independent_mesh_replay_grid_can_represent_submillisecond_controls():
    from retarget.env.grasp.forceaware_replay import replay_sim_dt
    assert replay_sim_dt(.0001) == .0001
    assert replay_sim_dt(.01) == .002
    for invalid in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            replay_sim_dt(invalid)


def test_native_gradient_gate_rejects_real_coarse_branch_switch_false_plateau():
    """The calibrated WUJI case had unchanged geometry but changed Newton sets."""
    from retarget.env.humanoid.dexforge_runner import _fd_assessment
    values = [(1e-3, -.38420917826442746, False), (3e-4, -.3975930834106597, False),
              (1e-4, .34076351766998414, True), (3e-5, .3359385232215561, True)]
    trials = [dict(epsilon=epsilon, derivative=value, same_contact_active=True,
        same_contact_features=True, same_contact_solver_branch=same_branch,
        same_actuator_branch=True, control_direction_error_norm=.0034) for epsilon, value, same_branch in values]
    adjoint = .33893873851539474
    previous = _fd_assessment(adjoint, trials, rtol=.05, atol=.002)
    assert not previous["passed"] and previous["selected_epsilon"] == 1e-3
    result = _native_fd_assessment(adjoint, trials, rtol=.05, atol=.002)
    assert result["passed"] and result["selected_epsilon"] == 1e-4
    assert result["outcome"] == "passed"


def test_native_gradient_gate_preserves_mismatch_and_reports_unresolved_controls():
    trials = [dict(epsilon=epsilon, derivative=-.4, same_contact_active=True,
        same_contact_features=True, same_contact_solver_branch=True,
        same_actuator_branch=True, control_direction_error_norm=0.) for epsilon in (1e-3, 3e-4)]
    mismatch = _native_fd_assessment(.3, trials, rtol=.05, atol=.002)
    assert not mismatch["passed"] and mismatch["outcome"] == "mismatch_on_same_branch"
    for row in trials:
        row["control_direction_error_norm"] = .1
    unresolved = _native_fd_assessment(.3, trials, rtol=.05, atol=.002)
    assert not unresolved["passed"] and unresolved["outcome"] == "inconclusive"


def test_saved_real_native_validation_contains_physics_step_force_and_velocity_traces():
    """Opt in to a produced real runtime report; never substitute mock physics."""
    destination = os.environ.get("GRASP_FORCEAWARE_VALIDATION")
    if not destination:
        pytest.skip("Set GRASP_FORCEAWARE_VALIDATION to an actual check_grasp_forceaware output")
    folder = Path(destination)
    report = json.loads((folder / "validation.json").read_text())
    assert report["upstream_commit"] == "fa6d671948ef03d18514062748b94e445b63cd49"
    assert report["baseline"]["native_collision_every_step"]
    assert report["baseline"]["native_qvel_state"]
    data = np.load(folder / "native_baseline.npz", allow_pickle=False)
    count = report["baseline"]["steps"]
    assert data["qpos"].shape == (count, 33)
    assert data["qvel"].shape == (count, 32)
    assert data["finger_normal_force_n"].shape == (count, 5)
    assert data["object_position_m"].shape == (count + 1, 3)
    finite = all(np.isfinite(data[name]).all() for name in ("qpos", "qvel", "contact_row_force", "contact_force_on_geom2_n"))
    assert report["finite"] == finite
    if not finite:
        first_bad = np.flatnonzero(~np.isfinite(data["qpos"]).all(1) | ~np.isfinite(data["qvel"]).all(1))[0] + 1
        assert report["baseline"]["first_nonfinite_post_step"] == first_bad
        assert not report["native_control_gradient_passed"]
    np.testing.assert_allclose(np.diff(data["time"]), report["baseline"]["dt"], atol=1e-7)
    assert all(probe["horizon"] in (1, 10) for probe in report.get("gradient_probes", []))
    assert not report["certified"]  # Native linear probes alone do not certify the grasp objective.
