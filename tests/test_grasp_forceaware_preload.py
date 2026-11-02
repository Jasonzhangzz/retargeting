"""Preload input/command contracts; synthetic records do not certify a hold."""

from types import SimpleNamespace

import numpy as np
import pytest

from retarget.env.grasp import forceaware_preload as preload


def test_convex_rim_normal_is_independent_of_tied_triangle_choice():
    import trimesh
    mesh = trimesh.creation.box(extents=(2., 2., 2.))
    points = np.asarray([[1.003, 0., -1.004], [.8, 0., -.999], [1., 0., 0.]])
    closest, distance, faces = trimesh.proximity.closest_point_naive(mesh, points)
    normals = preload.convex_surface_normals(mesh, points, closest, faces)
    np.testing.assert_allclose(normals, [[.6, 0., -.8], [0., 0., -1.], [1., 0., 0.]], atol=1e-12)
    alternative = faces.copy()
    alternative[0] = np.flatnonzero(mesh.face_normals[:, 0] > .99)[0]
    np.testing.assert_allclose(preload.convex_surface_normals(mesh, points, closest, alternative), normals)


def test_preload_feature_normals_reject_nonconvex_collision_surface():
    import trimesh
    mesh = trimesh.util.concatenate([trimesh.creation.box(),
        trimesh.creation.box(transform=trimesh.transformations.translation_matrix([2., 0., 0.]))])
    with pytest.raises(ValueError, match="closed convex"):
        preload.convex_surface_normals(mesh, [[0., 0., .6]], [[0., 0., .5]], [0])


class Buffer:
    def __init__(self, value): self.value = np.asarray(value).copy()
    def cpu(self): return self
    def numpy(self): return self.value.copy()
    def new_tensor(self, value): return Buffer(value)


def _source_values():
    model = SimpleNamespace(actuator_ctrlrange=np.tile([-1., 1.], (26, 1)),
        actuator_forcerange=np.tile([-.6, .6], (26, 1)), actuator_forcelimited=np.ones(26, bool))
    return dict(model=model, qref=np.zeros(26), bodies=np.arange(5), local=np.zeros((5, 3)),
        arms=np.zeros((5, 3)), outward=np.tile([1., 0., 0.], (5, 1)), mass=.2,
        friction=1., radius=.05, jacobian=np.zeros((5, 3, 26)), bias=np.full(26, .02),
        source_ca_commit="fixture", body_names=[f"right_finger{i}_link4" for i in range(1, 6)],
        points=np.zeros((5, 3)), distances=np.zeros(5))


@pytest.mark.parametrize("saturated", [False, True])
def test_preload_changes_only_private_bias_and_control_data(monkeypatch, saturated):
    from retarget.env.grasp import load
    values = _source_values()
    original_q = values["qref"].copy()
    original_bias = Buffer(np.zeros(26))
    matrices = dict(q=Buffer(original_q), jacobian=Buffer(values["jacobian"]), bias=original_bias)
    monkeypatch.setattr(preload, "preload_inputs", lambda artifacts: values)
    monkeypatch.setattr(load, "load_matrices", lambda *args, **kwargs: matrices)

    def solve(private, **kwargs):
        assert private is not matrices
        np.testing.assert_array_equal(private["bias"].numpy(), values["bias"])
        assert kwargs == dict(minimum_per_contact=.2, minimum_total=6., iterations=10000)
        command = original_q + .01
        if saturated: command[6] = 1.01
        return command, dict(feasible=True, actuator_torque_Nm=np.full(26, .01),
            force_N=np.zeros((5, 3)), wrench_residual=np.zeros(6),
            normal_force_N=np.full(5, 1.2), maximum_scaled_constraint_violation=0.)

    monkeypatch.setattr(load, "solve_load", solve)
    cfg = SimpleNamespace(sequence=SimpleNamespace(hand="wuji_right"), optimizer=SimpleNamespace(horizon=5))
    command, report = preload.prepare_preload(object(), cfg)
    np.testing.assert_array_equal(values["qref"], original_q)
    np.testing.assert_array_equal(original_bias.numpy(), 0.)
    assert report["hand_reference_unchanged"] and report["initial_state_unchanged"]
    assert report["candidate_command_clipped"] is False
    assert report["physical_hold_certified"] is False
    assert report["feasible"] is (not saturated)
    if saturated:
        assert command is None and report["recommended_previous_action"] is None
        assert report["command_max_violation_rad"] == pytest.approx(.01)
    else:
        np.testing.assert_array_equal(command, original_q+.01)
        np.testing.assert_array_equal(report["recommended_previous_action"], command)
        np.testing.assert_array_equal(report["recommended_knot_correction"], command-original_q)
        np.testing.assert_array_equal(report["hand_reference_qpos"], original_q)


def test_preload_uses_actual_object_face_not_anchor_or_hand_normal():
    from pathlib import Path
    from retarget.config.run import ROOT
    source = sorted((ROOT / "output/grasp_contactaware_mppi").glob("001*/*/input_00"))
    if not source:
        pytest.skip("Accepted middle-size CA material points required")
    source = source[0]
    artifacts = SimpleNamespace(root=source, viewer_xml=source/"runtime_scene.xml")
    values = preload.preload_inputs(artifacts)
    with np.load(source/"card_input.npz", allow_pickle=False) as card:
        anchors = card["object_normals"][card["reference_contact_mask"].astype(bool)]
        np.testing.assert_array_equal(values["qref"], card["hand_reference_qpos"])
    index = values["body_names"].index("right_finger2_link4")
    assert abs(anchors[index, 2]) < 1e-5
    assert values["outward"][index, 2] < -.99
    assert values["mass"] == .2 and values["friction"] == 1.
    assert max(values["distances"]) < .0005
    assert values["bias"][2] == pytest.approx(6.747318)


def test_finger_minima_follow_active_body_order_and_reject_partial_mappings():
    names = ["right_finger2_link4", "right_finger3_link2", "right_finger4_link4",
             "right_finger5_link4", "right_finger1_link4"]
    minima = {name: .2 for name in preload.FINGERS}; minima["middle"] = .35
    values, kind = preload._minimum_normal_forces(minima, names)
    np.testing.assert_array_equal(values, [.2, .35, .2, .2, .2])
    assert kind == "finger_mapping"
    vector, kind = preload._minimum_normal_forces([.1, .2, .3, .4, .5], names)
    np.testing.assert_array_equal(vector, [.1, .2, .3, .4, .5])
    assert kind == "contact_vector"
    with pytest.raises(ValueError, match="exactly all five fingers"):
        preload._minimum_normal_forces({"middle": .35}, names)
    with pytest.raises(ValueError, match="positive finite vector"):
        preload._minimum_normal_forces([.2, .2, -.3, .2, .2], names)


def test_per_finger_clarabel_qp_uses_actual_wrench_and_servo_bounds_without_gpu(monkeypatch):
    from retarget.config.run import ROOT
    from retarget.env.grasp import load
    source = sorted((ROOT / "output/grasp_contactaware_mppi").glob("001*/*/input_00"))
    if not source:
        pytest.skip("Accepted middle-size CA material points required")
    pytest.importorskip("clarabel")
    source = source[0]
    artifacts = SimpleNamespace(root=source, viewer_xml=source/"runtime_scene.xml")
    cfg = SimpleNamespace(sequence=SimpleNamespace(hand="wuji_right"), optimizer=SimpleNamespace(horizon=5))
    def no_cuda(*args, **kwargs):
        raise AssertionError("Mapped minima must use exact CPU source matrices, not CUDA load kinematics")
    monkeypatch.setattr(load, "load_matrices", no_cuda)
    minimum = {finger: .2 for finger in preload.FINGERS}; minimum["middle"] = .35
    command, report = preload.prepare_preload(artifacts, cfg, minimum_per_contact=minimum)
    assert command is not None and report["feasible"] is True
    assert report["backend"] == "clarabel_cpu_bounded_load_qp"
    assert report["solver_status"] == "Solved"
    assert report["maximum_scaled_constraint_violation"] < 2e-4
    assert report["forceaware_objective_modified"] is False
    assert report["static_qp_only"] is True
    inputs = preload.preload_inputs(artifacts)
    forces = np.asarray(report["force_N"])
    np.testing.assert_allclose(forces.sum(0), [0., 0., inputs["mass"]*9.81], atol=1e-8)
    np.testing.assert_allclose(np.cross(inputs["arms"], forces).sum(0), 0., atol=1e-8)
    pressure = -(inputs["outward"]*forces).sum(1)
    assert np.all(pressure >= np.asarray(report["resolved_minimum_per_contact_N"])-1e-8)
    np.testing.assert_allclose(pressure, report["normal_force_N"], atol=1e-9)
    torque = inputs["bias"]+np.einsum("cij,ci->j", inputs["jacobian"], forces)
    np.testing.assert_allclose(torque, report["actuator_torque_Nm"], atol=1e-10)
    np.testing.assert_allclose(command, inputs["qref"]+torque/inputs["model"].actuator_gainprm[:, 0], atol=1e-10)
    vector_command, vector_report = preload.prepare_preload(artifacts, cfg,
        minimum_per_contact=report["resolved_minimum_per_contact_N"])
    assert vector_report["minimum_specification"] == "contact_vector"
    np.testing.assert_allclose(vector_command, command, atol=1e-10)
    coefficients = np.asarray(report["ray_coefficients"])
    assert coefficients.min() >= 0.
    assert report["objective_value"] == pytest.approx(.005*np.sum(coefficients**2)+.01*coefficients.sum())
    # This middle-finger lower bound is inactive; naming it .35N must not be
    # mistaken for an actual increase over the previous ~2.24N static force.
    assert pressure[inputs["body_names"].index("right_finger3_link2")] > 2.
    rejected, invalid_report = preload.prepare_preload(artifacts, cfg,
        minimum_per_contact={finger: 1000. for finger in preload.FINGERS})
    assert rejected is None and invalid_report["feasible"] is False
    assert invalid_report["solver_status"] == "PrimalInfeasible"
    assert invalid_report["candidate_command_clipped"] is False
    assert invalid_report["recommended_previous_action"] is None
