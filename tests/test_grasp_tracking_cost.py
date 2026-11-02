"""Formula checks for the four-term static-grasp tracking profile."""

from types import SimpleNamespace
import math
import numpy as np
import pytest
import warp as wp

from retarget.task.dex import cost as dex
from retarget.task.grasp.tracking_cost import (
    aggregate, configuration, finish_tracking, map_velocity, score_state,
)
from retarget.task.grasp.cost import CostData, SearchState


def make_job(device="cpu", *, position_error=0.0, angle=0.0, gap=0.0):
    wp.init()
    arr = lambda value, dtype=float: wp.array(np.asarray(value), dtype=dtype, device=device)
    model = dex.LossModel()
    model.object_body = model.object_qpos = model.object_dof = 0
    model.guidance_count = model.output_count = 1
    model.output_contact_rows = arr([0], int)
    model.contact_thresholds = arr([0.01])
    target = dex.LossTarget()
    target.query_body = arr([1], int)
    target.query_local = arr([[0.0, 0.0, 0.0]], wp.vec3)
    target.query_position_object = arr([[0.0, 0.0, 0.0]], wp.vec3)
    target.contact_output = arr([0], int)
    target.query_mask = target.query_weight = arr([1.0])
    target.contact_mask = target.contact_weight = arr([1.0])
    target.object_qpos = arr([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    target.object_velocity_world = arr(np.zeros(6))
    weights = dex.LossWeights()
    for key, value in dict(reference_pose_scale=100.0, object_position=1.0,
            object_rotation=0.3, object_velocity=0.1, object_angular_velocity=0.03,
            contact_robustness=40.0, marked_query=1.0, soft_penetration=15.0,
            penetration=75.0, target_phi=-0.001, soft_penetration_limit=0.002,
            penetration_limit=0.003, marked_distance_mm=10.0, marked_temperature_mm=1.0).items():
        setattr(weights, key, value)
    state = dex.LossState()
    state.qpos = arr([[position_error, 0, 0, math.cos(angle / 2), 0, 0, math.sin(angle / 2)]])
    state.qvel = arr(np.zeros((1, 6)))
    state.body_position = arr([[[position_error, 0, 0], [position_error + 0.02, 0, 0]]], wp.vec3)
    state.body_matrix = arr(np.tile(np.eye(3), (1, 2, 1, 1)), wp.mat33)
    state.contact_distance = arr([[gap]])
    state.contact_active = arr([[1]], int)
    state.contact_position = arr([[[0, 0, 0]]], wp.vec3)
    aux = dex.LossAuxiliary()
    aux.marked_query_weight = wp.zeros((1, 1), device=device)
    job = dex.LossJob()
    job.model, job.weights, job.target, job.state, job.auxiliary = model, weights, target, state, aux
    return job


def evaluate(job):
    device = job.state.qpos.device
    parts = wp.zeros((1, 1, 4), device=device)
    metrics = wp.zeros((1, 1, 5), device=device)
    wp.launch(dex._prepare_auxiliary, (1, 1), inputs=[job], device=device)
    wp.launch(score_state, 1, inputs=[job, 0, parts, metrics], device=device)
    return parts.numpy()[0, 0], metrics.numpy()[0, 0]


def test_four_terms_match_independent_dex_formulas():
    job = make_job(position_error=0.01, angle=0.2, gap=-0.004)
    parts, metrics = evaluate(job)
    expected = [15 * 2**2 + 75 * 1**2, 20**2 / (1 + math.exp(-10)), 0,
                1e6 * 100 * (0.01**2 + (0.3 * 0.2)**2)]
    np.testing.assert_allclose(parts, expected, rtol=2e-6, atol=1e-4)
    np.testing.assert_allclose(metrics, [4, 20, 20, -4, -4], atol=1e-5)


def test_maintain_penalizes_separation_beyond_activation_distance():
    job = make_job(gap=0.02)
    parts, metrics = evaluate(job)
    assert parts[2] == pytest.approx(40 * 21**2, rel=1e-6)
    assert metrics[4] == pytest.approx(20)


def test_fixed_template_point_and_stationary_reference():
    job = make_job()
    before, _ = evaluate(job)
    # Only move the object away from its fixed world reference. The local
    # anchor follows the object, so common hand/object translation preserves
    # approach, while object tracking becomes strictly positive.
    job.state.qpos.assign([[0.01, 0, 0, 1, 0, 0, 0]])
    job.state.body_position.assign([[[0.01, 0, 0], [0.03, 0, 0]]])
    moved, _ = evaluate(job)
    assert moved[1] == pytest.approx(before[1], rel=1e-6)
    assert moved[3] > before[3] + 9000
    # Template local coordinates are real targets, not a candidate-selected
    # nearest surface point. Moving the template point changes approach.
    job.target.query_local.assign([[0.01, 0, 0]])
    changed, _ = evaluate(job)
    assert changed[1] > moved[1]


def test_quaternion_sign_and_world_angular_velocity_frame():
    job = make_job(angle=math.pi / 2)
    job.weights.object_position = job.weights.object_rotation = 0
    job.weights.object_velocity = 0
    # Explicit angular velocity is in object coordinates. Under a +90deg z
    # rotation body +x becomes world +y and exactly matches this reference.
    job.target.object_velocity_world.assign([0, 0, 0, 0, 1, 0])
    job.state.qvel.assign([[0, 0, 0, 1, 0, 0]])
    parts, _ = evaluate(job)
    assert parts[3] < 1e-6
    q = job.state.qpos.numpy()
    q[:, 3:] *= -1
    job.state.qpos.assign(q)
    signed, _ = evaluate(job)
    np.testing.assert_allclose(signed, parts, atol=1e-6)
    job.target.object_velocity_world.assign([0, 0, 0, 1, 0, 0])
    wrong, _ = evaluate(job)
    assert wrong[3] == pytest.approx(1e6 * 100 * 0.03**2 * 2, rel=1e-6)


def test_aggregation_includes_q0_and_terminal_weight():
    wp.init()
    steps = wp.array(np.arange(12).reshape(1, 3, 4), dtype=float, device="cpu")
    parts, cost = wp.zeros((1, 4), device="cpu"), wp.zeros(1, device="cpu")
    wp.launch(aggregate, 1, inputs=[steps, 2.0, parts, cost], device="cpu")
    expected = (np.arange(4) + np.arange(4, 8) + 2 * np.arange(8, 12)) / 4
    np.testing.assert_allclose(parts.numpy()[0], expected)
    assert cost.numpy()[0] == expected.sum()


def test_explicit_velocity_mapping_preserves_object_local_omega():
    wp.init()
    velocity = wp.array(np.arange(32).reshape(1, 32), dtype=float, device="cpu")
    hand_dof = wp.array(np.arange(26), dtype=int, device="cpu")
    out = wp.zeros((1, 32), device="cpu")
    wp.launch(map_velocity, 1, inputs=[velocity, hand_dof, 26, out], device="cpu")
    np.testing.assert_array_equal(out.numpy()[0], np.r_[np.arange(6, 32), np.arange(6)])


def test_configuration_uses_prepared_defaults_and_explicit_overrides():
    cfg = SimpleNamespace(loss=SimpleNamespace(obj_pos=2.0), contact=SimpleNamespace(target_phi=-0.002))
    weights, contact = configuration(SimpleNamespace(tracking_weights={"obj_rot": 0.4},
        tracking_contact={"soft_penetration_limit": 0.003}), cfg)
    assert weights["obj_pos"] == 2
    assert weights["obj_rot"] == 0.4
    assert contact["target_phi"] == -0.002
    assert contact["soft_penetration_limit"] == 0.003
    for name, value in [("tracking_weights", {"obj_pos": -1}),
                        ("tracking_contact", {"target_phi": 1}),
                        ("tracking_contact", {"marked_query_temp_mm": 0}),
                        ("tracking_weights", {"invented": 1})]:
        with pytest.raises(ValueError):
            configuration(SimpleNamespace(**{name: value}))


def test_tracking_finish_retains_initial_geometry_constraints_only():
    wp.init()
    d, s = CostData(), SearchState()
    d.penetration = wp.array([2.0, 3.01, 0.0], dtype=float, device="cpu")
    d.palm_gap = wp.array([7.0, 7.0, 5.99], dtype=float, device="cpu")
    d.self_gap = wp.array([1.0, 1.0, 1.0], dtype=float, device="cpu")
    d.self_clearance = 0.5
    # Deliberately large legacy costs must not leak into this profile.
    d.parts = wp.full((3, 4), 1e9, device="cpu")
    d.pose_cost = wp.full(3, 1e9, device="cpu")
    d.wrist_prior_cost = wp.full(3, 1e9, device="cpu")
    s.scores = wp.empty(3, device="cpu")
    total = wp.array([7.0, 7.0, 7.0], dtype=float, device="cpu")
    wp.launch(finish_tracking, 3, inputs=[d, s, total], device="cpu")
    actual = s.scores.numpy()
    assert actual[0] == 7
    assert np.isinf(actual[1:]).all()


def test_cuda_loss_formulas_match_cpu():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    cpu = evaluate(make_job(position_error=0.01, angle=0.2, gap=-0.004))
    gpu = evaluate(make_job("cuda:0", position_error=0.01, angle=0.2, gap=-0.004))
    for actual, expected in zip(gpu, cpu):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-4)
