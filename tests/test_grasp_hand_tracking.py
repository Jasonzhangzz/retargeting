"""Explicit ContactAware hand references and the fifth trajectory cost term."""

from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh
import warp as wp

from retarget.task.dex import cost as dex
from retarget.task.grasp.tracking_cost import (
    PART_NAMES, TrackingObjective, aggregate, configuration, load_hand_reference,
    load_reference_contact_mask, score_hand_tracking, score_state, synchronize_ranking_parts,
)


def _array(value, dtype=float, device="cpu"):
    return wp.array(np.asarray(value), dtype=dtype, device=device)


def _hand_job(q_reference, q_current=None, device="cpu"):
    wp.init()
    model, target, state, weights = dex.LossModel(), dex.LossTarget(), dex.LossState(), dex.LossWeights()
    model.action_dim = 26
    # Deliberately avoid first-26 indexing: the scene index owns the mapping.
    model.hand_qpos = _array(np.arange(7, 33), int, device)
    target.action = _array(q_reference, device=device)
    qpos = np.zeros((1, 33), np.float32)
    qpos[0, 7:] = q_reference if q_current is None else q_current
    state.qpos = _array(qpos, device=device)
    weights.action_base_position, weights.action_base_rotation, weights.action_finger = 10.0, 3.0, 0.03
    job = dex.LossJob()
    job.model, job.target, job.state, job.weights = model, target, state, weights
    return job


def _hand_value(job):
    device = job.state.qpos.device
    parts = wp.full((1, 1, 5), 17.0, device=device)
    wp.launch(score_hand_tracking, 1, inputs=[job, 0, parts], device=device)
    values = parts.numpy()[0, 0]
    np.testing.assert_array_equal(values[:4], np.full(4, 17.0))
    return float(values[4])


@pytest.mark.parametrize("coordinate,delta,weight", [(0, 0.002, 10), (3, 0.1, 3), (6, 0.2, 0.03), (25, -0.15, 0.03)])
def test_solved_hand_reference_and_scene_coordinates_determine_cost(coordinate, delta, weight):
    reference = np.linspace(-0.3, 0.4, 26, dtype=np.float32)
    job = _hand_job(reference)
    assert _hand_value(job) == 0
    current = reference.copy()
    current[coordinate] += delta
    job.state.qpos.assign(np.r_[np.zeros(7), current][None])
    actual_delta = float(current[coordinate] - reference[coordinate])
    assert _hand_value(job) == pytest.approx(1e6 * (weight * actual_delta) ** 2, rel=2e-6)
    # Updating the solved target independently updates only the fifth term.
    job.target.action.assign(current)
    assert _hand_value(job) == 0


def test_hand_cost_cuda_matches_cpu():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    reference = np.linspace(-0.3, 0.4, 26, dtype=np.float32)
    current = reference + np.r_[np.full(3, .001), np.full(3, .02), np.full(20, .1)].astype(np.float32)
    assert _hand_value(_hand_job(reference, current, "cuda:0")) == pytest.approx(
        _hand_value(_hand_job(reference, current)), rel=1e-6
    )


def test_hand_tracking_is_opt_in_even_with_legacy_prepared_action_weights():
    cfg = SimpleNamespace(loss=SimpleNamespace(action_ref_base=10, action_ref_base_rot=3, action_ref_finger=.03))
    disabled, _ = configuration(SimpleNamespace(), cfg)
    assert [disabled[k] for k in ("action_ref_base", "action_ref_base_rot", "action_ref_finger")] == [0, 0, 0]
    enabled, _ = configuration(SimpleNamespace(tracking_weights={"action_ref_finger": .03}), cfg)
    assert enabled["action_ref_finger"] == .03 and enabled["action_ref_base"] == 0


def test_reference_loader_does_not_fall_back_to_template_or_runtime_seed(tmp_path):
    reference = np.linspace(-.2, .3, 26)
    seed = np.full(26, .8)
    np.savez(tmp_path / "card_input.npz", hand_reference_qpos=reference,
        hand_reference_source="dexforge_contactaware", seed_qpos=seed)
    np.testing.assert_allclose(load_hand_reference(tmp_path, 26, True), reference)
    np.savez(tmp_path / "card_input.npz", seed_qpos=seed, hand_reference_qpos=reference,
        hand_reference_source="template")
    with pytest.raises(ValueError, match="originate"):
        load_hand_reference(tmp_path, 26, True)
    np.savez(tmp_path / "card_input.npz", seed_qpos=seed)
    with pytest.raises(ValueError, match="explicit"):
        load_hand_reference(tmp_path, 26, True)
    assert load_hand_reference(tmp_path / "absent", 26, False) is None


@pytest.mark.parametrize("reference", [np.zeros(20), np.full(26, np.nan), np.zeros((1, 26))])
def test_reference_loader_rejects_bad_shape_or_nonfinite_values(tmp_path, reference):
    np.savez(tmp_path / "card_input.npz", hand_reference_qpos=reference, hand_reference_source="dexforge_contactaware")
    with pytest.raises(ValueError, match="26 finite"):
        load_hand_reference(tmp_path, 26, True)


@pytest.mark.parametrize("mask", [np.zeros(2), np.ones(1), [1, .5], [1, np.nan], [-1, 1]])
def test_reference_contact_mask_rejects_released_all_or_invalid_slots(tmp_path, mask):
    np.savez(tmp_path / "card_input.npz", reference_contact_mask=mask)
    with pytest.raises(ValueError, match="binary value"):
        load_reference_contact_mask(tmp_path, 2)


def test_reference_contact_mask_defaults_and_explicit_releases(tmp_path):
    np.savez(tmp_path / "card_input.npz", object_points=np.zeros((2, 3)))
    np.testing.assert_array_equal(load_reference_contact_mask(tmp_path, 2), [1, 1])
    np.savez(tmp_path / "card_input.npz", reference_contact_mask=[1, 0])
    np.testing.assert_array_equal(load_reference_contact_mask(tmp_path, 2), [1, 0])


def test_released_contact_does_not_enter_contact_cost_or_contact_metrics():
    wp.init()
    model, target, state, weights, auxiliary = dex.LossModel(), dex.LossTarget(), dex.LossState(), dex.LossWeights(), dex.LossAuxiliary()
    model.object_body = model.object_qpos = model.object_dof = 0
    model.guidance_count = model.output_count = 2
    model.output_contact_rows, model.contact_thresholds = _array([0, 1], int), _array([.01, .01])
    target.query_body, target.query_local = _array([1, 2], int), _array(np.zeros((2, 3)), wp.vec3)
    target.query_position_object = _array(np.zeros((2, 3)), wp.vec3)
    target.contact_output = _array([0, 1], int)
    target.contact_mask = target.query_mask = _array([1, 0])
    target.contact_weight = target.query_weight = _array([1, 1])
    target.object_qpos, target.object_velocity_world = _array([0, 0, 0, 1, 0, 0, 0]), _array(np.zeros(6))
    state.body_position = _array([[[0, 0, 0], [.001, 0, 0], [.1, 0, 0]]], wp.vec3)
    state.body_matrix = _array(np.tile(np.eye(3), (1, 3, 1, 1)), wp.mat33)
    state.qpos, state.qvel = _array([[0, 0, 0, 1, 0, 0, 0]]), _array(np.zeros((1, 6)))
    state.contact_distance, state.contact_active = _array([[0, .1]]), _array([[1, 1]], int)
    state.contact_position = _array(np.zeros((1, 2, 3)), wp.vec3)
    auxiliary.marked_query_weight = wp.zeros((1, 2), device="cpu")
    weights.marked_query, weights.marked_distance_mm, weights.marked_temperature_mm = 1, 10, 1
    weights.contact_robustness, weights.target_phi = 40, 0
    weights.soft_penetration, weights.penetration = 15, 75
    weights.soft_penetration_limit, weights.penetration_limit = .002, .003
    job = dex.LossJob()
    job.model, job.target, job.state, job.weights, job.auxiliary = model, target, state, weights, auxiliary
    parts, metrics = wp.zeros((1, 1, 4), device="cpu"), wp.zeros((1, 1, 5), device="cpu")
    wp.launch(dex._prepare_auxiliary, (1, 2), inputs=[job], device="cpu")
    wp.launch(score_state, 1, inputs=[job, 0, parts, metrics], device="cpu")
    assert parts.numpy()[0, 0, 2] == 0
    assert parts.numpy()[0, 0, 1] < .001
    np.testing.assert_allclose(metrics.numpy()[0, 0], [0, 1, 1, 0, 0], atol=1e-5)
    target.contact_mask.assign([1, 1])
    wp.launch(score_state, 1, inputs=[job, 0, parts, metrics], device="cpu")
    assert parts.numpy()[0, 0, 2] == pytest.approx(40 * 100**2, rel=1e-6)
    np.testing.assert_allclose(metrics.numpy()[0, 0, [2, 4]], [100, 100], atol=1e-5)


def test_trajectory_aggregation_has_a_fifth_term_and_preserves_nonfinite_rejection():
    wp.init()
    values = np.arange(15, dtype=np.float32).reshape(1, 3, 5)
    steps, parts, cost = _array(values), wp.zeros((1, 5), device="cpu"), wp.zeros(1, device="cpu")
    wp.launch(aggregate, 1, inputs=[steps, 2.0, parts, cost], device="cpu")
    expected = np.average(values[0], axis=0, weights=[1, 1, 2])
    np.testing.assert_allclose(parts.numpy()[0], expected)
    assert cost.numpy()[0] == expected.sum()
    legacy = wp.zeros((1, 4), device="cpu")
    wp.launch(synchronize_ranking_parts, 1, inputs=[parts, cost, legacy], device="cpu")
    np.testing.assert_array_equal(legacy.numpy()[0], expected[:4])
    cost.assign([float("nan")])
    wp.launch(synchronize_ranking_parts, 1, inputs=[parts, cost, legacy], device="cpu")
    assert np.isinf(legacy.numpy()[0, 0])


def _objective_node(experiment, enabled):
    array = lambda x, dtype=float: _array(x, dtype)
    hand = SimpleNamespace(bodies=np.array([1]), local=np.array([[0., 0., 0.]]))
    args = SimpleNamespace(tracking_weights={"action_ref_finger": .03} if enabled else {})
    model = SimpleNamespace(jnt_qposadr=np.arange(27), jnt_dofadr=np.arange(27), nv=32)
    index = SimpleNamespace(object_body=0, object_qpos=26, hand_qpos=np.arange(26))
    runtime = SimpleNamespace(args=args, cfg=None, experiment=experiment, model=model, index=index,
        qpos=np.r_[np.full(26, .8), [0, 0, 0, 1, 0, 0, 0]], audit_context=(None, hand, args))
    cost = SimpleNamespace(sample_body=array([1], int), parts=wp.zeros((1, 4), device="cpu"))
    backend = SimpleNamespace(kinematics=SimpleNamespace(allocate=lambda _: None))
    return SimpleNamespace(runtime=runtime, cost=cost, state=SimpleNamespace(population=1),
        explicit=SimpleNamespace(horizon=2), device="cpu", array=array, backend=backend)


def test_objective_uses_explicit_ca_target_and_allocates_independent_dynamic_parts(tmp_path):
    wp.init()
    q_reference = np.linspace(-.1, .2, 26)
    np.savez(tmp_path / "card_input.npz", object_points=np.zeros((1, 3)),
        hand_reference_qpos=q_reference, hand_reference_source="dexforge_contactaware")
    node = _objective_node(tmp_path, True)
    objective = TrackingObjective(node)
    assert objective.part_names == PART_NAMES + ("hand_tracking",)
    assert objective.parts.shape == (1, 5) and objective.step_parts.shape == (1, 3, 5)
    assert objective.parts is not node.cost.parts
    np.testing.assert_allclose(objective.job.target.action.numpy(), q_reference)
    assert not np.allclose(objective.job.target.action.numpy(), node.runtime.qpos[:26])
    np.testing.assert_array_equal(objective.job.model.hand_qpos.numpy(), np.arange(26))
    assert objective.metadata["hand_reference_source"] == "dexforge_contactaware"
    disabled = TrackingObjective(_objective_node(tmp_path, False))
    assert disabled.part_names == PART_NAMES and disabled.parts.shape == (1, 4)


def test_experiment_persists_ca_reference_separately_from_initial_hand(tmp_path, monkeypatch):
    from retarget.env.grasp import hand as hand_module
    from retarget.env.grasp.runtime import prepare_experiment

    monkeypatch.setattr(hand_module, "model_hand", lambda _: "wuji2")
    monkeypatch.setattr(hand_module, "hand_tree", lambda _: ET.fromstring("<mujoco><option/><asset/><worldbody/></mujoco>"))
    config = tmp_path / "model.yaml"
    config.write_text("sequence: {}\n")
    mesh = trimesh.creation.box(extents=[.07, .07, .12])
    hand = SimpleNamespace(model=object(), surface=SimpleNamespace(triangles=mesh.triangles),
        bodies=np.array([1]), local=np.zeros((1, 3)))
    initial, reference = np.full(26, .8), np.linspace(-.1, .2, 26)
    guess = SimpleNamespace(hand_qpos=initial, contactaware_reference_qpos=reference,
        object_points=np.array([[.035, 0, 0]]), object_normals=np.array([[1, 0, 0]]),
        metadata={"contactaware": {"success": True, "method": "DexForge contact optimization and IK"}})
    args = SimpleNamespace(solver_config=config, object_mass=.2, friction=1., physics_reference=None)
    experiment = prepare_experiment(tmp_path / "experiment", hand, guess, args)
    with np.load(experiment / "card_input.npz", allow_pickle=False) as supplied:
        np.testing.assert_array_equal(supplied["hand_initial_qpos"], initial)
        np.testing.assert_array_equal(supplied["hand_reference_qpos"], reference)
        assert supplied["hand_reference_source"].item() == "dexforge_contactaware"
    trajectory = np.load(experiment / "contactaware/wuji_right_qpos.npy")
    np.testing.assert_allclose(trajectory, np.tile(reference, (2, 1)))
    np.testing.assert_allclose(load_hand_reference(experiment, 26, True), reference)
