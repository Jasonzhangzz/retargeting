"""Actual native pressure reaches captured MPPI cost; no grasp-success claim.

Small W=2/H=2 runs use unchanged accepted CA files. Contact loss is created by
moving the physical object's initial state, never by fabricating force inputs.
"""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco_warp")
import warp as wp

from retarget.config.run import ROOT
from retarget.env.grasp.spider_contact import required_finger_mask
from retarget.env.grasp.spider_reference import prepare_model, preload, read_input
from retarget.env.grasp.spider_rollout import SamplingRollout
from retarget.task.grasp.spider_cost import SpiderCostSettings, trajectory_cost


def test_retention_report_measures_ca_subset_with_newtons_and_normalized_deficit():
    from retarget.env.grasp.spider_runner import contact_retention_metrics
    settings = SpiderCostSettings(contact_keep_rew_scale=.2, contact_min_force_n=.05)
    required = np.asarray([True, False, True, False, False])
    forces = np.asarray([[.05, 0., .025, 0., 0.], [0., 0., .1, 0., 0.], [-1., 0., 0., 0., 0.]])
    report = contact_retention_metrics(forces, required, settings)
    assert report["available"] is True and report["finite"] is True
    assert report["physics_steps"] == 3
    assert report["all_required_positive_fraction"] == pytest.approx(1/3)
    assert report["all_required_at_target_fraction"] == 0.
    assert report["normalized_deficit_squared_mean"] == pytest.approx((.25+1.+2.)/3.)
    assert report["mean_contact_keep_cost"] == pytest.approx(.2*(.25+1.+2.)/3.)
    # Unrequired contact loss and additional recorded samples do not inflate
    # the per-state CA objective or claim all-five contact preservation.
    forces[:, ~required] = 100.
    repeated = contact_retention_metrics(np.repeat(forces, 10, axis=0), required, settings)
    assert repeated["mean_contact_keep_cost"] == pytest.approx(report["mean_contact_keep_cost"])


@pytest.mark.parametrize("forces,required", [
    (np.empty((0, 5)), np.ones(5, bool)),
    (np.zeros((2, 5)), np.zeros(5, bool)),
    (np.full((2, 5), np.nan), np.ones(5, bool)),
    (np.full((2, 5), np.inf), np.ones(5, bool)),
])
def test_retention_report_cannot_certify_missing_or_nonfinite_measurements(forces, required):
    from retarget.env.grasp.spider_runner import contact_retention_metrics
    result = contact_retention_metrics(forces, required, SpiderCostSettings(contact_keep_rew_scale=.2))
    assert result["available"] is False
    assert "all_required_at_target_fraction" not in result


def test_retention_report_disabled_cost_keeps_measured_detachment_visible():
    from retarget.env.grasp.spider_runner import contact_retention_metrics
    result = contact_retention_metrics(np.zeros((2, 5)), np.ones(5, bool), SpiderCostSettings())
    assert result["available"] is True
    assert result["mean_contact_keep_cost"] == 0.
    assert result["normalized_deficit_squared_mean"] == 5.
    assert result["all_required_positive_fraction"] == 0.


def _clone(state):
    return {name: value.clone() for name, value in state.items()}


@contextmanager
def _caller(bundle):
    with torch.cuda.stream(bundle.stream), wp.ScopedStream(wp.stream_from_torch(bundle.stream)), torch.no_grad():
        yield


@pytest.fixture(scope="module")
def rollout_factory(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for captured native pressure/cost integration")
    source = ROOT/"output/grasp_contactaware_feature_normals_final/001_radius_0.03500/1_Large_Diameter/input_00"
    if not (source/"card_input.npz").is_file():
        pytest.skip("Saved accepted middle-size CA reference is required")
    reference = read_input(source)
    destination = tmp_path_factory.mktemp("spider_contact_keep_native")
    cpu, model_path = prepare_model(reference, destination, .01)
    warm, report = preload(reference, model_path, 2)
    assert report["feasible"] is True
    wp.init(); wp.set_device("cuda:0")
    cache = {}

    def build(caller="default", *, enabled=True, thumb_only=False):
        key = caller, enabled, thumb_only
        if key in cache:
            return cache[key]
        subset = reference
        if thumb_only:
            # Duplicate the first accepted finger slot in a private in-memory
            # view. Both slots require one finger; source CA files stay intact.
            subset = replace(reference,
                contact_bodies=reference.contact_bodies[[0, 0]].copy(),
                contact_local=reference.contact_local[[0, 0]].copy(),
                object_local=reference.object_local[[0, 0]].copy(),
                active_slots=reference.active_slots[[0, 0]].copy())
        stream = torch.cuda.default_stream() if caller == "default" else torch.cuda.Stream()
        bundle = SimpleNamespace(stream=stream, reference=subset)
        settings = SpiderCostSettings(contact_rew_scale=1., terminal_rew_scale=1.7,
            contact_keep_rew_scale=.2 if enabled else 0., contact_min_force_n=.05)
        with _caller(bundle):
            bundle.rollout = SamplingRollout(cpu, subset, warm, worlds=2, horizon_steps=2,
                cost_settings=settings, device="cuda:0", nconmax=256, njmax=2048)
            bundle.rollout.physics.reset(subset.qpos, subset.qvel, control=warm)
            bundle.initial = bundle.rollout.physics.snapshot()
            bundle.controls = torch.as_tensor(warm, device="cuda:0").expand(2, 2, -1).clone()
        cache[key] = bundle
        return bundle

    return build


@pytest.fixture(scope="module", params=("default", "nondefault"))
def real_contact_rollout(rollout_factory, request):
    return rollout_factory(request.param)


def _evaluate(bundle, state=None, controls=None):
    rollout = bundle.rollout
    with _caller(bundle):
        rollout.prepare(_clone(bundle.initial if state is None else state))
        costs = rollout.evaluate(bundle.controls if controls is None else controls).clone()
        return dict(cost=costs, qpos=rollout.qtrace.clone(), qvel=rollout.vtrace.clone(),
            points=rollout.ctrace.clone(), force=None if rollout.ftrace is None else rollout.ftrace.clone(),
            parts={name: value.clone() for name, value in rollout.parts.items()})


def _detached_state(bundle):
    with _caller(bundle):
        result = _clone(bundle.initial)
        result["qpos"][:, 26] += 1.
        return result


def _expected_shortfall(rollout, forces):
    settings = rollout.settings
    # Express the same physical rule as missing Newtons divided by threshold,
    # independently from the implementation's 1-Fn/Fmin arrangement.
    missing_fraction = (settings.contact_min_force_n-forces.clamp_min(0.)).clamp_min(0.)/settings.contact_min_force_n
    step = (missing_fraction.square()*rollout.contact_loads.required_fingers).sum(-1)
    weights = forces.new_tensor([1., settings.terminal_rew_scale])
    return settings.contact_keep_rew_scale*(step*weights).mean(1)


def test_actual_native_forces_are_copied_at_each_solver_stage(real_contact_rollout):
    bundle, rollout = real_contact_rollout, real_contact_rollout.rollout
    captured = _evaluate(bundle)
    manual = []
    with _caller(bundle):
        rollout.physics.restore(_clone(bundle.initial))
        for step in range(2):
            rollout.physics.step(bundle.controls[:, step], use_graph=False)
            manual.append(rollout.contact_loads.sample().clone())
        manual = torch.stack(manual, 1)
        assert bool(torch.isfinite(captured["force"]).all())
        assert bool((captured["force"] > .05).any())
        torch.testing.assert_close(captured["force"], manual, rtol=8e-4, atol=3e-4)
        torch.testing.assert_close(captured["parts"]["contact_keep"],
            _expected_shortfall(rollout, captured["force"]), rtol=2e-5, atol=2e-6)
        assert captured["force"].data_ptr() != rollout.contact_loads.sample().data_ptr()


def test_detached_physical_object_has_zero_load_and_penalizes_required_ca_fingers(real_contact_rollout):
    bundle, rollout = real_contact_rollout, real_contact_rollout.rollout
    loaded = _evaluate(bundle)
    detached = _evaluate(bundle, _detached_state(bundle))
    with _caller(bundle):
        assert bool(torch.isfinite(detached["cost"]).all())
        assert torch.count_nonzero(detached["force"]) == 0
        required = int(rollout.contact_loads.required_fingers.sum())
        assert required == 5
        expected = rollout.settings.contact_keep_rew_scale*required*(1.+rollout.settings.terminal_rew_scale)/2.
        torch.testing.assert_close(detached["parts"]["contact_keep"],
            torch.full_like(detached["cost"], expected), rtol=0., atol=2e-7)
        assert bool((detached["parts"]["contact_keep"] > loaded["parts"]["contact_keep"]).all())
        # Configuration/contact-point tracking also changes when an object is
        # physically translated. Verify the added term separately in total.
        without_keep = trajectory_cost(detached["qpos"], detached["qvel"], rollout.qref, rollout.vref,
            contact_positions=detached["points"], contact_target_world=rollout.targets,
            contact_mask=rollout.mask, settings=replace(rollout.settings, contact_keep_rew_scale=0.))
        torch.testing.assert_close(detached["cost"],
            without_keep["total"]+detached["parts"]["contact_keep"], rtol=2e-6, atol=2e-6)


def test_force_trace_and_cost_replay_without_previous_candidate_leak(real_contact_rollout):
    first = _evaluate(real_contact_rollout)
    detached = _evaluate(real_contact_rollout, _detached_state(real_contact_rollout))
    again = _evaluate(real_contact_rollout)
    assert torch.count_nonzero(detached["force"]) == 0
    with _caller(real_contact_rollout):
        torch.testing.assert_close(again["force"], first["force"], rtol=8e-4, atol=3e-4)
        torch.testing.assert_close(again["cost"], first["cost"], rtol=2e-4, atol=2e-6)


def test_ca_subset_deduplicates_required_finger_and_ignores_other_finger_losses(rollout_factory):
    bundle = rollout_factory(thumb_only=True)
    rollout = bundle.rollout
    expected_mask = required_finger_mask(rollout.physics.cpu, bundle.reference.contact_bodies)
    assert expected_mask.sum() == 1
    with _caller(bundle):
        np.testing.assert_array_equal(rollout.contact_loads.required_fingers.cpu().numpy(), expected_mask)
    loaded = _evaluate(bundle)
    detached = _evaluate(bundle, _detached_state(bundle))
    with _caller(bundle):
        # Collector still observes actual loads for all physical fingers; only
        # accepted CA finger membership selects the detach objective.
        assert bool((loaded["force"][..., ~rollout.contact_loads.required_fingers] > .05).any())
        expected = rollout.settings.contact_keep_rew_scale*(1.+rollout.settings.terminal_rew_scale)/2.
        torch.testing.assert_close(detached["parts"]["contact_keep"],
            torch.full_like(detached["cost"], expected), rtol=0., atol=1e-7)
        torch.testing.assert_close(loaded["parts"]["contact_keep"],
            _expected_shortfall(rollout, loaded["force"]), rtol=2e-5, atol=2e-6)


def test_disabled_extension_omits_force_collection_and_preserves_original_cost(rollout_factory):
    bundle = rollout_factory(enabled=False)
    rollout = bundle.rollout
    actual = _evaluate(bundle)
    assert rollout.contact_loads is None and rollout.ftrace is None
    assert actual["force"] is None
    with _caller(bundle):
        expected = trajectory_cost(actual["qpos"], actual["qvel"], rollout.qref, rollout.vref,
            contact_positions=actual["points"], contact_target_world=rollout.targets,
            contact_mask=rollout.mask, settings=rollout.settings)
        torch.testing.assert_close(actual["cost"], expected["total"], rtol=2e-6, atol=2e-7)
        assert torch.count_nonzero(actual["parts"]["contact_keep"]) == 0


def test_contact_keep_evaluation_preserves_ca_source_bytes_and_reference(real_contact_rollout):
    bundle = real_contact_rollout
    before = bundle.reference.qpos.copy(), bundle.reference.contact_local.copy(), bundle.reference.object_local.copy()
    _evaluate(bundle)
    _evaluate(bundle, _detached_state(bundle))
    for filename, expected in bundle.reference.source_hashes.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected
    for actual, expected in zip((bundle.reference.qpos, bundle.reference.contact_local, bundle.reference.object_local), before):
        np.testing.assert_array_equal(actual, expected)
