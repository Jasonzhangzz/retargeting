"""Actual accepted WUJI CA references through the captured native MPPI rollout.

Small W=2/H=2 tests exercise staging and live-state restoration, not grasp
success or a performance benchmark. Source ContactAware files stay immutable.
"""

from pathlib import Path
import hashlib

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco_warp")
import warp as wp

from retarget.config.run import ROOT
from retarget.env.grasp.spider_reference import read_input, prepare_model, preload
from retarget.env.grasp.spider_rollout import SamplingRollout
from retarget.task.grasp.spider_cost import SpiderCostSettings


def _clone(state):
    return {name: value.clone() for name, value in state.items()}


@pytest.fixture(scope="module", params=("default", "nondefault"))
def real_rollout(tmp_path_factory, request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for a real WUJI native rollout")
    source = ROOT / "output/grasp_contactaware_feature_normals_final/001_radius_0.03500/1_Large_Diameter/input_00"
    if not (source / "card_input.npz").is_file():
        pytest.skip("Accepted middle-size ContactAware material-point reference is required")
    reference = read_input(source)
    assert len(reference.active_slots) == 5
    destination = tmp_path_factory.mktemp("wuji_native_sampling")
    cpu, model_path = prepare_model(reference, destination, .01)
    warm, report = preload(reference, model_path, 2)
    assert report["feasible"] is True
    wp.init(); wp.set_device("cuda:0")
    # Both the normal runner and a nondefault caller must interoperate with
    # SamplingRollout's own capture stream; event sync must precede capture.
    stream = torch.cuda.current_stream() if request.param == "default" else torch.cuda.Stream()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)), torch.no_grad():
        rollout = SamplingRollout(cpu, reference, warm, worlds=2, horizon_steps=2,
            cost_settings=SpiderCostSettings(contact_rew_scale=1.), nconmax=256, njmax=2048)
        state = rollout.physics.snapshot()
        # Constructor capture buffers may contain an evaluated horizon; reset to
        # the immutable CA initial state before saving the shared test snapshot.
        rollout.physics.reset(reference.qpos, reference.qvel, control=warm)
        state = rollout.physics.snapshot()
        controls = torch.as_tensor(warm, device="cuda:0").expand(2, 2, -1).clone()
        yield rollout, reference, state, controls


def _evaluate(bundle, state=None, controls=None):
    rollout, _, initial, baseline = bundle
    rollout.prepare(_clone(initial if state is None else state))
    result = rollout.evaluate(baseline if controls is None else controls).clone()
    return result, rollout.qtrace.clone(), rollout.vtrace.clone(), rollout.ctrace.clone()


def test_real_captured_rollout_restarts_each_candidate_and_repeats(real_rollout):
    first = _evaluate(real_rollout)
    second = _evaluate(real_rollout)
    assert bool(torch.isfinite(first[0]).all())
    torch.testing.assert_close(second[0], first[0], rtol=2e-4, atol=1e-6)
    torch.testing.assert_close(second[1], first[1], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(second[2], first[2], rtol=5e-4, atol=2e-4)
    assert bool(real_rollout[0].valid.all())


def test_new_initial_qpos_is_read_instead_of_a_stale_captured_pointer(real_rollout):
    _, _, initial, _ = real_rollout
    first = _evaluate(real_rollout)
    moved = _clone(initial)
    # Translate the complete hand and object in X. First-step pre-integration
    # material-point geometry must read the new measured pose before controls
    # have advanced the state, independently of later contact dynamics.
    moved["qpos"][:, 0] += .005
    moved["qpos"][:, 26] += .005
    second = _evaluate(real_rollout, moved)
    expected = torch.zeros_like(second[3][:, 0]); expected[..., 0] = .005
    torch.testing.assert_close(second[3][:, 0]-first[3][:, 0], expected, rtol=0., atol=3e-7)
    assert not torch.allclose(second[0], first[0], rtol=1e-3, atol=1e-5)


def test_new_initial_velocity_changes_the_actual_native_prediction(real_rollout):
    _, _, initial, _ = real_rollout
    first = _evaluate(real_rollout)
    moving = _clone(initial)
    moving["qvel"][:, 26] += .2
    second = _evaluate(real_rollout, moving)
    assert bool(torch.isfinite(second[0]).all())
    assert float((second[2]-first[2]).abs().max()) > 1e-3
    assert float((second[1]-first[1]).abs().max()) > 1e-5


def test_uncaptured_and_captured_real_rollouts_have_the_same_cost_and_staging(real_rollout):
    rollout, _, initial, controls = real_rollout
    captured = _evaluate(real_rollout)
    saved_cost, saved_parts = rollout.cost, rollout.parts
    try:
        rollout.physics.restore(_clone(initial))
        rollout.controls.copy_(controls)
        rollout._run()
        torch.testing.assert_close(rollout.cost, captured[0], rtol=2e-4, atol=1e-6)
        torch.testing.assert_close(rollout.qtrace, captured[1], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(rollout.ctrace, captured[3], rtol=2e-5, atol=2e-6)
    finally:
        # _run allocates fresh Torch output tensors outside capture. Restore the
        # captured output references so the production evaluate API stays valid.
        rollout.cost, rollout.parts = saved_cost, saved_parts


def test_contactaware_reference_inputs_remain_byte_identical(real_rollout):
    _, reference, _, _ = real_rollout
    _evaluate(real_rollout)
    for filename, expected in reference.source_hashes.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected
    np.testing.assert_array_equal(np.load(reference.source / "contactaware/wuji_right_qpos.npy")[0],
                                  reference.qpos[:26].astype(np.float32))

