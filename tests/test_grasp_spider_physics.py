"""Native dynamics/state semantics, not a synthetic claim of grasp success."""

import numpy as np
import pytest
import torch
import mujoco

pytest.importorskip("mujoco_warp")
import warp as wp

from retarget.env.grasp.spider_physics import BatchedPhysics, STATE_FIELDS, control_bounds


def source_model():
    return mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep=".002" gravity="0 0 0" integrator="implicitfast" cone="pyramidal" jacobian="dense"/>
      <worldbody><geom type="plane" size="1 1 .1"/>
        <body name="moving"><joint name="slide" type="slide" axis="0 0 1" ref=".11"/>
          <geom type="sphere" size=".1" mass="1"/></body></worldbody>
      <actuator><position joint="slide" kp="100" kv="10" ctrllimited="true" ctrlrange="0 .2"
          forcelimited="true" forcerange="-3 3"/></actuator>
    </mujoco>''')


def test_optional_control_projection_retains_unlimited_channels():
    model = source_model()
    low, high = control_bounds(model)
    np.testing.assert_array_equal(low, [0.])
    np.testing.assert_allclose(high, [.2])
    model.actuator_ctrllimited[:] = False
    low, high = control_bounds(model)
    assert np.isneginf(low).all() and np.isposinf(high).all()


@pytest.fixture
def native():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for native MJWarp dynamics")
    wp.init(); wp.set_device("cuda:0")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        physics = BatchedPhysics(source_model(), 2, "cuda:0", nconmax=8, njmax=32, nccdmax=8)
        physics.reset([.11], [0.], control=[.15])
        physics.warmup()
        yield physics


def test_graph_warmup_preserves_measured_state_and_model(native):
    original = source_model()
    for name in ("body_mass", "body_inertia", "dof_armature", "actuator_gainprm",
                 "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange", "geom_friction"):
        np.testing.assert_array_equal(getattr(native.cpu, name), getattr(original, name))
    torch.testing.assert_close(native.qpos, torch.full_like(native.qpos, .11), rtol=0., atol=0.)
    assert torch.count_nonzero(native.qvel) == 0
    assert torch.count_nonzero(wp.to_torch(native.data.time)) == 0


def test_complete_restore_prevents_previous_candidate_force_or_warmstart_leak(native):
    state = native.snapshot()
    assert set(state) == set(STATE_FIELDS)
    for _ in range(4):
        native.step([.15])
    expected_q, expected_v = native.qpos.clone(), native.qvel.clone()
    wp.to_torch(native.data.qfrc_applied).fill_(100.)
    wp.to_torch(native.data.xfrc_applied).fill_(10.)
    wp.to_torch(native.data.qacc_warmstart).fill_(50.)
    for _ in range(3):
        native.step([.2])
    native.restore(state)
    for _ in range(4):
        native.step([.15])
    torch.testing.assert_close(native.qpos, expected_q, rtol=0., atol=0.)
    torch.testing.assert_close(native.qvel, expected_v, rtol=0., atol=0.)
    assert bool(native.check()["valid"].all())
    with pytest.raises(ValueError, match="complete"):
        native.restore({"qpos": state["qpos"], "qvel": state["qvel"]})


def test_native_step_retains_raw_commands_and_uses_original_force_clamp(native):
    native.step([.5])
    torch.testing.assert_close(wp.to_torch(native.data.ctrl), torch.full((2, 1), .5, device="cuda:0"))
    torch.testing.assert_close(wp.to_torch(native.data.actuator_force), torch.full((2, 1), 3., device="cuda:0"))
    torch.testing.assert_close(native.project_ctrl(torch.tensor([.5], device="cuda:0")),
                               torch.tensor([.2], device="cuda:0"))


def test_entire_reset_and_rollout_capture_reads_live_controls_without_state_leak(native):
    q = torch.tensor([.11], device="cuda:0")
    v = torch.zeros(1, device="cuda:0")
    control = torch.full((2, 1), .15, device="cuda:0")
    def rollout():
        native.reset(q, v)
        for _ in range(4):
            native.step(control, use_graph=False)
    with wp.ScopedCapture() as capture:
        rollout()
    wp.capture_launch(capture.graph)
    expected = native.qpos.clone()
    control.fill_(.11)
    wp.capture_launch(capture.graph)
    assert not torch.equal(native.qpos, expected)
    control.fill_(.15)
    wp.capture_launch(capture.graph)
    torch.testing.assert_close(native.qpos, expected, rtol=0., atol=0.)

