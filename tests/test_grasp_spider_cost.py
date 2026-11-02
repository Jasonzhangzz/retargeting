"""SPIDER reward parity on real tensor layouts; no simulated hold claims."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from retarget.task.grasp.spider_cost import (
    SpiderCostSettings,
    make_hand_scale,
    quaternion_difference,
    step_cost,
    trajectory_cost,
)


def states(batch=2, horizon=None, *, dtype=torch.float32, device="cpu"):
    prefix = (batch,) if horizon is None else (batch, horizon)
    q = torch.zeros(prefix+(33,), dtype=dtype, device=device)
    q[..., 29] = 1.
    return q, torch.zeros(prefix+(32,), dtype=dtype, device=device)


def original_reward(qpos, qvel, qref, vref, positions, targets, mask, settings, *, terminal=False):
    """Execute retained, byte-hashed source functions rather than retype math."""
    from retarget.task.humanoid import cost as source

    wp.init()
    cfg = SimpleNamespace(**asdict(settings), embodiment_type="right", nq_obj=7,
        nv=32, nu=26, device="cpu", contact_site_ids=[2, 0, 3])
    # Include unused sites and a nontrivial order so guidance indexing matters.
    sites = torch.zeros((len(qpos), 4, 3), dtype=qpos.dtype)
    sites[:, cfg.contact_site_ids] = positions
    env = SimpleNamespace(data_wp=SimpleNamespace(
        qpos=wp.array(qpos.numpy(), dtype=float, device="cpu"),
        qvel=wp.array(qvel.numpy(), dtype=float, device="cpu"),
        site_xpos=wp.array(sites.numpy(), dtype=wp.vec3, device="cpu")))
    reference = (qref, vref, torch.zeros(26), mask, targets)
    fn = source.get_terminal_reward if terminal else source.get_reward
    return fn(cfg, env, reference)[0]


def test_weighted_state_is_one_l2_norm_not_added_hand_and_object_norms():
    q, v = states()
    ref = q[0].clone()
    q[:, 0], q[:, 26] = 3., 4.
    result = step_cost(q, v, ref, torch.zeros(32))
    torch.testing.assert_close(result["qpos"], torch.full((2,), 5.))
    torch.testing.assert_close(result["hand_diagnostic"], torch.full((2,), 3.))
    torch.testing.assert_close(result["object_diagnostic"], torch.full((2,), 4.))
    torch.testing.assert_close(result["total"], torch.full((2,), 5.))


def test_cost_matches_retained_spider_dex_reward_on_all_requested_terms():
    generator = torch.Generator().manual_seed(1226)
    q, v = states(batch=5)
    q[:, :29] = torch.randn((5, 29), generator=generator)*.07
    q[:, 29:] = torch.nn.functional.normalize(torch.randn((5, 4), generator=generator), dim=-1)
    v[:] = torch.randn((5, 32), generator=generator)*.02
    ref = q[0].clone(); ref[:29] += .015
    ref[29:] = torch.tensor([1., 0., 0., 0.])
    vref = torch.full((32,), .013)
    points = torch.randn((5, 3, 3), generator=generator)*.03
    target = torch.randn((3, 3), generator=generator)*.02
    mask = torch.tensor([1., 0., .5])
    settings = SpiderCostSettings(vel_rew_scale=.4, contact_rew_scale=10.)
    expected = -original_reward(q, v, ref, vref, points, target, mask, settings)
    actual = step_cost(q, v, ref, vref, contact_positions=points,
        contact_target_world=target, contact_mask=mask, settings=settings)
    torch.testing.assert_close(actual["total"], expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(actual["total"], actual["qpos"]+actual["qvel"]+actual["contact"])


def test_contact_scale_is_original_enable_gate_and_released_slots_are_zero_cost():
    q, v = states(batch=1)
    points = torch.tensor([[[3., 0., 0.], [100., 0., 0.], [0., 4., 0.]]])
    target, mask = torch.zeros((3, 3)), torch.tensor([.5, 0., 1.])
    kwargs = dict(contact_positions=points, contact_target_world=target, contact_mask=mask)
    enabled = step_cost(q, v, q[0], v[0], settings=SpiderCostSettings(contact_rew_scale=1.), **kwargs)
    larger_gate = step_cost(q, v, q[0], v[0], settings=SpiderCostSettings(contact_rew_scale=100.), **kwargs)
    disabled = step_cost(q, v, q[0], v[0], **kwargs)
    assert enabled["contact"].item() == 5.5
    torch.testing.assert_close(enabled["total"], larger_gate["total"])
    assert disabled["contact"].item() == 0.


def test_contact_target_is_reference_world_point_not_inferred_sim_object_attachment():
    q, v = states(batch=1)
    ref = q[0].clone()
    q[:, 26] = .1
    # Moving the actual object and matching hand material point together still
    # deviates from the immutable world-space CA target used by SPIDER.
    actual_point = torch.tensor([[[.1, 0., 0.]]])
    result = step_cost(q, v, ref, v[0], contact_positions=actual_point,
        contact_target_world=torch.zeros((1, 3)), contact_mask=torch.ones(1),
        settings=SpiderCostSettings(contact_rew_scale=1.))
    assert result["contact"].item() == pytest.approx(.1)
    assert result["total"].item() == pytest.approx(.2)


def test_moving_reference_trajectory_and_terminal_mean_match_original_rollout():
    generator = torch.Generator().manual_seed(42)
    q, v = states(batch=3, horizon=4)
    q[..., :29] = torch.randn((3, 4, 29), generator=generator)*.03
    q[..., 29:] = torch.nn.functional.normalize(torch.randn((3, 4, 4), generator=generator), dim=-1)
    v[:] = torch.randn(v.shape, generator=generator)*.01
    qref = states(batch=4)[0]
    qref[:, 26] = torch.arange(4)*.01
    vref = torch.randn((4, 32), generator=generator)*.01
    points = torch.randn((3, 4, 3, 3), generator=generator)*.02
    targets = torch.randn((4, 3, 3), generator=generator)*.01
    masks = torch.tensor([[1., 0., 1.], [0., 1., 1.], [1., 1., 0.], [.5, 1., 1.]])
    settings = SpiderCostSettings(contact_rew_scale=1., terminal_rew_scale=2.3)
    expected = torch.zeros(3)
    for time in range(4):
        expected -= original_reward(q[:, time], v[:, time], qref[time], vref[time],
            points[:, time], targets[time], masks[time], settings, terminal=time == 3)/4
    result = trajectory_cost(q, v, qref, vref, contact_positions=points,
        contact_target_world=targets, contact_mask=masks, settings=settings)
    torch.testing.assert_close(result["total"], expected, rtol=2e-6, atol=2e-7)


def test_post_step_horizon_uses_H_denominator_and_no_q0():
    q, v = states(batch=1, horizon=3)
    ref = q[0].clone()
    q[0, :, 26] = torch.tensor([1., 2., 3.])
    result = trajectory_cost(q, v, ref, v[0], settings=SpiderCostSettings(terminal_rew_scale=3.))
    assert result["total"].item() == 4.  # (1 + 2 + 3*3) / 3


def test_quaternion_sign_and_original_axis_angle_convention():
    from retarget.task.humanoid.math import quat_sub
    generator = torch.Generator().manual_seed(44717007)
    actual = torch.nn.functional.normalize(torch.randn((50, 4), generator=generator), dim=-1)
    target = torch.nn.functional.normalize(torch.randn((50, 4), generator=generator), dim=-1)
    torch.testing.assert_close(quaternion_difference(actual, target), quat_sub(actual, target), rtol=2e-6, atol=3e-7)
    torch.testing.assert_close(quaternion_difference(-actual, target), quaternion_difference(actual, target), rtol=3e-6, atol=5e-7)
    q, v = states(batch=2)
    q[1, -4:] *= -1
    assert torch.equal(step_cost(q, v, q[0], v[0])["total"], torch.zeros(2))


def test_velocity_uses_original_generalized_coordinates_without_frame_rotation():
    q, v = states(batch=1)
    vref = v[0].clone()
    q[0, -4:] = torch.tensor([np.sqrt(.5), 0., 0., np.sqrt(.5)])
    v[0, -3], vref[-2] = 1., 1.
    result = step_cost(q, v, q[0], vref, settings=SpiderCostSettings(vel_rew_scale=1.))
    assert result["qvel"].item() == pytest.approx(np.sqrt(2.))


def test_nonfinite_or_zero_quaternion_candidates_cannot_win_mppi():
    q, v = states(batch=3, horizon=2)
    ref, vref = q[0].clone(), v[0].clone()
    q[1, 1, 0] = torch.nan
    q[2, 0, -4:] = 0.
    result = trajectory_cost(q, v, ref, vref, settings=SpiderCostSettings(terminal_rew_scale=0.))
    assert result["total"][0] == 0.
    assert torch.isinf(result["total"][1:]).all()


def test_enabled_contact_tracking_cannot_silently_drop_missing_correspondences():
    q, v = states()
    with pytest.raises(ValueError, match="Enabled contact"):
        step_cost(q, v, q[0], v[0], settings=SpiderCostSettings(contact_rew_scale=1.))
    with pytest.raises(ValueError, match="qvel"):
        step_cost(q, v[:, :-1], q[0], v[0])
    with pytest.raises(ValueError, match="nonempty_horizon"):
        trajectory_cost(q[:, None, :][:, :0], v[:, None, :][:, :0], q[0], v[0])
    with pytest.raises(ValueError, match="nonnegative"):
        SpiderCostSettings(pos_rew_scale=-1.)


def test_cached_weights_match_standalone_cost_and_reject_wrong_shape():
    q, v = states()
    q[:, 6] = .2
    settings = SpiderCostSettings(joint_rew_scale=.4)
    scale = make_hand_scale(settings)
    direct = step_cost(q, v, states(batch=1)[0][0], v[0], settings=settings)
    cached = step_cost(q, v, states(batch=1)[0][0], v[0], settings=settings, hand_scale=scale)
    for name in direct:
        torch.testing.assert_close(cached[name], direct[name])
    with pytest.raises(ValueError, match="hand_scale"):
        step_cost(q, v, q[0], v[0], hand_scale=scale[:-1])


def test_capture_temporal_cache_preserves_nonunit_terminal_scaling():
    q, v = states(batch=2, horizon=3)
    ref = q[0].clone()
    q[..., 26] = torch.tensor([1., 2., 3.])
    settings = SpiderCostSettings(terminal_rew_scale=2.5)
    direct = trajectory_cost(q, v, ref, v[0], settings=settings)
    cached = trajectory_cost(q, v, ref, v[0], settings=settings,
        hand_scale=make_hand_scale(settings), temporal_weights=torch.tensor([1., 1., 2.5]))
    for name in direct:
        torch.testing.assert_close(cached[name], direct[name])
    torch.testing.assert_close(cached["total"], torch.full((2,), 3.5))
    with pytest.raises(ValueError, match="Cached temporal weights"):
        trajectory_cost(q, v, ref, v[0], settings=settings, temporal_weights=torch.ones(2))


@pytest.mark.parametrize("normal_force, expected", [(.1, 0.), (.05, 0.), (.025, .5), (0., 2.), (-.1, 2.)])
def test_local_contact_keep_shortage_is_bounded_and_uses_actual_force(normal_force, expected):
    q, v = states(batch=2)
    result = step_cost(q, v, q[0], v[0],
        finger_normal_forces=torch.tensor(normal_force),
        required_finger_mask=torch.tensor([True, False, False, False, False]),
        settings=SpiderCostSettings(contact_keep_rew_scale=2., contact_min_force_n=.05))
    torch.testing.assert_close(result["contact_keep"], torch.full((2,), expected))
    torch.testing.assert_close(result["total"], result["contact_keep"])
    assert torch.equal(result["contact"], torch.zeros(2))


def test_local_contact_keep_counts_required_fingers_once_and_ignores_inactive_loads():
    q, v = states(batch=2)
    force = torch.tensor([[0., -.3, .05, 0., .025], [0., 1., .05, -10., .025]])
    # Two inactive fingers may differ arbitrarily without changing the score.
    result = step_cost(q, v, q[0], v[0], finger_normal_forces=force,
        required_finger_mask=torch.tensor([1., 0., 1., 0., 1.]),
        settings=SpiderCostSettings(contact_keep_rew_scale=4.))
    torch.testing.assert_close(result["contact_keep"], torch.full((2,), 5.))
    scalar_mask = step_cost(q, v, q[0], v[0], finger_normal_forces=torch.tensor(0.),
        required_finger_mask=torch.tensor(True), settings=SpiderCostSettings(contact_keep_rew_scale=4.))
    torch.testing.assert_close(scalar_mask["contact_keep"], torch.full((2,), 20.))
    inactive = step_cost(q, v, q[0], v[0], finger_normal_forces=force,
        required_finger_mask=torch.tensor(False), settings=SpiderCostSettings(contact_keep_rew_scale=4.))
    assert torch.equal(inactive["total"], torch.zeros(2))


def test_local_contact_keep_batch_and_horizon_broadcast_with_cached_terminal_weights():
    q, v = states(batch=2, horizon=3, dtype=torch.float64)
    ref = q[0].clone()
    q[..., 26] = torch.tensor([1., 2., 3.])
    # The force trajectory broadcasts over candidates; each candidate has a
    # distinct required-finger set that broadcasts over the horizon.
    forces = torch.tensor([[.05]*5, [.025]*5, [0.]*5], dtype=torch.float64)
    mask = torch.tensor([[[1., 1., 1., 1., 1.]], [[1., 0., 1., 0., 0.]]], dtype=torch.float64)
    settings = SpiderCostSettings(contact_keep_rew_scale=2., terminal_rew_scale=3.)
    kwargs = dict(finger_normal_forces=forces, required_finger_mask=mask, settings=settings)
    steps = step_cost(q, v, ref, v[0], **kwargs)
    torch.testing.assert_close(steps["contact_keep"], torch.tensor([[0., 2.5, 10.], [0., 1., 4.]], dtype=q.dtype))
    result = trajectory_cost(q, v, ref, v[0], **kwargs)
    cached = trajectory_cost(q, v, ref, v[0], hand_scale=make_hand_scale(settings, dtype=q.dtype),
        temporal_weights=torch.tensor([1., 1., 3.], dtype=q.dtype), **kwargs)
    for name in result:
        torch.testing.assert_close(cached[name], result[name])
    torch.testing.assert_close(result["contact_keep"], torch.tensor([32.5/3., 13./3.], dtype=q.dtype))
    torch.testing.assert_close(result["qpos"], torch.full((2,), 4., dtype=q.dtype))
    torch.testing.assert_close(result["total"], result["qpos"]+result["qvel"]+result["contact"]+result["contact_keep"])


@pytest.mark.parametrize("bad", [torch.nan, torch.inf, -torch.inf])
def test_local_contact_keep_nonfinite_force_or_mask_rejects_candidate(bad):
    q, v = states(batch=3, horizon=2)
    force = torch.full((3, 2, 5), .05)
    mask = torch.ones_like(force)
    force[1, -1, 0] = bad
    mask[2, 0, 4] = bad
    # Invalid final measurements remain rejected even at terminal scale zero.
    result = trajectory_cost(q, v, q[0], v[0], finger_normal_forces=force,
        required_finger_mask=mask,
        settings=SpiderCostSettings(contact_keep_rew_scale=1., terminal_rew_scale=0.))
    assert result["total"][0] == 0.
    assert torch.isinf(result["total"][1:]).all()
    force[0, 0, 0] = bad
    inactive = step_cost(q[:, 0], v[:, 0], q[0, 0], v[0, 0], finger_normal_forces=force[:, 0],
        required_finger_mask=torch.tensor(False), settings=SpiderCostSettings(contact_keep_rew_scale=1.))
    assert torch.isinf(inactive["total"][0])


def test_disabled_local_contact_keep_preserves_spider_without_force_inputs():
    q, v = states(batch=2, horizon=3)
    q[..., 26] = .04
    ref = states(batch=3)[0]
    unchanged = trajectory_cost(q, v, ref, v[0])
    # Disabled extension does not inspect optional inputs, including NaNs or
    # otherwise incompatible masks; the original objective stays bitwise equal.
    ignored = trajectory_cost(q, v, ref, v[0], finger_normal_forces=torch.tensor(torch.nan),
        required_finger_mask=torch.tensor([1, 2], dtype=torch.int64),
        settings=SpiderCostSettings(contact_keep_rew_scale=0., contact_min_force_n=.2))
    for name in unchanged:
        assert torch.equal(unchanged[name], ignored[name])
    assert torch.equal(unchanged["contact_keep"], torch.zeros(2))


@pytest.mark.parametrize("invalid_mask", [-1., 1.01])
def test_local_contact_keep_invalid_mask_cannot_reward_contact_loss(invalid_mask):
    q, v = states(batch=2)
    mask = torch.tensor([[1., 0., .5, 0., 1.], [1., 0., .5, 0., 1.]])
    mask[1, 0] = invalid_mask
    result = step_cost(q, v, q[0], v[0], finger_normal_forces=torch.tensor(0.),
        required_finger_mask=mask, settings=SpiderCostSettings(contact_keep_rew_scale=2.))
    assert result["total"][0] == 5.
    assert torch.isinf(result["total"][1])


def test_local_contact_keep_requires_valid_configuration_inputs_dtype_and_shape():
    q, v = states()
    settings = SpiderCostSettings(contact_keep_rew_scale=1.)
    with pytest.raises(ValueError, match="Enabled contact keeping"):
        step_cost(q, v, q[0], v[0], settings=settings)
    with pytest.raises(ValueError, match="Enabled contact keeping"):
        step_cost(q, v, q[0], v[0], finger_normal_forces=torch.ones(5), settings=settings)
    with pytest.raises(ValueError, match="finger_normal_forces.*dtype"):
        step_cost(q, v, q[0], v[0], finger_normal_forces=torch.ones(5, dtype=torch.float64),
            required_finger_mask=torch.ones(5), settings=settings)
    with pytest.raises(ValueError, match="required_finger_mask.*dtype"):
        step_cost(q, v, q[0], v[0], finger_normal_forces=torch.ones(5),
            required_finger_mask=torch.ones(5, dtype=torch.int64), settings=settings)
    with pytest.raises(ValueError, match="finger_normal_forces.*broadcast"):
        step_cost(q, v, q[0], v[0], finger_normal_forces=torch.ones(4),
            required_finger_mask=torch.ones(5), settings=settings)
    with pytest.raises(ValueError, match="required_finger_mask.*broadcast"):
        step_cost(q, v, q[0], v[0], finger_normal_forces=torch.ones(5),
            required_finger_mask=torch.ones(6), settings=settings)
    for minimum in (0., -1., torch.nan, torch.inf, True):
        with pytest.raises(ValueError):
            SpiderCostSettings(contact_min_force_n=minimum)
    with pytest.raises(ValueError, match="contact_keep_rew_scale"):
        SpiderCostSettings(contact_keep_rew_scale=-1.)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_tensor_cost_matches_cpu_without_host_scalar_branches():
    q, v = states(batch=3, horizon=4)
    ref = q[0].clone(); q[..., 26] = .012
    cpu = trajectory_cost(q, v, ref, v[0])
    gpu = trajectory_cost(q.cuda(), v.cuda(), ref.cuda(), v[0].cuda())
    for name in cpu:
        torch.testing.assert_close(gpu[name].cpu(), cpu[name], rtol=1e-6, atol=1e-7)
