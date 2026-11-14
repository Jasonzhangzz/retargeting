import numpy as np
import pytest
import torch
import warp as wp
from retarget.env.contact.gpu import GPURollout
from retarget.env.contact.native import snapshot
from retarget.env.contact.strategy import ContactStrategy
from retarget.task.contact.tasks import Task

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA')


def test_proposal_order_and_repeated_rollouts_are_independent():
    wp.config.log_level = wp.LOG_WARNING
    task = Task('h1_jog')
    state = snapshot(task.state)
    packet = task.packet(state, 0)
    rollout = GPURollout(task, 4, 'cuda:0')
    u = np.repeat(packet['ctrl_ref'][None], 4, axis=0)
    u[1, :, 3] += .1; u[2, :, 4] -= .05
    cost = rollout.evaluate(u, state, packet, [[], [], [], []])
    reverse = rollout.evaluate(u[::-1].copy(), state, packet, [[], [], [], []])
    again = rollout.evaluate(u, state, packet, [[], [], [], []])
    assert np.all(np.isfinite(cost))
    np.testing.assert_allclose(cost, reverse[::-1], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cost, again, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(task.state.qpos, state['qpos'])
    strategy = ContactStrategy(task, rollout)
    branches = strategy.branches(state, packet)
    strategy.nominal = packet['ctrl_ref']
    controls, events = strategy.decode(np.zeros((4, 30)), 0)
    assert controls.shape == u.shape
    assert len(branches) >= 2 and len(events[0]) <= 3
    assert np.isfinite(controls).all()
    out = rollout.evaluate(controls, state, packet, events)
    assert np.isfinite(out).all()
    assert rollout.physics.__class__.__module__.startswith('robokit.sim')


def test_rotating_coin_velocities_stay_in_the_surface_tangent_plane():
    wp.config.log_level = wp.LOG_WARNING
    task = Task('sharpa_coin')
    state = snapshot(task.state)
    packet = task.packet(state, 0)
    rollout = GPURollout(task, 4, 'cuda:0')
    strategy = ContactStrategy(task, rollout)
    branches = strategy.branches(state, packet)
    strategy.nominal = packet['ctrl_ref']
    values = np.zeros((4, 30))
    values[:, 8] = .05; values[:, 9] = -.03
    controls, events = strategy.decode(values, 1)
    assert np.isfinite(controls).all()
    for row in events:
        for event in row:
            _, normal, rotation = strategy.target_pose(event)
            assert abs(normal @ event['tangential_velocity']) < 1e-10
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
            assert np.linalg.det(rotation) > .999999


def test_partial_ik_batch_restores_world_count_and_nan_proposal_is_rejected():
    wp.config.log_level = wp.LOG_WARNING
    task = Task('h1_jog')
    state = snapshot(task.state); packet = task.packet(state, 0)
    rollout = GPURollout(task, 4, 'cuda:0')
    strategy = ContactStrategy(task, rollout)
    strategy.branches(state, packet); strategy.nominal = packet['ctrl_ref']
    controls, events = strategy.decode(np.zeros((2, 30)), 0)
    assert rollout.physics.data.nworld == 4 and controls.shape[0] == 2
    controls[0, 0, 0] = np.nan
    cost = rollout.evaluate(controls, state, packet, events)
    assert np.isinf(cost[0]) and np.isfinite(cost[1])


def test_contact_lm_uses_future_reference_root_and_preserves_control_feedforward():
    from tests.contact.test_reference import frozen_motion
    task = Task('h1_jog'); task.reference = frozen_motion(task)
    packet = task.packet(task.initial, 0)
    rollout = GPURollout(task, 4, 'cuda:0')
    strategy = ContactStrategy(task, rollout)
    branches = strategy.branches(snapshot(task.state), packet)
    strategy.nominal = packet['ctrl_ref']
    controls, events = strategy.decode(np.zeros((4, 30)), 0)
    np.testing.assert_array_equal(controls, np.broadcast_to(packet['ctrl_ref'], controls.shape))
    for e in events[0]:
        k = round(e['start_time'] / packet['dt'])
        point = np.asarray(e['object_local_point'])
        # A future moving root must move the reference landing point too.
        if k > 0:
            assert point[0] > .03
        assert k >= 0 and e['duration'] > 0


def test_h1_alternatives_change_support_times_and_separate_approach_from_stance():
    from types import SimpleNamespace
    task = Task('h1_jog')
    packet = task.packet(task.initial, 0)
    strategy = ContactStrategy(task, SimpleNamespace(device='cuda:0'))
    branches = strategy.branches(task.initial, packet)
    first = {e['geom1']: (e['start_time'], e['duration']) for e in branches[0].events}
    other = {e['geom1']: (e['start_time'], e['duration']) for e in branches[2].events}
    assert first.keys() == other.keys() and first != other
    strategy.nominal = packet['ctrl_ref']
    x = np.zeros((2, 30)); x[:, 7] = .2; x[:, 17] = .2
    _, events = strategy.decode(x, 0)
    for row in events:
        for event in row:
            assert event['normal_velocity'] == 0.
            assert event['approach_normal_velocity'] <= 0.
            assert event['start_time'] + event['duration'] <= task.profile.horizon * packet['dt'] + 1e-9
    assert np.all(branches[0].sigma[20:] == 0.)


def test_whole_body_nominal_is_state_isolated_and_its_extra_budget_is_counted():
    from retarget.env.contact.robokit_nominal import SourceNominalPlanner
    task = Task('h1_jog')
    state = snapshot(task.state); packet = task.packet(state, 0)
    planner = SourceNominalPlanner(task, 'cuda:0')
    controls = planner.propose(state, 0, packet)
    assert controls.shape == (task.profile.horizon + 1, task.model.nu)
    assert np.isfinite(controls).all()
    np.testing.assert_array_equal(task.state.qpos, state['qpos'])
    np.testing.assert_array_equal(task.state.qvel, state['qvel'])
    assert task.state.time == state['time']
    assert planner.trajectories == task.profile.initial_iterations * (task.profile.samples - 1) + 1
    assert planner.cold_rollouts > 0


def test_lazy_nominal_validation_inherits_frozen_window_references():
    from retarget.env.contact.robokit_nominal import SourceNominalPlanner
    from tests.contact.test_reference import frozen_motion
    task = Task('h1_jog'); task.reference = frozen_motion(task); task.bind_reference()
    packet = task.packet(task.initial, 0)
    planner = SourceNominalPlanner(task, 'cuda:0')
    planner.propose(task.initial, 0, packet)
    assert {built.sim.nworld for built in planner.reference_term.bindings} >= {1, task.profile.samples - 1}
    for built in planner.reference_term.bindings:
        np.testing.assert_allclose(built.qref.numpy(), packet['qpos_ref'], atol=1e-6)
        np.testing.assert_allclose(built.vref.numpy(), packet['qvel_ref'], atol=1e-6)
