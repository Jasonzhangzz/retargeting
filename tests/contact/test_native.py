from pathlib import Path
import tempfile
import mujoco
import numpy as np
from retarget.env.contact.native import snapshot, restore
from retarget.env.contact.client import NativeClient
from retarget.task.contact.tasks import Task


def test_worker_candidates_do_not_advance_execution_and_order_is_independent(tmp_path):
    root = Path(__file__).resolve().parents[2]
    if not (root / '.venv-contact4/bin/python').exists():
        import pytest
        pytest.skip('Set up native MuJoCo 4 worker first')
    task = Task('h1_jog')
    xml = tmp_path / 'scene.xml'
    xml.write_text(task.scene.to_xml())
    state = snapshot(task.state)
    packet = task.packet(state, 0)
    packet['target_geoms'] = [task.model.geom('floor').id]
    u = np.repeat(packet['ctrl_ref'][None, :3], 2, axis=0)
    u[1, :, 3] += .1
    with NativeClient(xml, state, tmp_path) as native:
        state = native.info['state']
        result = native.call('evaluate', state=state, controls=u, events=[[], []], packet=packet, dynamics_cost=True)
        reverse = native.call('evaluate', state=state, controls=u[::-1].copy(), events=[[], []], packet=packet, dynamics_cost=True)
        assert np.allclose([r['cost'] for r in result], [r['cost'] for r in reverse][::-1])
        assert np.array_equal(result[0]['qpos'], reverse[1]['qpos'])
        rows = native.call('execute', controls=u[0, :1])
        assert abs(rows[0]['state']['time'] - state['time'] - task.model.opt.timestep) < 1e-10


def test_complete_restore_clears_previous_controls_forces_and_warmstart():
    task = Task('h1_jog')
    d = task.state
    state = snapshot(d)
    d.qfrc_applied[:] = 10
    d.qacc_warmstart[:] = 100
    d.ctrl[:] = 3
    restore(task.model, d, state)
    assert np.array_equal(d.qfrc_applied, state['qfrc_applied'])
    assert np.array_equal(d.ctrl, state['ctrl'])


def test_migrated_scenes_have_local_assets_and_correct_dofs():
    for name, expected in [('sharpa_pen', 22), ('sharpa_coin', 22), ('h1_jog', 19)]:
        task = Task(name)
        assert task.model.nu == expected and np.isfinite(task.state.qpos).all()
        from retarget.task.contact.tasks import PROFILES
        assert task.profile == PROFILES[name]
        xml = task.scene.to_xml()
        assert '/home/zz/robokit' not in xml and '/home/zz/.robokit' not in xml


def test_parallel_native_rollouts_match_serial_cost_states_and_predictions(tmp_path):
    import os
    if (os.cpu_count() or 1) < 3:
        import pytest
        pytest.skip('Parallel comparison needs at least three logical CPUs')
    from retarget.env.contact.prediction import distance_query
    task = Task('h1_jog'); m, d = task.model, task.state
    xml = tmp_path / 'scene.xml'; xml.write_text(task.scene.to_xml())
    state = snapshot(d); packet = task.packet(state, 0)
    packet['target_geoms'] = [m.geom('floor').id]
    g1, g2 = m.geom('left_foot').id, m.geom('floor').id
    _, p1, p2, normal = distance_query(m, d, g1, g2)
    e = dict(geom1=g1, geom2=g2,
        link_local_point=(d.geom_xmat[g1].reshape(3, 3).T @ (p1 - d.geom_xpos[g1])).tolist(),
        object_local_point=p2.tolist(),
        link_local_normal=(d.geom_xmat[g1].reshape(3, 3).T @ normal).tolist(),
        object_local_normal=(-normal).tolist(), start_time=0., duration=.1,
        normal_velocity=-.02, tangential_velocity=[0., 0., 0.], mode='stick')
    u = np.repeat(packet['ctrl_ref'][None, :3], 3, axis=0); u[1, :, 3] += .1
    results = []
    for workers in (1, 2):
        directory = tmp_path / str(workers); directory.mkdir()
        with NativeClient(xml, state, directory, workers=workers) as native:
            assert native.info['workers'] == workers
            results.append(native.call('evaluate', state=native.info['state'], controls=u,
                events=[[e]] * 3, packet=packet, dynamics_cost=True))
    for a, b in zip(*results):
        assert a['cost'] == b['cost'] and a['parts'] == b['parts']
        assert a['predictions'] == b['predictions']
        np.testing.assert_array_equal(a['qpos'], b['qpos'])
        np.testing.assert_array_equal(a['qvel'], b['qvel'])
