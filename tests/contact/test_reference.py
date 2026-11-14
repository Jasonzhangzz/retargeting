"""Reference tracking has a frozen kinematic target, separate from jog commands."""
import numpy as np
from retarget.task.contact.tasks import Task
from retarget.task.contact.metrics import assess


def frozen_motion(task):
    m = task.model
    q = np.repeat(task.state.qpos[None], 401, axis=0)
    q[:, 0] += np.arange(401) * m.opt.timestep * 2.4
    v = np.zeros((401, m.nv)); v[:, 0] = 2.4
    sensor = np.repeat(task.state.sensordata[None], 401, axis=0)
    sensor[:, task.adr['torso_vel']] = 2.4
    sensor[:, task.adr['torso_pos']] += np.arange(401) * m.opt.timestep * 2.4
    return dict(qpos=q, qvel=v, ctrl=np.tile(task.grasp, (400, 1)), sensordata=sensor)


def test_exact_frozen_tracking_is_not_confused_with_the_original_speed_command():
    task = Task('h1_jog')
    reference = frozen_motion(task)
    task.reference = reference
    packet = task.packet(task.initial, 0)
    assert np.all(packet['ref'][:, task.adr['torso_vel']] == 2.4)
    data = {k: reference[k][1:] for k in ('qpos', 'qvel', 'sensordata')}
    data['ctrl'] = reference['ctrl']
    rows = [dict(contacts=[]) for _ in range(400)]
    tracking = assess(task, data, rows, True)
    assert tracking['velocity_tracking_error'] == 0 and tracking['task_success']
    assert tracking['source_command_velocity_error'] > .025
    task.reference = None
    generation = assess(task, data, rows, True)
    assert not generation['task_success'] and not generation['fell']
