"""Independent free-fall and display-coordinate checks for gravity playback."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from test_grasp_device import make_runtime
from retarget.grasp_playback import playback_paths, simulate_playback
from scripts.view_grasp_results import load_geometry, playback_geometry

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def saved_case(tmp_path, *, free_object=False):
    runtime = make_runtime(tmp_path/'runtime', .025)
    case = tmp_path/'case'
    case.mkdir()
    (case/'input_00').symlink_to(tmp_path/'runtime/prepared', target_is_directory=True)
    (case/'scene.xml').write_text(runtime.scene_xml)
    (case/'result.json').write_text(json.dumps(dict(selected=dict(start=0))))
    q = runtime.qpos.copy()
    if free_object:
        q[runtime.index.object_qpos] += .4  # Out of hand reach, no floor or support.
    np.savez_compressed(case/'grasp.npz', qpos=q, qvel=np.zeros(runtime.model.nv), hold_action=runtime.initial_action)
    return case, runtime, q


def test_playback_matches_unassisted_free_fall_and_preserves_frame_zero(tmp_path):
    case, runtime, q = saved_case(tmp_path, free_object=True)
    path, report = simulate_playback(case, duration=.1)
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved['qpos'][0], q.astype(np.float32))
        np.testing.assert_array_equal(saved['qvel'][0], 0.)
        times = saved['time']
        fall = q[runtime.index.object_qpos+2]-saved['qpos'][:, runtime.index.object_qpos+2]
        # Semi-implicit Euler under gravity, allowing the free-joint armature.
        expected = .5*9.81*(times*times+times*.0005)
        np.testing.assert_allclose(fall, expected, atol=6.e-5, rtol=.002)
    assert report['status'] == 'dropped' and report['first_drop_seconds'] is not None
    assert report['backend'].startswith('native ComFree Warp') and report['simulation_error'] is None
    assert report['frames'] == 6 and report['saved_velocity'] is False
    stamp = path.stat().st_mtime_ns
    cached, cached_report = simulate_playback(case, duration=.1)
    assert cached == path and path.stat().st_mtime_ns == stamp
    assert cached_report == report
    different, _, _ = playback_paths(case, duration=.1, saved_velocity=True)
    assert different != path


def test_animated_mesh_transforms_match_saved_mujoco_kinematics(tmp_path):
    case, _, q = saved_case(tmp_path)
    meshes, _, center, model = load_geometry(str(case))
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    for vertices, _, _, _, position, quaternion, geom in meshes:
        displayed = Rotation.from_quat(quaternion, scalar_first=True).apply(vertices[0])+position
        expected = data.geom_xmat[geom].reshape(3, 3)@vertices[0]+data.geom_xpos[geom]-center
        np.testing.assert_allclose(displayed, expected, atol=2.e-7)
    next_q = q.copy()
    next_q[model.jnt_qposadr[model.joint('obj_joint').id]+2] -= .1
    trajectory = tmp_path/'trajectory.npz'
    np.savez(trajectory, qpos=np.array([q, next_q]), qvel=np.zeros((2, model.nv)), time=[0., .02])
    playback = playback_geometry(trajectory, model, meshes, center)
    for index, mesh in enumerate(meshes):
        np.testing.assert_allclose(playback['geom_positions'][0, index], mesh[4], atol=2.e-7)
        if mesh[3]:
            np.testing.assert_allclose(playback['geom_positions'][1, index]-mesh[4], [0., 0., -.1], atol=2.e-7)
