"""Independent transfer starts at the saved state and replays only dense controls."""

from pathlib import Path

import mujoco
import numpy as np

from retarget.env.humanoid.dexforge_replay import replay_controls


def test_independent_replay_has_no_initialization_step(tmp_path):
    data_dir = Path(__file__).resolve().parents[1] / "data/humanoid/move_largebox"
    model = mujoco.MjModel.from_xml_path(str(data_dir / "scene.xml"))
    source = np.load(data_dir / "reference.npz")
    qpos0, qvel0 = source["qpos"][0].astype(float), np.zeros(model.nv)
    qvel0[0] = .02
    control = source["ctrl"][:2].astype(float)
    time = np.asarray([.7, .7+1/60, .7+2/60])
    target = np.repeat(qpos0[None], 3, axis=0)
    path = tmp_path / "native_trajectory.npz"
    np.savez(path, qpos0=qpos0, qvel0=qvel0, ctrl=control, time=time, qpos_ref=target)
    result = replay_controls(data_dir / "scene.xml", path, tmp_path, complete=False)
    with np.load(tmp_path / "independent_mujoco/trajectory.npz") as replay:
        np.testing.assert_array_equal(replay["qpos"][0], qpos0)
        np.testing.assert_array_equal(replay["qvel"][0], qvel0)
        np.testing.assert_array_equal(replay["ctrl"], control)
        np.testing.assert_array_equal(replay["time"], time)
        expected = mujoco.MjData(model)
        expected.qpos[:], expected.qvel[:], expected.time = qpos0, qvel0, time[0]
        mujoco.mj_forward(model, expected)
        for i, action in enumerate(control):
            model.opt.timestep = time[i+1]-time[i]
            expected.ctrl[:] = action
            mujoco.mj_step(model, expected)
            np.testing.assert_array_equal(replay["qpos"][i+1], expected.qpos)
            np.testing.assert_array_equal(replay["qvel"][i+1], expected.qvel)
    assert result["evaluation"] == "independent_mujoco_transfer"
    assert result["executed_steps"] == 2
    assert not result["initialization_prestep"]
    assert not result["full_clip_completed"] and not result["task_success"]
