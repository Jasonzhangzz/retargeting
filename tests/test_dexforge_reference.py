"""Raw G1 references replace CA results without changing poses or joint angles."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from retarget.env.humanoid.dexforge_reference import load_reference


DATA = Path(__file__).resolve().parents[1] / "data/humanoid/move_largebox"


@pytest.fixture
def reference():
    model = mujoco.MjModel.from_xml_path(str(DATA / "scene.xml"))
    return load_reference(DATA / "reference.npz", model)


def test_raw_ca_replacement_and_explicit_free_root_mapping(reference):
    with np.load(DATA / "reference.npz") as raw:
        for key in ("qpos", "qvel", "ctrl"):
            np.testing.assert_array_equal(getattr(reference, key), raw[key])
    assert reference.qpos.shape == (145, 43)
    np.testing.assert_array_equal(reference.index.hand_qpos, np.arange(7, 36))
    np.testing.assert_array_equal(reference.index.hand_ctrl, np.arange(29))
    assert reference.index.object_qpos == 36
    assert reference.index.action_dim == 29
    np.testing.assert_array_equal(reference.sequence.hand, reference.qpos[:, 7:36])
    np.testing.assert_array_equal(reference.sequence.hand, reference.ctrl)
    np.testing.assert_array_equal(reference.sequence.object_pose_xyzw,
                                  reference.qpos[:, [40,41,42,39,36,37,38]])
    assert reference.duration_seconds == pytest.approx(4.7999952)
    assert not reference.sequence.contact.mask.any()
    assert not reference.sequence.contact.weight.any()
    assert reference.sequence.hand_source == "dataset"


def test_upstream_quaternion_interpolation_and_full_root_initial_velocity(reference):
    qpos, velocity = reference.initial_state()
    np.testing.assert_allclose(qpos, reference.qpos[0], atol=2e-7, rtol=0)
    assert not np.allclose(qpos[:7], reference.model.qpos0[:7])
    expected = np.zeros(reference.model.nv)
    following = reference.qpos_at([1])[0]
    mujoco.mj_differentiatePos(reference.model, expected, reference.ref_dt,
                             qpos.astype(np.float64), following.astype(np.float64))
    np.testing.assert_allclose(velocity, expected, atol=3e-7, rtol=1e-7)
    assert np.linalg.norm(velocity[:6]) > 0.1
    assert np.count_nonzero(reference.qvel) == 0
    _, held = reference.initial_state(hold=True)
    np.testing.assert_array_equal(held, np.zeros(41))
    middle = reference.qpos_at([0.5, 30.25, 144])
    np.testing.assert_allclose(np.linalg.norm(middle[:, 3:7], axis=-1), 1., atol=2e-7)
    np.testing.assert_allclose(np.linalg.norm(middle[:, -4:], axis=-1), 1., atol=2e-7)
    np.testing.assert_allclose(reference.qpos_at([144])[0], reference.qpos[-1], atol=2e-7, rtol=0)


def test_window_uses_original_upstream_frame_grid_and_supplies_root(reference):
    from forceaware.time_grid import TimeGrid
    from forceaware.targets import make_window

    grid = TimeGrid.create(mpc_dt=1/60, action_dt=.1, knot_dt=.1, exec_dt=1/60,
                           ref_dt=reference.ref_dt)
    arguments = dict(executed_steps=6, horizon=8, time_grid=grid,
                     start_frame=3.5, end_frame=144, hold_frame=-1)
    actual = reference.make_window(**arguments)
    expected = make_window(reference.sequence, **arguments)
    for name in vars(expected):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    np.testing.assert_array_equal(actual.root_qpos, reference.root_at(actual.frames))
    assert actual.actions.shape == (8, 29)
    assert actual.root_qpos.shape == (48, 7)
    ending = reference.make_window(executed_steps=285, horizon=3, time_grid=grid)
    assert ending.frames[-1] == 144
    np.testing.assert_allclose(ending.object_qpos[-1], reference.qpos[-1, 36:], atol=2e-7, rtol=0)
