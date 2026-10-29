"""Verify that the counterfactual score executes the upstream pose definition."""
from pathlib import Path

import mujoco
import numpy as np

from scripts.analyze_humanoid_cost import spider_pose_cost


def test_spider_pose_score_uses_object_pose_and_weighted_unsquared_norm():
    root = Path(__file__).resolve().parents[1] / 'data/spider/processed/omomo/unitree_g1/humanoid_object/move_largebox'
    model = mujoco.MjModel.from_xml_path(str(root / 'scene.xml'))
    with np.load(root / '0/trajectory_kinematic.npz') as arrays:
        reference = arrays['qpos'][0].astype(float)
    actual = reference.copy()
    actual[0] += .2
    actual[7] += .1
    actual[-7] += .3
    velocity = np.zeros(model.nv)
    velocity[-3] = .4
    mujoco.mj_integratePos(model, actual, velocity, 1.)
    score, parts = spider_pose_cost(actual, reference)
    np.testing.assert_allclose(score, np.sqrt(.2**2 + .1**2 + .3**2 + (.3*.4)**2), rtol=2.e-6)
    np.testing.assert_allclose(parts['box_position'], .3**2, rtol=2.e-6)
    np.testing.assert_allclose(parts['box_rotation'], (.3*.4)**2, rtol=2.e-6)
    assert parts['root_rotation'] < 1.e-12


def test_spider_pose_score_keeps_individual_batched_reference_states():
    root = Path(__file__).resolve().parents[1] / 'data/spider/processed/omomo/unitree_g1/humanoid_object/move_largebox'
    with np.load(root / '0/trajectory_kinematic.npz') as arrays:
        reference = arrays['qpos'][:4].astype(float)
    actual = np.broadcast_to(reference[:, None, :], (4, 2, 43)).copy()
    actual[:, 1, -7] += .25
    score, _ = spider_pose_cost(actual, reference[:, None, :])
    np.testing.assert_allclose(score[:, 0], 0., atol=1.e-6)
    np.testing.assert_allclose(score[:, 1], .25, atol=1.e-6)
