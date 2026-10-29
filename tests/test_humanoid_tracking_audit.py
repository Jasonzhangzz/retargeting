"""Check that diagnostic coordinates distinguish posture from task tracking."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from retarget.humanoid import HumanoidProblem
from scripts.audit_humanoid_tracking import tracking_errors


@pytest.fixture(scope='module')
def problem():
    root = Path(__file__).resolve().parents[1] / 'data/spider/processed/omomo/unitree_g1/humanoid_object/move_largebox'
    return HumanoidProblem(root / 'scene.xml', root / '0/trajectory_kinematic.npz')


def state(problem, translation=None, translate_box=False):
    data = problem.initial_state(20)
    data.qvel[:] = problem.tracking_vref[20]
    if translation is not None:
        data.qpos[:3] += translation
        if translate_box:
            body = problem.model.geom_bodyid[problem.object_geom]
            adr = problem.model.jnt_qposadr[problem.model.body_jntadr[body]]
            data.qpos[adr:adr+3] += translation
    mujoco.mj_forward(problem.model, data)
    mujoco.mj_subtreeVel(problem.model, data)
    return data


def test_reference_match_has_zero_tracking_and_balance_errors(problem):
    result = tracking_errors(problem, state(problem), state(problem))
    for key, value in result.items():
        if ('error' in key and not key.endswith('_hand_task_error_m')) or key.endswith('_cost') or key in ('torso_tilt_deg', 'height_deficit_m'):
            assert value == pytest.approx(0., abs=1e-12), key
    assert not result['balance_failure']
    # Contact-compatible targets project the kinematic hand onto the box;
    # the released reference itself can therefore have nonzero task error.
    data=state(problem);geom=problem.object_geom
    centers=(data.geom_xpos[problem.hand_geoms]-data.geom_xpos[geom])@data.geom_xmat[geom].reshape(3,3)
    surface=problem.surface.project(centers)
    expected=np.linalg.norm(centers-surface,axis=1)-problem.model.geom_size[problem.hand_geoms,0]
    for side,value in zip(('left','right'),expected):
        assert result[side+'_hand_task_error_m']==pytest.approx(abs(value),abs=1.e-10)


def test_robot_translation_is_visible_in_world_and_box_but_not_pelvis_tracking(problem):
    delta = np.array([.25, 0., 0.])
    result = tracking_errors(problem, state(problem, delta), state(problem))
    for side in ('left', 'right'):
        assert result[side + '_wrist_world_error_m'] == pytest.approx(.25)
        assert result[side + '_wrist_pelvis_error_m'] == pytest.approx(0., abs=1e-12)
        assert result[side + '_hand_box_error_m'] == pytest.approx(.25)
    assert result['keypoint_position_cost'] == pytest.approx(0., abs=1e-12)
    assert result['com_xy_error_m'] == pytest.approx(.25)
    assert result['balance_com_xy_cost'] == pytest.approx(problem.weights.balance * (.25 / problem.settings.balance_xy_scale) ** 2)


def test_common_box_robot_translation_cancels_in_box_relative_error(problem):
    delta = np.array([.2, -.15, 0.])
    result = tracking_errors(problem, state(problem, delta, translate_box=True), state(problem))
    for side in ('left', 'right'):
        assert result[side + '_hand_world_error_m'] == pytest.approx(.25)
        assert result[side + '_hand_box_error_m'] == pytest.approx(0., abs=1e-12)
    assert result['box_position_error_m'] == pytest.approx(.25)
