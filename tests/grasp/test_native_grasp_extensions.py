import numpy as np
import mujoco
from omegaconf import OmegaConf

from retarget.env.grasp.native_grasp_extensions import (
    Settings, configure_penetration, evaluate_wrenches, world_wrench,
)


def test_signed_penetration_configuration_preserves_pregrasp_clearance():
    cfg=OmegaConf.create({'op':{'grasp':{'filter':{'collision':{'ho_thre':0.,'hh_thre':0.}}},
                                 'pregrasp':{'filter':{'collision':{'ho_thre':.04,'hh_thre':0.}}}}})
    configure_penetration(cfg,Settings())
    assert cfg.op.grasp.filter.collision.ho_thre == -.002
    assert cfg.op.grasp.filter.collision.hh_thre == -.002
    assert cfg.op.pregrasp.filter.collision.ho_thre == .04
    assert cfg.op.pregrasp.filter.collision.hh_thre == -.002


class FreeObject:
    def __init__(self):
        self._model=mujoco.MjModel.from_xml_string('<mujoco><option gravity="0 0 0" timestep="0.002"/><worldbody><body pos=".1 .2 .3"><freejoint/><geom type="sphere" size=".05" mass="1"/></body></worldbody></mujoco>')
        self._data=mujoco.MjData(self._model);self._obj_active_bodyid=1;self.resets=0
        mujoco.mj_forward(self._model,self._data)
    def reset_qpos(self,q):
        self.resets+=1;mujoco.mj_resetData(self._model,self._data)
        self._data.qpos[:]=q;mujoco.mj_forward(self._model,self._data)
    def run_trajectory(self,*args):pass
    def get_active_obj_pose(self):return np.r_[self._data.xpos[1],self._data.xquat[1]]
    def step_sim(self,substep):
        for _ in range(substep):mujoco.mj_step(self._model,self._data)


def test_real_mujoco_torques_are_pure_signed_rotations_and_reset_each_trial():
    env=FreeObject();q=env._data.qpos.copy();reference=env.get_active_obj_pose().copy()
    result=evaluate_wrenches(env,[q],['ee_pose'],[None],reference,.05)
    assert len(result['trials'])==12 and env.resets==12
    assert not result['force_passed'] and not result['torque_passed']
    assert np.isclose(result['torque_magnitude_nm'],.49)
    for trial in result['trials']:
        assert trial['failure'] in ('translation_limit','rotation_limit')
        if trial['kind']=='torque':
            assert trial['translation_m']<1e-10
            assert trial['rotation_deg']>=15
            axis='XYZ'.index(trial['axis'])
            assert np.asarray(trial['final_qpos'])[4+axis]*trial['sign']>0
        else:
            assert trial['rotation_deg']<1e-8
            assert trial['translation_m']>=.05
    assert not env._data.xfrc_applied.any() and not env._data.qfrc_applied.any()


def test_explicit_wrench_clears_previous_forces():
    env=FreeObject();world_wrench(env._model,env._data,1,[1,2,3],[4,5,6])
    world_wrench(env._model,env._data,1,[0,0,0],[-1,0,0])
    np.testing.assert_array_equal(env._data.xfrc_applied[1],[0,0,0,-1,0,0])
