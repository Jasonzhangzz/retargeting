"""Direct MANO IK must depend on human geometry, never a feasible hand file."""
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from retarget.mano_noise import FINGERS,retarget_from_mano

ROOT=Path(__file__).resolve().parents[1]


def test_direct_reference_is_rigidly_equivariant_without_trajectory_file_reads(monkeypatch):
    model=mujoco.MjModel.from_xml_path(str(ROOT/'data/example/sequence/scene/wuji_right/sugar_box_mesh.xml'))
    data=mujoco.MjData(model)
    for joint in range(26):
        data.qpos[joint]=np.clip(0.,*model.jnt_range[joint])
    mujoco.mj_kinematics(model,data)
    human=np.zeros((21,3))
    for name,(_,robot,start) in FINGERS.items():
        human[start:start+3]=[data.xpos[model.body(f'right_finger{robot}_link{k}').id] for k in (2,3,4)]
        human[start+3]=data.site_xpos[model.site('right_'+name+'_tip').id]
    rotation=Rotation.from_euler('xyz',[.15,-.2,.3]).as_matrix();translation=np.array([.3,-.2,.4])
    transformed=human@rotation.T+translation
    def forbidden(*a,**kw):raise AssertionError('A pure geometry retarget must not load a hand trajectory')
    monkeypatch.setattr(np,'load',forbidden)
    q,report=retarget_from_mano(model,np.stack((human,transformed)))
    assert not report['original_wuji_reference_used'] and not report['object_or_contact_objective_used']
    assert report['chain_direction_rms_max']<.003
    positions=[]
    for frame in q:
        data.qpos[:26]=frame;mujoco.mj_kinematics(model,data)
        positions.append(np.array([data.site_xpos[model.site('right_'+name+'_tip').id] for name in FINGERS]))
    np.testing.assert_allclose(positions[1],positions[0]@rotation.T+translation,atol=2e-5)

