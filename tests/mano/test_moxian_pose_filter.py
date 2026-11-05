import numpy as np
from scipy.spatial.transform import Rotation
from moxian_web.pose_filter import SensorPoseFilter

def poses(p=0.,angle=0.):
    return np.tile([p,0.,0.],(5,1)),Rotation.from_euler('z',[angle]*5).as_matrix()

def test_isolated_pose_spike_is_rejected_without_mutating_input():
    f=SensorPoseFilter();p,r=poses();f.update(p,r,1000)
    p,r=poses(.03,.5);before=p.copy();q,s,quality=f.update(p,r,1033)
    assert np.array_equal(p,before) and np.linalg.norm(q)<1e-9
    assert len(quality['rejected_fingers'])==5
    q,s,quality=f.update(*poses(),1066);assert np.linalg.norm(q)<1e-9

def test_persistent_motion_is_followed_within_two_input_frames():
    f=SensorPoseFilter();f.update(*poses(),1000)
    f.update(*poses(.03,.5),1033)
    p,r,quality=f.update(*poses(.03,.5),1066)
    assert np.max(abs(p[:,0]-.03))<.001
    assert np.rad2deg(Rotation.from_matrix(r).magnitude()).min()>27
    assert not quality['rejected_fingers']

def test_oscillation_is_damped_but_continuous_ramp_is_not_frozen():
    f=SensorPoseFilter();p=[];a=[]
    for i in range(100):
        x=.004*(-1)**i;q=.08*(-1)**i
        out,rot,_=f.update(*poses(x,q),1000+i*25);p.append(out[0,0]);a.append(Rotation.from_matrix(rot[0]).as_rotvec()[2])
    assert np.std(p[30:])<.004*.6 and np.std(a[30:])<.08*.6
    f=SensorPoseFilter()
    for i in range(20):
        target=.003*i;out,rot,_=f.update(*poses(target,.1*i),1000+i*25)
    assert abs(out[0,0]-target)<.004
