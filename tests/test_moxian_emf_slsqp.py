"""Pose Jacobian, physical extension and persistent EMF IK state contracts."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from moxian_web.emf_slsqp import SkeletonKinematics,rotation_residual_jacobian,SLSQPSkeleton
from moxian_web.skeleton import JOINT_NAMES,validate_pose


def asset(side):
    files=list(Path('.venv-moxian-ik/lib').glob(f'python*/site-packages/qnbot_sdk/glove/_private/algorithms/skeleton/*/share/qnbot_hand/hands/dynamic_openxr_hand/resources/pinocchio/{side}.urdf'))
    if not files:pytest.skip('Local calibrated skeleton resources required')
    return files[0]


@pytest.mark.parametrize('side',['left','right'])
def test_fk_pose_gradient_and_physical_extension(side):
    import xml.etree.ElementTree as ET
    path=asset(side);names=[j.get('name') for j in ET.parse(path).getroot().findall('joint') if j.get('type')!='fixed']
    k=SkeletonKinematics(path,side,names,JOINT_NAMES)
    q=np.clip(k.straight+.1,k.bounds[:,0],k.bounds[:,1])
    for f,ids in enumerate(k.finger_indices):
        tf,jp,jw=k.finger(f,q);target=tf[:3,:3]@Rotation.from_rotvec([.2,-.1,.3]).as_matrix();er,jr=rotation_residual_jacobian(tf[:3,:3],target,jw)
        for i,index in enumerate(ids):
            a=q.copy();b=q.copy();a[index]+=1e-6;b[index]-=1e-6
            ta,_,_=k.finger(f,a);tb,_,_=k.finger(f,b)
            np.testing.assert_allclose(jp[:,i],(ta[:3,3]-tb[:3,3])/2e-6,atol=1e-8)
            ra=Rotation.from_matrix(ta[:3,:3].T@target).as_rotvec();rb=Rotation.from_matrix(tb[:3,:3].T@target).as_rotvec()
            np.testing.assert_allclose(jr[:,i],(ra-rb)/2e-6,atol=1e-8)
    for name in k.extension_limits:
        i=names.index(name)
        assert k.bounds[i,0]<=k.straight[i]<=k.bounds[i,1]
        assert np.isclose(k.straight[i]-k.bounds[i,0],np.deg2rad(20 if name.endswith('_dip_flexion') else 10))


def test_exact_synthetic_pose_stays_still_and_ignores_native_seed():
    import xml.etree.ElementTree as ET
    path=asset('right');names=[j.get('name') for j in ET.parse(path).getroot().findall('joint') if j.get('type')!='fixed']
    k=SkeletonKinematics(path,'right',names,JOINT_NAMES)
    q=k.straight.copy();nodes=k.nodes(q)
    tips=[k.finger(f,q)[0] for f in range(5)]
    class Mapper:
        def step(self,frame):
            return {'pose':{'timestamp_ms':frame['timestamp_ms'],'hand_side':'right','coordinate_frame':'wrist_local','names':JOINT_NAMES,
                            'joint_positions_rad':[1000]*len(q)},
                    'targets':{'positions_local':[t[:3,3].tolist() for t in tips],
                               'rotations_local':[(t[:3,:3]@np.diag([-1.,-1.,1.])).ravel().tolist() for t in tips]}}
    s=SLSQPSkeleton.__new__(SLSQPSkeleton);s.mapper=Mapper();s.kinematics=k;s.joint_names=names;s.joint_limits=dict(zip(names,k.bounds));s.previous=None;s.previous_timestamp=None;s.frames=0;s.mounts=np.tile(np.eye(4),(5,1,1))
    a=s.step({'timestamp_ms':1000});b=s.step({'timestamp_ms':1016})
    validate_pose(b['pose'],'right')
    np.testing.assert_allclose(a['pose']['joint_positions_rad'],q,atol=1e-8)
    np.testing.assert_allclose(b['pose']['joint_positions_rad'],q,atol=1e-8)
    assert not a['ik_diagnostics']['previous_pose_initial_guess']
    assert b['ik_diagnostics']['previous_pose_initial_guess']
    assert not b['ik_diagnostics']['native_angles_used']


@pytest.mark.parametrize('side',['left','right'])
def test_emf_pose_can_recover_small_reverse_thumb_flexion(side):
    import xml.etree.ElementTree as ET
    path=asset(side);names=[j.get('name') for j in ET.parse(path).getroot().findall('joint') if j.get('type')!='fixed']
    k=SkeletonKinematics(path,side,names,JOINT_NAMES);wanted=k.straight.copy()
    for suffix in ('mcp_flexion','ip_flexion'):wanted[names.index(side+'_thumb_'+suffix)]-=np.deg2rad(7)
    tips=[k.finger(f,wanted)[0] for f in range(5)]
    class Mapper:
        def step(self,frame):
            return dict(pose=dict(timestamp_ms=frame['timestamp_ms'],hand_side=side,coordinate_frame='wrist_local',names=JOINT_NAMES),targets=dict(positions_local=[t[:3,3].tolist() for t in tips],rotations_local=[(t[:3,:3]@np.diag([-1.,-1.,1.])).ravel().tolist() for t in tips]))
    s=SLSQPSkeleton.__new__(SLSQPSkeleton);s.mapper=Mapper();s.kinematics=k;s.joint_names=names;s.joint_limits=dict(zip(names,k.bounds));s.previous=None;s.previous_timestamp=None;s.frames=0;s.mounts=np.tile(np.eye(4),(5,1,1))
    result=s.step(dict(timestamp_ms=1000));q=np.array(result['pose']['joint_positions_rad'])
    assert result['ik_diagnostics']['fingers'][0]['position_error_mm']<.5
    assert result['ik_diagnostics']['fingers'][0]['rotation_error_deg']<2
    i=names.index(side+'_thumb_ip_flexion');assert q[i]<k.straight[i]-np.deg2rad(4)


def test_calibrated_bone_lengths_are_used_instead_of_resource_defaults():
    import xml.etree.ElementTree as ET
    path=asset('right');names=[j.get('name') for j in ET.parse(path).getroot().findall('joint') if j.get('type')!='fixed']
    original=SkeletonKinematics(path,'right',names,JOINT_NAMES)
    changed=SkeletonKinematics(path,'right',names,JOINT_NAMES,{'first':{'proximal':.055,'intermediate':.031,'distal':.024}})
    nodes=changed.nodes(changed.straight)
    ids=[JOINT_NAMES.index('index_'+s) for s in ('proximal','intermediate','distal','tip')]
    np.testing.assert_allclose(np.linalg.norm(np.diff(nodes[ids,:3,3],axis=0),axis=-1),[.055,.031,.024],atol=1e-10)
    assert not np.allclose(original.nodes(original.straight)[ids,:3,3],nodes[ids,:3,3])


def test_a_pinky_at_all_flexion_limits_can_return_to_an_observed_straight_hand():
    import xml.etree.ElementTree as ET
    path=asset('right');names=[j.get('name') for j in ET.parse(path).getroot().findall('joint') if j.get('type')!='fixed']
    k=SkeletonKinematics(path,'right',names,JOINT_NAMES);tips=[k.finger(f,k.straight)[0] for f in range(5)]
    class Mapper:
        def step(self,frame):
            return dict(pose=dict(timestamp_ms=frame['timestamp_ms'],hand_side='right',coordinate_frame='wrist_local',names=JOINT_NAMES),
                        targets=dict(positions_local=[t[:3,3].tolist() for t in tips],
                                     rotations_local=[(t[:3,:3]@np.diag([-1.,-1.,1.])).ravel().tolist() for t in tips]))
    s=SLSQPSkeleton.__new__(SLSQPSkeleton);s.mapper=Mapper();s.kinematics=k;s.joint_names=names
    s.joint_limits=dict(zip(names,k.bounds));s.previous=k.straight.copy();s.previous[-3:]=k.bounds[-3:,1]
    s.previous_timestamp=1000;s.frames=1;s.mounts=np.tile(np.eye(4),(5,1,1))
    result=s.step(dict(timestamp_ms=1016));pinky=result['ik_diagnostics']['fingers'][-1]
    assert pinky['observation_reseeded'] and pinky['tracking_satisfied']
    assert pinky['position_error_mm']<.01
    np.testing.assert_allclose(result['pose']['joint_positions_rad'][-3:],k.straight[-3:],atol=1e-5)


def test_offline_clip_cannot_mix_contact_geometry_across_pose_calibrations():
    from moxian_web.card_offline import prepare_clip
    frames=[{'skeleton_runtime_revision':'ik','pose_calibration_version':v} for v in ['before','after']]
    with pytest.raises(ValueError,match='分段重算'):prepare_clip(None,frames)
