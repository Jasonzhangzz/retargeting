from pathlib import Path
import copy
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from moxian_web.glove_geometry import COUNT,validate,export_skeleton,skeleton_urdf
from moxian_web.hand_model import DEFAULT_MANO
from moxian_web.mano_runtime import ManoSurface
from moxian_web.official_fk import official_urdf,OfficialSkeletonFK
from moxian_web.emf_slsqp import SkeletonKinematics,SLSQPSkeleton
from moxian_web.skeleton import JOINT_NAMES
from moxian_web.contact import default_mapping
from scripts.calibrate_wuji_mano import CHAINS,TIPS


@pytest.fixture(params=['left','right'])
def model(request):
    path=DEFAULT_MANO.with_name('MANO_'+request.param.upper()+'.pkl')
    if not path.exists():pytest.skip('Licensed MANO assets required')
    torch.set_num_threads(1)
    m=ManoSurface(path,device='cpu',side=request.param);m.enable_official_fk()
    m.build_binding(dict(version='test',assignments=default_mapping(request.param),confirmed=[]))
    return m


def test_zero_geometry_preserves_upstream_and_nonzero_geometry_has_gradients(model):
    beta=torch.zeros(10);rot=torch.eye(3).expand(1,15,3,3)
    baseline=model.model.official_matrix_forward(beta,rot)
    p=torch.zeros(COUNT,requires_grad=True)
    actual=model.model.official_matrix_forward(beta,rot,p)
    for a,b in zip(actual,baseline):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-5)
    def evaluate(x):
        v,j,_=model.model.official_matrix_forward(beta,rot,x)
        return v.double().square().sum()+j.double().square().sum()
    gradient=torch.autograd.grad(evaluate(p),p)[0]
    assert torch.isfinite(gradient).all()
    for i in (0,15,20,25,34):
        d=torch.zeros(COUNT);d[i]=.001
        fd=(evaluate(p+d)-evaluate(p-d))/.002
        torch.testing.assert_close(gradient[i].double(),fd,atol=2e-5,rtol=.01)
    with pytest.raises(ValueError):validate([float('nan')]*COUNT)


def test_exported_skeleton_and_skin_share_joint_centres_and_sensor_frames(model,tmp_path):
    import xml.etree.ElementTree as ET
    source=official_urdf(model.side)
    if model.side=='left':
        tree=ET.parse(source)
        for joint in tree.getroot().findall('joint'):
            if joint.get('name') in ('left_thumb_mcp_abduction','left_thumb_ip_flexion'):joint.find('origin').set('rpy','0 0 0')
        source=tmp_path/'straight-zero-source.urdf';tree.write(source,encoding='unicode')
    from moxian_web.official_fk import urdf_reference
    reference={**urdf_reference(source,model.side),'hand_side':model.side,'coordinate_frame':'wrist_local','source_urdf_path':str(source)}
    model.official_fk.parameters(model,dict(timestamp_ms=0.,skeleton=reference),CHAINS)
    assert model.official_fk.path==source
    p=np.zeros(COUNT);p[0]=.1;p[4]=-.1;p[15]=.1;p[20]=-.1;p[25]=.1;p[37]=.03
    model.set_glove_geometry(p)
    payload=export_skeleton(model)
    before=source.read_bytes()
    path=skeleton_urdf(payload,source,tmp_path/'personalized.urdf')
    names=[j.get('name') for j in ET.parse(path).getroot().findall('joint') if j.get('type')!='fixed']
    k=SkeletonKinematics(path,model.side,names,JOINT_NAMES)
    nodes=k.nodes(np.zeros(len(names)))
    for f,c in enumerate(CHAINS):
        finger=('thumb','index','middle','ring','little')[f]
        segments=('metacarpal','proximal','distal') if f==0 else ('proximal','intermediate','distal')
        indices=[JOINT_NAMES.index(finger+'_'+s) for s in segments]
        np.testing.assert_allclose(nodes[indices,:3,3],np.asarray(payload['joints_m'])[list(c)],atol=1e-7)
        np.testing.assert_allclose(nodes[JOINT_NAMES.index(finger+'_tip'),:3,3],payload['tips_m'][f],atol=1e-7)
    model.official_fk=OfficialSkeletonFK(path,model.side,payload)
    engine=SLSQPSkeleton.__new__(SLSQPSkeleton);engine.kinematics=k;engine.personalized=payload;engine.side=model.side
    q=np.clip(k.straight+.12,k.bounds[:,0],k.bounds[:,1]);nodes=k.nodes(q)
    frame=dict(timestamp_ms=1.,skeleton=dict(timestamp_ms=1.,hand_side=model.side,coordinate_frame='wrist_local',names=JOINT_NAMES,
               positions_local=nodes[:,:3,3].tolist(),orientations_local_xyzw=Rotation.from_matrix(nodes[:,:3,:3]).as_quat().tolist(),
               personalized_model={**payload,'urdf_path':str(path)}))
    pose,root,*_=model.official_fk.parameters(model,frame,CHAINS)
    with torch.no_grad():v,j,r,_=model.decode(torch.tensor(pose[None]),torch.tensor(root[None]))
    transform=np.asarray(payload['mano_to_wrist_rotation'])
    for f,c in enumerate(CHAINS):
        prediction=(j[0,c[-1]].numpy()+r[0,c[-1]].numpy()@model.sensor_offsets()[f])@transform.T
        np.testing.assert_allclose(engine.effector(f,q)[0][:3,3],prediction,atol=2e-7)
        _,jac,_=engine.effector(f,q)
        for col,index in enumerate(k.finger_indices[f]):
            a=q.copy();b=q.copy();a[index]+=.000001;b[index]-=.000001
            fd=(engine.effector(f,a)[0][:3,3]-engine.effector(f,b)[0][:3,3])/.000002
            np.testing.assert_allclose(jac[:,col],fd,atol=1e-8)
    assert not engine.refine_contact_pair(dict(timestamp_ms=1.,tactile_timestamp_ms=100.,contact=dict(ready=True,patches=[])),q,np.zeros((5,3)),np.tile(np.eye(3),(5,1,1)),q)['applied']
    assert source.read_bytes()==before
    model.set_shape(np.zeros(10));assert np.array_equal(model.glove_parameters,p)
    baseline=OfficialSkeletonFK(side=model.side)
    reference={**baseline.reference,'timestamp_ms':1.}
    model.official_fk.parameters(model,dict(timestamp_ms=1.,skeleton=reference),CHAINS)
    assert model.official_fk.fixed_binding is None and model.official_fk.path==baseline.path


def test_motion_optimizer_freezes_shape_before_size_phase_and_reports_weak_axes(model):
    import threading
    from moxian_web.mano_objective import ManoObjective,SolverSettings,solve_calibration
    rows=[dict(stage=stage,timestamp_ms=float(i),capture_protocol='geort_motion_v2',
               fingers={f:dict(position=[.01,.02,.03],quaternion_xyzw=[0,0,0,1]) for f in ('thumb','index','middle','ring','pinky')})
          for i,stage in enumerate(('open','pinch_index'))]
    model.flat_reference={'reference_id':'synthetic'}
    objective=ManoObjective(model,rows,np.zeros((2,15,3)),np.tile(np.eye(3),(2,1,1)),training=True,shared=True,tactile=False,
                            settings=SolverSettings(adam_steps=2,lm_steps=0))
    report=solve_calibration(objective,threading.Event(),lambda _:None,max_seconds=30)
    assert report['parameter_snapshot']['shared_size']==89
    assert report['observability']['rank']<43
    assert set(range(37,43))<=set(report['observability']['frozen_coordinates'])
    np.testing.assert_allclose(report['parameter_snapshot']['vector'][:10],report['phases'][0]['parameter_vector'][:10],atol=0)
    assert np.isfinite(report['parameter_snapshot']['vector']).all()


def test_approach_and_release_never_create_training_contact_equalities(model):
    from moxian_web.tactile_pose import frame_constraints
    row=dict(timestamp_ms=1000.,tactile_timestamp_ms=1000.,stage='pinch_index',capture_protocol='geort_motion_v2',
             contact=dict(ready=True,patches=[]),contact_confirmed=False)
    ordered=np.concatenate((model.rest_j_np,model.rest_np[list(TIPS)]))[model.ordered]
    for phase in ('apart','approach','release'):
        row['motion_phase']=phase
        assert frame_constraints(model,row,model.rest_np,ordered,guided=True)==([],[])
    row.update(motion_phase='touch',contact_confirmed=True)
    anchors,pairs=frame_constraints(model,row,model.rest_np,ordered,guided=True)
    assert len(pairs)==1 and pairs[0]['evidence']=='confirmed_calibration_gesture'
