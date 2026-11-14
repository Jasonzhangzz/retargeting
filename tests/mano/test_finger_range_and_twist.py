"""DIP physical straight and EMF-driven root torsion across fit/IK/skin."""
import copy,json
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from moxian_web.qnbot_surface import QnbotSurface
from moxian_web.bone_extrinsics import definition_from_surface,model_hash,validate_candidate,BoneKinematics
from moxian_web.emf_qp import QPSkeleton
from moxian_web.moxian_six_calibration import SixObjective,training_rows
from moxian_web.bone_calibration import frozen_contacts
from moxian_web.contact import default_mapping
from moxian_web.wuji_native_abi import SDK_POSES,FINGERS
from moxian_web.wuji_native_capture import PROTOCOL


def candidate(m):
    value=m.artifact();value.update(serial='TEST',calibration_fingerprint='test',bone_extrinsics=definition_from_surface(m),delta_policy=None)
    value['model_hash']=model_hash(value);validate_candidate(value);return value


def frame(c,q,stamp):
    k=BoneKinematics(c['bone_extrinsics']);a=k.alignment;nodes={}
    for i,f in enumerate(FINGERS):
        r,p=k.sensor(i,q);nodes[f]=dict(position=(a[:3,:3].T@(p-a[:3,3])).tolist(),quaternion_xyzw=Rotation.from_matrix(a[:3,:3].T@r).as_quat().tolist())
    return dict(timestamp_ms=stamp,fingers=nodes)


@pytest.fixture(params=['left','right'])
def model(request):
    torch.set_num_threads(1);m=QnbotSurface(request.param,finger_twist=True).double();m.bind_mapping(dict(assignments=default_mapping(m.side),confirmed=[]))
    for p in m.parameters():p.requires_grad_(False)
    with torch.no_grad():m.mount_p0[:,0]=-.012;m.mount_p0[:,1]=-.004
    return m


def test_physical_dip_bounds_and_unchanged_native_assets(model):
    m=model;base=QnbotSurface(m.side).double();c=candidate(m);k=BoneKinematics(c['bone_extrinsics'])
    assert len(m.joint_names)==29 and len(k.joint_names)==29
    for f in ('index','middle','ring','little'):
        i=m.joint_names.index(m.side+'_'+f+'_dip_flexion');straight=m.finger_straight[m.joint_names[i]]
        assert float(m.limits[i,0])==pytest.approx(straight-np.deg2rad(20),abs=1e-7)
        assert k.limits[i,0]==pytest.approx(float(m.limits[i,0]))
        assert c['bone_extrinsics']['rest_angles_rad'][i]==pytest.approx(straight)
    torch.testing.assert_close(m.body_p0,base.body_p0);torch.testing.assert_close(m.body_r0,base.body_r0)
    torch.testing.assert_close(m.vertices0,base.vertices0);torch.testing.assert_close(m.weights,base.weights)
    torch.testing.assert_close(m.joint_axis0[:25],base.joint_axis0)
    with torch.no_grad():torch.testing.assert_close(m(m.q0[None])[0],base(base.q0[None])[0],atol=1e-12,rtol=1e-12)


def test_reverse_dip_and_root_twist_fk_skin_jacobian_and_online_ik(model):
    m=model;c=candidate(m);k=BoneKinematics(c['bone_extrinsics']);q=np.array(c['bone_extrinsics']['rest_angles_rad'])
    for f in ('index','middle','ring','little'):q[m.joint_names.index(m.side+'_'+f+'_dip_flexion')]-=np.deg2rad(15)
    q[25:]=np.deg2rad([24,-28,21,-25]);r,p=k.forward(q)
    with torch.no_grad():v,pj,rj=m(torch.tensor(q[None]));sp,sr=m.sensor_prediction(torch.tensor(q[None]));plain=q.copy();plain[25:]=0;vplain=m(torch.tensor(plain[None]))[0]
    assert float((v-vplain).abs().max())>.001
    np.testing.assert_allclose(p,pj[0],atol=1e-10);np.testing.assert_allclose(r,rj[0],atol=1e-10)
    for f in range(1,5):
        rr,pp,jp,jw=k.sensor(f,q,True);np.testing.assert_allclose(pp,sp[0,f],atol=1e-10);np.testing.assert_allclose(rr,sr[0,f],atol=1e-10)
        col=list(k.finger_indices[f]).index(24+f);delta=np.zeros(29);delta[24+f]=1e-6
        rp,pp=k.sensor(f,q+delta);rm,pm=k.sensor(f,q-delta)
        np.testing.assert_allclose(jp[:,col],(pp-pm)/2e-6,atol=1e-8)
        np.testing.assert_allclose(jw[:,col],Rotation.from_matrix(rp@rm.T).as_rotvec()/2e-6,atol=1e-8)
    engine=QPSkeleton(c)
    for i in range(30):result=engine.step(frame(c,q,1000+i*20))
    errors=result['ik_diagnostics']['fingers'][1:]
    assert max(v['position_error_mm'] for v in errors)<1.5
    assert max(v['rotation_error_deg'] for v in errors)<2
    found=np.array(result['pose']['joint_positions_rad'])
    for f in ('index','middle','ring','little'):
        i=m.joint_names.index(m.side+'_'+f+'_dip_flexion');assert found[i]<m.finger_straight[m.joint_names[i]]-np.deg2rad(7)
    assert np.abs(found[25:]).min()>np.deg2rad(10)
    json.dumps(result,allow_nan=False)


def test_root_rotation_and_single_frame_spikes_obey_physical_bounds(model):
    c=candidate(model);engine=QPSkeleton(c);rest=np.array(c['bone_extrinsics']['rest_angles_rad'])
    engine.step(frame(c,rest,1000))
    target=rest.copy();target[25:]=np.deg2rad(100)
    # A long-running nonlinear solve and every retry share ONE frame budget.
    trace=[rest.copy()]
    for i,dt in enumerate((1.,10.,20.,50.,16.)):
        stamp=engine.previous_timestamp+dt
        result=engine.step(frame(c,target,stamp));trace.append(np.array(result['pose']['joint_positions_rad']))
        assert np.max(abs(trace[-1]-trace[-2]))<=c['bone_extrinsics']['solver_config']['v_max']*dt/1000+1e-10
        assert np.abs(trace[-1][25:]).max()<=np.deg2rad(30)+1e-7
    # A corrupt packet does not discard the valid history and cold-reseed.
    previous=engine.previous.copy();bad=frame(c,target,engine.previous_timestamp+10)
    bad['fingers']['thumb']['quaternion_xyzw']=[0,0,0,0]
    with pytest.raises(ValueError,match='无效EMF'):engine.step(bad)
    np.testing.assert_array_equal(engine.previous,previous)
    result=engine.step(frame(c,rest,engine.previous_timestamp+20))
    assert result['continuity']['reused_engine']
    assert result['continuity']['max_joint_step_deg']<=7.2+1e-7


def test_225_coordinate_fit_bounds_gradient_and_25_29_artifact_switch(model):
    m=model;c=candidate(m);q=np.array(c['bone_extrinsics']['rest_angles_rad']);q[25:]=[.1,-.1,.05,-.05]
    frames={name:[[frame(c,q,1000)['fingers'][f] for f in FINGERS]] for name in SDK_POSES}
    _,rows=training_rows(dict(protocol=PROTOCOL,completed_groups=list(SDK_POSES),state=dict(state='complete'),pose_frames=frames))
    for row in rows:row['skeleton']=dict(joint_names=m.joint_names,joint_positions_rad=q.tolist())
    obj=SixObjective(m,rows,frozen_contacts(m,rows),dict(position_sigma_m=.003,rotation_sigma_rad=.15));lo,hi=obj.bounds()
    assert len(lo)==225 and (lo[51:].reshape(6,29)[:,25:]<-.5).all()
    x=torch.tensor(np.r_[np.zeros(51),np.tile(q,6)],requires_grad=True);loss,_=obj.loss(x);g=torch.autograd.grad(loss,x)[0]
    for i in (51+9,51+25,51+28,51+29+25):
        delta=torch.zeros_like(x);delta[i]=1e-6;numeric=(obj.loss(x.detach()+delta)[0]-obj.loss(x.detach()-delta)[0])/2e-6
        assert float(g[i])==pytest.approx(float(numeric),rel=2e-4,abs=.003)
    base=QnbotSurface(m.side).double();old=candidate(base)
    m.load_artifact(old);assert len(m.joint_names)==25;torch.testing.assert_close(m.limits,base.limits)
    m.load_artifact(c);assert len(m.joint_names)==29;np.testing.assert_allclose(m.limits,c['bone_extrinsics']['rig']['limits'])


def test_historical_25_dof_qp_config_remains_loadable_without_new_optional_weight():
    m=QnbotSurface('right').double();c=candidate(m)
    for key in ('root_twist_posture_scale','thumb_palm_posture_weight','thumb_palm_history_weight'):
        del c['bone_extrinsics']['solver_config'][key]
    c['model_hash']=model_hash(c)
    validate_candidate(c);engine=QPSkeleton(c)
    result=engine.step(frame(c,np.array(c['bone_extrinsics']['rest_angles_rad']),1000))
    assert len(result['pose']['joint_positions_rad'])==25


def test_flat_pose_anchors_thumb_and_palm_branch_has_gradient(model):
    m=model;c=candidate(m);q=np.array(c['bone_extrinsics']['rest_angles_rad'])
    frames={name:[[frame(c,q,1000)['fingers'][f] for f in FINGERS]] for name in SDK_POSES}
    _,rows=training_rows(dict(protocol=PROTOCOL,completed_groups=list(SDK_POSES),state=dict(state='complete'),pose_frames=frames))
    for row in rows:row['skeleton']=dict(joint_names=m.joint_names,joint_positions_rad=q.tolist())
    obj=SixObjective(m,rows,frozen_contacts(m,rows),dict(position_sigma_m=.003,rotation_sigma_rad=.15))
    lo,hi=obj.bounds();lower=lo[51:].reshape(6,29);upper=hi[51:].reshape(6,29)
    hinge=[m.joint_names.index(n) for n in m.thumb_straight]
    np.testing.assert_allclose(lower[-1,hinge],q[hinge]-np.deg2rad(5),atol=1e-7)
    np.testing.assert_allclose(upper[-1,hinge],q[hinge]+np.deg2rad(5),atol=1e-7)
    x=torch.tensor(np.r_[np.zeros(51),np.tile(q,6)],requires_grad=True)
    _,terms=obj.loss(x);assert terms['flat_thumb_straight']==0 and terms['thumb_palm_branch']==0
    altered=x.detach().clone();i=m.joint_names.index(m.side+'_thumb_cmc_flexion');altered[51+i]+=.3;altered.requires_grad_()
    _,terms=obj.loss(altered);assert terms['thumb_palm_branch']>.7
    g=torch.autograd.grad(terms['thumb_palm_branch'],altered)[0]
    assert g[51+i]>0 and g[51+5*29+i]<0
    # A flat thumb bend is explicitly penalized, even if its fitted RX mount
    # could otherwise absorb that rotation.
    altered=x.detach().clone();altered[51+5*29+hinge[0]]+=np.deg2rad(15)
    assert obj.loss(altered)[1]['flat_thumb_straight']==pytest.approx(25)


def test_thumb_palm_ambiguity_resolves_toward_flat_reference(model):
    c=candidate(model);engine=QPSkeleton(c);rest=np.array(c['bone_extrinsics']['rest_angles_rad'])
    # Start in a redundant palm rotation combination, then observe a static
    # flat thumb with small EMF jitter. It must recover instead of extrapolate.
    start=rest.copy();ids=[model.joint_names.index(model.side+'_thumb_'+n) for n in ('cmc_flexion','mcp_abduction')]
    start[ids]=[.25,-.25];engine.previous=start.copy();engine.saved=start.copy();engine.previous_timestamp=1000
    errors=[]
    for i in range(35):
        row=frame(c,rest,1020+i*20)
        row['fingers']['thumb']['position'][0]+=.0003*(-1 if i%2 else 1)
        result=engine.step(row);errors.append(np.linalg.norm(engine.previous[ids]-rest[ids]))
        assert result['continuity']['max_joint_step_deg']<=7.2+1e-7
    assert errors[-1]<.03 and errors[-1]<errors[0]*.2
    assert result['ik_diagnostics']['fingers'][0]['position_error_mm']<1
