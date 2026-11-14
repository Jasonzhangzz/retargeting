"""Independent native FK, 51-coordinate fitting, and shared runtime contracts."""
import copy
import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from moxian_web.bone_extrinsics import (FINGERS,BoneKinematics,definition_from_surface,
    model_hash,segment_indices,publish_active,read_active,validate_candidate)
from moxian_web.qnbot_surface import QnbotSurface
from moxian_web.bone_calibration import Coordinates,Objective,noise_from_training,replay
from moxian_web.emf_qp import QPSkeleton,box_qp,rotation_error_jacobian


@pytest.fixture(params=['left','right'])
def model(request):
    torch.set_num_threads(1)
    return QnbotSurface(request.param).double()


def artifact(m):
    value=m.artifact();value.update(serial='test',calibration_fingerprint='eight',
        bone_extrinsics=definition_from_surface(m),delta_policy=None)
    value['model_hash']=model_hash(value)
    return value


def frame(candidate,q,stamp=1000):
    k=BoneKinematics(candidate['bone_extrinsics']);a=k.alignment;fingers={}
    for i,f in enumerate(FINGERS):
        r,p=k.sensor(i,q)
        fingers[f]=dict(position=(a[:3,:3].T@(p-a[:3,3])).tolist(),
            quaternion_xyzw=Rotation.from_matrix(a[:3,:3].T@r).as_quat().tolist())
    return dict(timestamp_ms=stamp,fingers=fingers,stage='synthetic',calibration_fingerprint=candidate.get('calibration_fingerprint'),
        skeleton=dict(joint_names=k.joint_names,joint_positions_rad=list(q)))


def test_native_fk_and_sensor_jacobian_match_torch(model):
    m=model
    with torch.no_grad():
        m.geometry['length'][segment_indices(m)]=torch.linspace(-.08,.08,15,dtype=torch.float64)
        m.geometry['mount_pos'].fill_(.1);m.geometry['mount_rot'].fill_(.05)
    candidate=artifact(m);validate_candidate(candidate);k=BoneKinematics(candidate['bone_extrinsics'])
    q=np.clip(np.asarray(candidate['bone_extrinsics']['rest_angles_rad'])+.12,k.limits[:,0],k.limits[:,1])
    r,p=k.forward(q)
    with torch.no_grad():_,pj,rj=m(torch.tensor(q[None]));sp,sr=m.sensor_prediction(torch.tensor(q[None]))
    np.testing.assert_allclose(p,pj[0],atol=1e-10);np.testing.assert_allclose(r,rj[0],atol=1e-10)
    for f,indices in enumerate(k.finger_indices):
        rr,pp,jp,jw=k.sensor(f,q,True)
        np.testing.assert_allclose(pp,sp[0,f],atol=1e-10);np.testing.assert_allclose(rr,sr[0,f],atol=1e-10)
        target=rr@Rotation.from_rotvec([.1,-.2,.15]).as_matrix();_,jr=rotation_error_jacobian(rr,target,jw)
        for i,index in enumerate(indices):
            plus=q.copy();minus=q.copy();plus[index]+=1e-6;minus[index]-=1e-6
            ra,pa=k.sensor(f,plus);rb,pb=k.sensor(f,minus)
            np.testing.assert_allclose(jp[:,i],(pa-pb)/2e-6,atol=1e-8)
            # Native matrices originate in float32; SciPy additionally projects
            # them onto SO(3), leaving roughly 2e-8 derivative roundoff.
            np.testing.assert_allclose(jr[:,i],(Rotation.from_matrix(ra.T@target).as_rotvec()-Rotation.from_matrix(rb.T@target).as_rotvec())/2e-6,atol=1e-7)


def test_all_fifteen_lengths_drive_fk_and_skin_without_other_parameters(model):
    m=model;c=Coordinates(m);q=m.q0[None]
    with torch.no_grad():v0,j0,_=m(q)
    for index,bone in enumerate(segment_indices(m)):
        x=torch.zeros(51,dtype=torch.float64);x[index]=.08;g=c.geometry(x)
        with torch.no_grad():v,j,_=m(q,geometry=g)
        f=index//3
        assert float((j[:,m.tip_bodies[f]]-j0[:,m.tip_bodies[f]]).norm())>.0005
        assert float((v-v0).norm(dim=-1).max())>.0005
        others=[body for i,body in enumerate(m.tip_bodies) if i!=f]
        torch.testing.assert_close(j[:,others],j0[:,others])
        for key in ('width','depth','root','axis','residual'):assert not bool(g[key].any())
    # In particular the terminal bone has no SKN tip bone; longitudinal skin
    # derivation must still move its physical surface endpoint.
    assert len(c.ids)==15 and c.bounds()[0].shape==(51,)


def test_joint_objective_gradients_and_nlopt_synthetic_fit(model):
    from moxian_web.wuji_calibration_solver import WujiCalibrationLbfgs
    m=model;base=artifact(m);k=BoneKinematics(base['bone_extrinsics']);rows=[]
    for n in range(3):
        q=np.clip(np.asarray(base['bone_extrinsics']['rest_angles_rad'])+.12+n*.15,k.limits[:,0],k.limits[:,1])
        rows.append(frame(base,q,1000+n*1000))
    obj=Objective(m,rows,{},dict(position_sigma_m=.003,rotation_sigma_rad=.15))
    truth=np.zeros(51);truth[3]=.07;truth[18]=.3;truth[45]=.035;truth[48]=.2
    with torch.no_grad():
        p,r=m.sensor_prediction(obj.q0,obj.coordinates.geometry(torch.tensor(truth)))
        a,t=m.alignment(obj.coordinates.geometry(torch.tensor(truth)))
        obj.p=(p-t)@a;obj.r=a.T@r
    initial=np.r_[np.zeros(51),obj.q0.numpy().ravel()]
    x=torch.tensor(initial,requires_grad=True);loss,_=obj.loss(x);loss.backward()
    assert torch.isfinite(x.grad).all()
    for i in (3,18,32,45,48,52):
        plus=initial.copy();minus=initial.copy();plus[i]+=1e-6;minus[i]-=1e-6
        fd=(float(obj.loss(torch.tensor(plus))[0].detach())-float(obj.loss(torch.tensor(minus))[0].detach()))/2e-6
        assert float(x.grad[i])==pytest.approx(fd,rel=2e-4,abs=1e-5)
    lo,hi=obj.bounds();lo[51:]=initial[51:];hi[51:]=initial[51:]
    opt=WujiCalibrationLbfgs(len(initial));opt.set_lower_bounds(lo);opt.set_upper_bounds(hi);opt.set_maxeval(60)
    def fun(v,g):
        xx=torch.tensor(v,requires_grad=True);value,_=obj.loss(xx);value.backward()
        if g.size:g[:]=xx.grad.numpy()
        return float(value.detach())
    opt.set_min_objective(fun);result=opt.optimize(initial)
    assert fun(result,np.array([]))<float(loss.detach())*.1
    assert opt.provenance['native_solver_reused'] and opt.provenance['nlopt_version']=='2.9.1'


def test_qp_exact_pose_reset_and_no_action_labels(model):
    candidate=artifact(model);engine=QPSkeleton(candidate);q=np.asarray(candidate['bone_extrinsics']['rest_angles_rad'])
    a=frame(candidate,q);first=engine.step(a)
    second=engine.step(frame(candidate,q,1020))
    assert max(r['position_error_mm'] for r in first['ik_diagnostics']['fingers'])<1e-6
    np.testing.assert_allclose(second['pose']['joint_positions_rad'],q,atol=1e-8)
    assert second['ik_diagnostics']['previous_pose_initial_guess']
    assert not engine.step(frame(candidate,q,1400))['ik_diagnostics']['previous_pose_initial_guess']
    with pytest.raises(ValueError,match='递增'):engine.step(frame(candidate,q,1400))
    alternate=copy.deepcopy(a);alternate.update(stage='contact_fist',contact_confirmed=True,partition='validation')
    import threading
    one=replay(model,candidate,[a],threading.Event());two=replay(model,candidate,[alternate],threading.Event())
    assert one[0]['pose']==two[0]['pose']


def test_native_studio_qp_unreachable_observation_keeps_calibration_and_frame_motion_budget(model):
    from moxian_web.studio_emf_solver import StudioQPSkeleton
    candidate=artifact(model);before=json.dumps(candidate,sort_keys=True);engine=StudioQPSkeleton(candidate)
    rest=np.asarray(candidate['bone_extrinsics']['rest_angles_rad']);previous=None
    for i in range(5):
        observation=frame(candidate,rest,1000+i*16)
        observation['fingers']['thumb']['position'][0]+=.08
        result=engine.step(observation);q=np.asarray(result['pose']['joint_positions_rad'])
        assert np.isfinite(q).all()
        assert result['ik_diagnostics']['native_solver'] and not result['ik_diagnostics']['official_sdk_binary']
        assert not result['ik_diagnostics']['fingers'][0]['tracking_satisfied']
        if previous is not None:assert abs(q-previous).max()<=engine.config['v_max']*.016+1e-9
        previous=q
    assert engine.policy()['solver_dof']==len(rest)
    assert json.dumps(candidate,sort_keys=True)==before


@pytest.mark.parametrize('twist',[False,True])
def test_compiled_studio_fk_and_sensor_jacobians_on_25_and_29_dof(model,twist):
    from moxian_web.studio_emf_solver import NativeStudioQP
    m=QnbotSurface(model.side,finger_twist=twist).double();candidate=artifact(m)
    native=NativeStudioQP(candidate['bone_extrinsics']);reference=BoneKinematics(candidate['bone_extrinsics'])
    q=np.clip(np.asarray(candidate['bone_extrinsics']['rest_angles_rad'])+.15,reference.limits[:,0],reference.limits[:,1])
    assert native.nq==(29 if twist else 25)
    r,p=native.forward(q);rr,pp=reference.forward(q)
    np.testing.assert_allclose(r,rr,atol=1e-12);np.testing.assert_allclose(p,pp,atol=1e-12)
    for f in range(5):
        actual=native.sensor(f,q);expected=reference.sensor(f,q,True)
        for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=1e-12)
        for j,index in enumerate(reference.finger_indices[f]):
            delta=np.eye(len(q))[index]*1e-6
            fd=(native.sensor(f,q+delta)[1]-native.sensor(f,q-delta)[1])/(2e-6)
            # Rest rotations originate in Float32 SKN data, as in the Python
            # reference's independent Jacobian check above.
            np.testing.assert_allclose(actual[2][:,j],fd,atol=1e-8)


def test_compiled_studio_qp_matches_reference_tracking_cost_and_branch(model):
    from moxian_web.studio_emf_solver import StudioQPSkeleton
    candidate=artifact(model);reference=QPSkeleton(candidate);native=StudioQPSkeleton(candidate)
    q=np.clip(np.asarray(candidate['bone_extrinsics']['rest_angles_rad'])+.16,native.kinematics.limits[:,0],native.kinematics.limits[:,1])
    for i in range(8):
        observation=frame(candidate,q,1000+16*i);a=native.step(observation);b=reference.step(observation)
        np.testing.assert_allclose(a['pose']['joint_positions_rad'],b['pose']['joint_positions_rad'],atol=2e-6)
        assert a['pose']['model_hash']==b['pose']['model_hash']


def test_atomic_selection_rejects_tamper_and_roundtrips_in_isolated_sdk(model,tmp_path):
    value=artifact(model);publish_active(tmp_path,value)
    loaded=read_active(tmp_path,serial='test',side=model.side,fingerprint='eight')
    assert loaded['model_hash']==value['model_hash']
    bad=copy.deepcopy(value);bad['geometry']['width'][1]=.01;bad['model_hash']=model_hash(bad)
    with pytest.raises(ValueError,match='width'):publish_active(tmp_path,bad)
    assert read_active(tmp_path)['model_hash']==value['model_hash']
    with torch.no_grad():model.geometry['length'][segment_indices(model)[3]]=.04
    second=artifact(model);publish_active(tmp_path,second)
    assert read_active(tmp_path)['model_hash']==second['model_hash']!=value['model_hash']
    q=np.asarray(value['bone_extrinsics']['rest_angles_rad'])
    switched=QPSkeleton(read_active(tmp_path)).step(frame(value,q))
    assert not switched['ik_diagnostics']['previous_pose_initial_guess']
    publish_active(tmp_path,value)
    assert read_active(tmp_path)['model_hash']==value['model_hash']
    python=Path('.venv-moxian-ik/bin/python')
    if python.exists():
        root=str(Path.cwd());command="import sys;sys.path.insert(0,sys.argv[1]);from moxian_web.bone_extrinsics import read_active;from moxian_web.emf_qp import QPSkeleton;m=read_active(sys.argv[2]);QPSkeleton(m);assert 'torch' not in sys.modules"
        subprocess.run([str(python),'-I','-c',command,root,str(tmp_path)],check=True,capture_output=True,text=True)
    publish_active(tmp_path,None);assert read_active(tmp_path) is None


def test_small_box_qp_matches_independent_slsqp():
    rng=np.random.default_rng(37)
    for n in range(1,7):
        for _ in range(8):
            a=rng.normal(size=(n,n));h=a.T@a+.5*np.eye(n);g=rng.normal(size=n)
            lo=np.full(n,-.2);hi=np.full(n,.3);x=box_qp(h,g,lo,hi)
            reference=minimize(lambda v:(.5*v@h@v+g@v,h@v+g),np.zeros(n),jac=True,
                bounds=list(zip(lo,hi)),method='SLSQP',options={'ftol':1e-12,'maxiter':100})
            assert .5*x@h@x+g@x==pytest.approx(reference.fun,abs=2e-8)


def test_noise_uses_accepted_training_only():
    raw=dict(static_accepted=True,static_key='train:a:target',partition='train',
        fingers={f:dict(position=[0,0,0],quaternion_xyzw=[0,0,0,1]) for f in FINGERS})
    held=copy.deepcopy(raw);held['partition']='validation';held['fingers']['thumb']['position']=[100,100,100]
    assert noise_from_training([raw,raw,held])==noise_from_training([raw,raw])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='full original-surface acceptance requires CUDA')
@pytest.mark.parametrize('two_phase',[False,True])
def test_full_native_fit_with_older_flat_initialization_rejects_bad_calibration(two_phase,monkeypatch):
    """Diagnostic fixture deliberately does NOT satisfy instructed contacts."""
    import threading
    from moxian_web.bone_calibration import fit_capture
    from moxian_web.static_capture import PROTOCOL,static_schedule
    from moxian_web.calibration_targets import target
    from moxian_web.static_pose import REVISION as POSE_REVISION
    from moxian_web.contact import default_mapping
    m=QnbotSurface('right').double();base=artifact(m);k=BoneKinematics(base['bone_extrinsics'])
    rest=np.asarray(base['bone_extrinsics']['rest_angles_rad']);rows=[];aggregates=[]
    mapping=dict(assignments=default_mapping('right'),confirmed=[])
    for i in range(4):
        q=np.clip(rest+i*.15,k.limits[:,0],k.limits[:,1]);r=frame(base,q,100+i*20)
        r.update(stage='open' if i==0 else 'hook',mapping=mapping,initialization=True,partition='initialization',calibration_fingerprint='eight')
        rows.append(r)
    for i,(part,stage,endpoint) in enumerate(static_schedule()):
        q=np.clip(rest+.2+(i%4)*.1,k.limits[:,0],k.limits[:,1]);r=frame(base,q,2000+i*3000)
        r.update(stage=stage,mapping=mapping,partition=part,initialization=False,capture_protocol=PROTOCOL,
            static_endpoint=endpoint,static_key=f'{part}:{stage}:{endpoint}',static_aggregate=True,
            static_accepted=True,static_sample_count=1,training_eligible=True,
            static_pose_validation=dict(accepted=True,revision=POSE_REVISION,evidence_source='synthetic_diagnostic_fixture'),
            contact_confirmed=target(stage)['kind']!='noncontact_motion',
            observation_weights=[float(f in target(stage)['active_fingers']) for f in FINGERS])
        # These old personalized seeds must not enter the new training model.
        r['skeleton']['joint_positions_rad']=[1000.]*25
        rows.append(r);aggregates.append(copy.deepcopy(r))
    profile=dict(serial='synthetic-diagnostic-only',side='right',calibration_fingerprint='eight',
        static_aggregates=aggregates,device='cuda:0')
    if two_phase:
        first_rows=[r for r in rows if r.get('partition')!='validation']
        first_profile=dict(profile,static_aggregates=aggregates[:18],capture_session_id='two-phase-test')
        pending,pending_report=fit_capture(None,first_profile,mapping,first_rows,lambda _:None,threading.Event(),solver_options={'maxeval':2})
        assert pending_report['validation_pending'] and not pending_report['accepted']
        assert not pending_report['validation_complete'] and pending_report['validation_frames_checked']==0
        def no_second_optimizer(*args,**kwargs):raise AssertionError('independent validation must reuse the fitted model')
        monkeypatch.setattr('moxian_web.bone_calibration.WujiCalibrationLbfgs',no_second_optimizer)
        profile['capture_session_id']='two-phase-test'
        candidate,report=fit_capture(None,profile,mapping,rows,lambda _:None,threading.Event(),initial_candidate=pending)
        assert report['reused_fitted_model'] and not report['validation_pending']
        assert candidate['model_hash']==pending['model_hash']
    else:
        candidate,report=fit_capture(None,profile,mapping,rows,lambda _:None,threading.Event(),solver_options={'maxeval':2})
    assert report['solver']['native_solver_reused']
    assert report['solver']['evaluations']==2 and not report['accepted']
    assert report['validation_complete'] and report['validation_frames_checked']==18
    assert candidate['delta_policy'] is None and report['independent_geometry_parameters']==51
    validate_candidate(candidate)


def test_streaming_skin_waits_for_identical_native_model(model):
    from moxian_web.qnbot_worker import StreamingModel
    candidate=artifact(model);runtime=StreamingModel(model);runtime.load(candidate)
    q=np.asarray(candidate['bone_extrinsics']['rest_angles_rad']);r=frame(candidate,q)
    r['skeleton']['model_hash']='different'
    with pytest.raises(ValueError,match='版本'):runtime.infer(r)
    r['skeleton']['model_hash']=candidate['model_hash']
    v,j,_,_,_,error=runtime.infer(r)
    assert torch.isfinite(v).all() and torch.isfinite(j).all() and error<1e-5
