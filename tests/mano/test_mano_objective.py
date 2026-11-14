"""Task isolation, differentiability and full-surface acceptance regressions."""
import copy
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from moxian_web.hand_model import DEFAULT_MANO, calibration_report_valid, MANO_FIT_REVISION,MANO_SOLVER_REVISION
from moxian_web.mano_runtime import ManoSurface
from moxian_web.mano_collision import SurfaceCollision, collision_residual
from moxian_web.mano_objective import (ManoObjective, SolverSettings, anatomical_pose,
    bounded_lm, participation, stage_weights, decode,neutral_rotations,pose_angles)
from scripts.calibrate_wuji_mano import CHAINS


@pytest.fixture(scope='module')
def model():
    if not DEFAULT_MANO.exists(): pytest.skip('Licensed MANO weights required')
    torch.set_num_threads(1)
    m=ManoSurface(DEFAULT_MANO,device='cpu');m.flat_reference={'reference_id':'synthetic'}
    return m


def objective(model, rows=None, shared=True):
    row=dict(stage='pinch_index',timestamp_ms=1.,fingers={f:dict(position=[.01,.02,.03],quaternion_xyzw=[0,0,0,1])
            for f in ('thumb','index','middle','ring','pinky')})
    rows=rows or [row]
    return ManoObjective(model,rows,np.zeros((len(rows),15,3),np.float32),np.tile(np.eye(3),(len(rows),1,1)),
                         shared=shared,training=True,tactile=False)


def test_non_target_finger_observations_have_exactly_zero_loss_and_gradient(model):
    a=objective(model);rows=copy.deepcopy(a.rows)
    for f in ('middle','ring','pinky'):
        rows[0]['fingers'][f]['position']=[1000.,-2000.,3000.]
        rows[0]['fingers'][f]['quaternion_xyzw']=[.5,.5,.5,.5]
    b=objective(model,rows)
    def loss_grad(o):
        x=o.initial.clone().requires_grad_(True);loss=o.residual(x).square().sum()
        return loss.detach(),torch.autograd.grad(loss,x)[0]
    al,ag=loss_grad(a);bl,bg=loss_grad(b)
    torch.testing.assert_close(al,bl,rtol=0,atol=0)
    torch.testing.assert_close(ag,bg,rtol=0,atol=0)
    assert not a.free[0,np.array(CHAINS[2:]).ravel()-1].any()


@pytest.mark.parametrize('side',['left','right'])
def test_beta_and_pose_gradients_match_finite_differences(side):
    path=DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl')
    if not path.exists():pytest.skip('Licensed MANO weights required')
    m=ManoSurface(path,device='cpu',side=side)
    beta=torch.full((10,),.2,requires_grad=True)
    angles=torch.full((1,15,3),.08,requires_grad=True)
    basis=torch.tensor(m.anatomy.basis,dtype=torch.float32);root=torch.eye(3)[None]
    def evaluate(b,a):
        _,v,j,_=decode(m,b,a,root,basis)
        return v[0,744].square().sum()+j[0,4].square().sum()
    g,h=torch.autograd.grad(evaluate(beta,angles),(beta,angles))
    assert torch.isfinite(g).all() and g.abs().max()>1e-6 and torch.isfinite(h).all()
    for index in (0,3,9):
        delta=torch.zeros_like(beta);delta[index]=.001
        fd=(evaluate(beta+delta,angles)-evaluate(beta-delta,angles))/.002
        torch.testing.assert_close(g[index],fd,atol=1e-6,rtol=.02)
    delta=torch.zeros_like(angles);delta[0,12,0]=.001
    fd=(evaluate(beta,angles+delta)-evaluate(beta,angles-delta))/.002
    torch.testing.assert_close(h[0,12,0],fd,atol=1e-6,rtol=.02)


def test_pinch_has_twice_the_weight_and_repeated_frames_do_not_reweight_it():
    rows=[{'stage':'open'},{'stage':'pinch_index'},{'stage':'pinch_index'}]
    w=stage_weights(rows,participation(rows),True)
    assert np.isclose(w[0].sum(),1/3) and np.isclose(w[1:].sum(),2/3)
    np.testing.assert_allclose(w[1],[1/6,1/6,0,0,0])
    repeated=rows+[rows[-1]]*9
    other=stage_weights(repeated,participation(repeated),True)
    assert np.isclose(other[1:].sum(),w[1:].sum())


def test_training_and_registered_runtime_have_identical_physical_joint_bounds(model):
    row=copy.deepcopy(objective(model).rows[0]);row['stage']='all_fingers'
    previous=model.mounting_calibrated
    try:
        model.mounting_calibrated=True
        def make(training,shared):return ManoObjective(model,[row],np.zeros((1,15,3)),np.eye(3)[None],training=training,shared=shared,tactile=False)
        train=make(True,True);live=make(False,False)
        assert live.observations_enabled and live.correction_limit_rad is None
        torch.testing.assert_close(train.lower[46:],live.lower);torch.testing.assert_close(train.upper[46:],live.upper)
    finally:model.mounting_calibrated=previous


def test_only_confirmed_together_contact_adds_thumb_participation():
    rows=[dict(stage='together'),dict(stage='together',thumb_index_side_contact=True),dict(stage='open',thumb_index_side_contact=True)]
    assert participation(rows).tolist()==[[0,1,1,1,1],[1,1,1,1,1],[0,1,1,1,1]]


def test_initialization_phase_excludes_uninitialized_actions_from_shape_gradient(model):
    rows=[dict(stage='open',timestamp_ms=1.),dict(stage='fist',timestamp_ms=2.)]
    roots=np.tile(np.eye(3),(2,1,1));q=np.zeros((2,15,3),np.float32)
    def make(pose):
        o=ManoObjective(model,rows,pose,roots,training=True,shared=True,tactile=False)
        o.phase_stages={'open'};o.weights[1]=0;o.weights/=o.weights.sum()
        x=o.initial.clone().requires_grad_();loss=o.residual(x).square().sum()
        return loss.detach(),torch.autograd.grad(loss,x)[0][:46]
    loss,grad=make(q);q[1]=np.tile([1.4,.3,.2],(15,1));other,changed=make(q)
    torch.testing.assert_close(loss,other,rtol=0,atol=0);torch.testing.assert_close(grad,changed,rtol=0,atol=0)


def test_training_resume_keeps_completed_phase_and_finishes_joint_phase():
    from moxian_web.mano_objective import solve_calibration
    class Quadratic:
        initial=torch.tensor([1.,2.,3.]);lower=torch.full((3,),-4.);upper=torch.full((3,),4.)
        training=True;shared=False;shared_size=0;rows=[dict(stage='open',timestamp_ms=1.)]
        free=torch.ones((1,1,3),dtype=torch.bool);weights=torch.ones((1,5));contacts=[]
        mask_np=np.ones((1,5))
        settings=SolverSettings(adam_steps=2,lm_steps=0)
        t=staticmethod(lambda v:torch.tensor(v,dtype=torch.float32))
        def refresh(self,x,contacts=False):pass
        def residual(self,x,record=False):
            if record:self.last_terms=dict(quadratic=float(x.square().sum()))
            return x
        def install(self,x):self.final=x.clone()
    o=Quadratic();r=solve_calibration(o,threading.Event(),lambda v:None,max_seconds=10,resume_phases={'reference':2})
    assert [(p['phase'],p['steps']) for p in r['phases']]==[('reference',2),('pinches',2),('all_gestures',2)]
    assert r['phases'][0]['resumed_steps']==2 and o.final.square().sum()<o.initial.square().sum()
    assert r['parameter_snapshot']['vector']==o.final.tolist()


def test_snapshot_resume_preserves_mount_prior_centres_and_full_forward(model):
    from moxian_web.mano_fit import restore_training_snapshot
    fields=('shape','flat_reference','sensor_joint_position','mount_pos','mount_rot','raw_rotation','raw_translation')
    saved={k:copy.deepcopy(getattr(model,k)) for k in fields}
    try:
        a=objective(model);x=a.initial.clone();x[:10]=.2;x[10:46]=.03
        expected=a.surface(x);base=[a.mp0.clone(),a.mr0.clone(),a.align0.clone(),a.trans0.clone()]
        a.install(x)
        candidate={k:copy.deepcopy(getattr(model,k)) for k in ('shape','flat_reference','mount_pos','mount_rot','raw_rotation','raw_translation')}
        candidate['sensor_joint_position']=model.sensor_offsets().tolist()
        candidate['report']=dict(solver=dict(parameter_snapshot=dict(vector=x.tolist(),shared_size=46)))
        restored=restore_training_snapshot(model,candidate)
        b=objective(model);b.initial=torch.tensor(restored,dtype=torch.float32)
        for before,after in zip(base,(b.mp0,b.mr0,b.align0,b.trans0)):torch.testing.assert_close(before,after,atol=1e-6,rtol=1e-5)
        actual=b.surface(b.initial)
        for before,after in zip(expected[:4],actual[:4]):torch.testing.assert_close(before,after,atol=2e-6,rtol=1e-5)
    finally:
        model.set_shape(saved['shape'])
        for k in fields[1:]:setattr(model,k,saved[k])


def test_full_surface_detects_pinky_inside_palm_without_any_tactile_or_normals_gate(model):
    collision=SurfaceCollision(model)
    v=model.rest_np.copy()
    # Translate a distal segment into the palm, regardless of face orientation.
    ids=np.flatnonzero(model.labels==CHAINS[4][-1])
    palm=model.rest_np[model.labels==0].mean(0)
    v[ids]+=palm-v[ids].mean(0)
    witnesses,report=collision.query(v)
    assert report['max_penetration_m']>.002
    assert not report['constraints_satisfied'] and report['query_valid']
    live=torch.tensor(v,requires_grad=True)
    residual=collision_residual(live,witnesses)
    residual.square().sum().backward()
    assert torch.isfinite(live.grad).all() and live.grad[ids].abs().sum()>0


def test_cancel_timeout_and_nonfinite_never_silently_succeed(model):
    o=objective(model,shared=False);event=threading.Event();event.set()
    with pytest.raises(InterruptedError): bounded_lm(o,o.initial,cancel=event)
    x,report=bounded_lm(o,o.initial,deadline=time.monotonic()-1)
    assert report['budget_exhausted'];torch.testing.assert_close(x,o.initial)
    bad=o.initial.clone();bad[0]=float('nan')
    with pytest.raises(ValueError):bounded_lm(o,bad,iterations=1)


def test_partial_flat_or_contact_success_cannot_bypass_complete_gate():
    report=dict(fit_revision=MANO_FIT_REVISION,solver_revision=MANO_SOLVER_REVISION,accepted=True,flat_validation_passed=True,contact_validation_passed=True,
        together_validation_passed=True,penetration_validation_passed=True,
        runtime_validation_passed=True,per_stage_validation_passed=True,holdout_validation_passed=True,
        validation_pose_source='runtime_mano_sensor_only',capture_protocol='official_eight_v1',
        max_penetration_mm=1.5,runtime_max_penetration_mm=2.)
    assert calibration_report_valid(report)
    assert not calibration_report_valid({**report,'fit_revision':'official_mano_task_masked_lm_v10'})
    for key in ('penetration_validation_passed','runtime_validation_passed','holdout_validation_passed'):
        assert not calibration_report_valid({**report,key:False})
    assert not calibration_report_valid({**report,'max_penetration_mm':2.01})
    assert not calibration_report_valid({**report,'runtime_max_penetration_mm':12})
    assert not calibration_report_valid({'accepted':True,'validation':{'valid':True}})


def test_bounded_lm_decreases_the_shared_objective(model):
    o=objective(model,shared=False)
    before=float(o.residual(o.initial).square().sum())
    x,report=bounded_lm(o,o.initial,iterations=2)
    assert report['accepted_steps']>0 and report['cost']<before
    assert (x>=o.lower).all() and (x<=o.upper).all()


@pytest.mark.parametrize('side',['left','right'])
def test_zero_anatomical_angles_mean_straight_coplanar_physical_bones(side):
    path=DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl')
    if not path.exists():pytest.skip('Licensed MANO weights required')
    m=ManoSurface(path,device='cpu',side=side)
    neutral=torch.tensor(neutral_rotations(m),dtype=torch.float32)
    q=anatomical_pose(torch.zeros(1,15,3),torch.tensor(m.anatomy.basis,dtype=torch.float32),neutral)
    with torch.no_grad():v,j,r,o=m.decode(q,torch.eye(3)[None])
    for f,chain in enumerate(CHAINS[1:],1):
        directions=[]
        for k,joint in enumerate(chain):
            end=m.rest_np[[744,320,443,554,671][f]] if k==2 else m.rest_j_np[chain[k+1]]
            d=r[0,joint].numpy()@(end-m.rest_j_np[joint]);directions.append(d/np.linalg.norm(d))
        np.testing.assert_allclose(directions,np.tile(directions[0],(3,1)),atol=1e-5)
        assert np.max(np.abs(np.array(directions)@m.anatomy.palmar_normal))<1e-5
    np.testing.assert_allclose(pose_angles(m,q.numpy()),0,atol=2e-6)


def test_batched_live_residual_matches_single_residual(model):
    row=objective(model).rows[0]
    o=ManoObjective(model,[row],np.zeros((1,15,3)),np.eye(3)[None],training=False,tactile=False)
    x=o.initial.clone();batch=torch.stack((x,(x+.01).clamp(o.lower,o.upper)))
    with torch.no_grad():expected=torch.stack([o.residual(z) for z in batch]);actual=o.runtime_batch(batch)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


def test_lm_backtracks_across_a_max_penetration_kink():
    # At x=0 the GN direction is valid, but every step >= 1/4 crosses a
    # different closest-surface branch and increases the worst penetration.
    class Kink:
        initial=torch.tensor([0.]);lower=torch.tensor([-1.]);upper=torch.tensor([1.])
        training=True;shared=False;rows=[{}]
        def refresh(self,x):pass
        def residual(self,x,record=False):
            r=torch.maximum(1-x,100*x)
            if record:self.last_terms={'penetration':float(r.square().sum())}
            return r
    o=Kink();x,report=bounded_lm(o,o.initial,iterations=1)
    assert report['accepted_steps']==1 and report['cost']<1. and 0<float(x)<.02


def test_runtime_never_uses_unvalidated_or_stale_emf_mount(model):
    row=objective(model).rows[0]
    def live():return ManoObjective(model,[row],np.zeros((1,15,3)),np.eye(3)[None],tactile=False)
    previous=(model.mounting_calibrated,model.flat_reference)
    try:
        model.mounting_calibrated=False
        assert not live().observations_enabled
        model.mounting_calibrated=True
        assert live().observations_enabled
        model.flat_reference={**model.flat_reference,'calibration_fingerprint':'old'}
        row['calibration_fingerprint']='new'
        assert not live().observations_enabled
    finally:model.mounting_calibrated,model.flat_reference=previous


def test_deep_palm_collision_reports_conflicts_without_erasing_the_current_command(model):
    from moxian_web.mano_objective import solve_pose
    angles=np.zeros((1,15,3),np.float32)
    for chain in CHAINS[1:]:angles[0,np.array(chain)-1,0]=[1.5,1.7,1.3]
    q=anatomical_pose(torch.tensor(angles),torch.tensor(model.anatomy.basis,dtype=torch.float32),
                      torch.tensor(neutral_rotations(model),dtype=torch.float32))[0].numpy()
    with torch.no_grad():v,_,_,_=model.decode(torch.tensor(q[None]),torch.eye(3)[None])
    collision=SurfaceCollision(model);_,before=collision.query(v[0].numpy())
    assert before['max_penetration_m']>.002
    row=copy.deepcopy(objective(model).rows[0]);row['timestamp_ms']=9876.
    result,report,_=solve_pose(model,row,q,np.eye(3),tactile=False,iterations=2,budget_ms=300)
    assert not report['collision_escape_seed'] and np.isfinite(result).all()
    actual=pose_angles(model,result)
    assert np.max(np.abs(actual-angles[0]))<=np.deg2rad(8)+1e-5
    assert report['constraints_satisfied']==report['collision']['constraints_satisfied']
