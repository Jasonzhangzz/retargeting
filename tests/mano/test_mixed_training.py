"""Full-buffer scheduling, sample ownership and fixed deployment parity."""
import json,threading
from pathlib import Path
import numpy as np
import pytest
import torch
from moxian_web.hand_model import DEFAULT_MANO
from moxian_web.mano_runtime import ManoSurface
from moxian_web.mano_mixed_training import geometry_gradient_mask,masked_geometry
from moxian_web.training_options import training_rows
from moxian_web.card_math import exp_so3
from scripts.calibrate_wuji_mano import TIPS,CHAINS

SOURCE=Path(__file__).resolve().parents[2]/'output/moxian/hand_model/348640923d9e2f3f/fit-wrist-targets.json'


def test_full_buffer_does_not_subsample_or_mix_holdout():
    if not SOURCE.exists():pytest.skip('Retained capture required')
    rows=json.loads(SOURCE.read_text())['rows']
    for size in (0,512,768):
        train=training_rows(rows,size,full_buffer=True)
        assert len(train)==2442 and all(r['partition']=='train' for r in train)
        assert [r['timestamp_ms'] for r in train]==sorted(r['timestamp_ms'] for r in rows if r['partition']=='train')
        assert len({r['timestamp_ms'] for r in train})==2442
        train[0]['stage']='changed';assert rows[0]['stage']!='changed'
    with pytest.raises(ValueError,match='训练分区'):
        geometry_gradient_mask([dict(stage='pinch_index',partition='validation')])


@pytest.mark.parametrize('side',['left','right'])
def test_sample_geometry_gradient_ownership_and_batched_lbs_parity(side):
    path=DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl')
    if not path.exists():pytest.skip('Licensed MANO required')
    m=ManoSurface(path,device='cpu',side=side)
    rows=[dict(stage=s,partition='train') for s in ('pinch_index','pinch_pinky','range_middle')]
    mask=geometry_gradient_mask(rows)
    x=torch.zeros(89,requires_grad=True)
    with torch.no_grad():x[:10]=.03;x[46:71]=.01
    beta,geometry=masked_geometry(x,mask)
    rotations=exp_so3(torch.randn(3,15,3,generator=torch.Generator().manual_seed(91))*.15)
    v,j,r,rest,rest_j=m.model.official_batched_matrix_forward(beta,rotations,geometry)
    expected=m.model.official_matrix_forward(x[:10],rotations,x[46:89])
    for actual,reference in zip((v,j,r),expected):torch.testing.assert_close(actual,reference,atol=1e-7,rtol=1e-5)
    loss=(v[0,TIPS[0]]-v[0,TIPS[1]]).square().sum()
    gradient=torch.autograd.grad(loss,x,retain_graph=True)[0]
    assert gradient[46:89].abs().sum()>0 and not gradient[:10].any()
    assert not gradient[52:61].any()  # other three fingers' lengths
    assert not gradient[63:66].any() and not gradient[68:71].any()
    other=torch.autograd.grad(v[2].square().sum(),x,retain_graph=True)[0]
    assert other[:10].abs().sum()>0
    # Check a permitted derivative, including the masked vmap backward.
    k=49;eps=1e-3
    def value(z):
        b,g=masked_geometry(z,mask);vv=m.model.official_batched_matrix_forward(b,rotations,g)[0]
        return (vv[0,TIPS[0]]-vv[0,TIPS[1]]).square().sum()
    offset=torch.zeros_like(x);offset[k]=eps
    numeric=(value(x+offset)-value(x-offset))/(2*eps)
    torch.testing.assert_close(gradient[k],numeric,rtol=.01,atol=1e-7)


def test_noncritical_ema_keeps_wrist_coordinates_fixed_when_beta_changes():
    from moxian_web.mano_shape_prior import ShapePrior
    from moxian_web.mano_local_geometry import spatial_ema_step
    m=ManoSurface(DEFAULT_MANO,device='cpu');prior=ShapePrior(m)
    previous=torch.zeros(89);proposal=previous.clone();proposal[:10]=.001
    frozen=torch.zeros(89,dtype=torch.bool)
    weights=m.model.backend.lbs_weights
    critical=torch.zeros(778,dtype=torch.bool)
    for partner in range(1,5):critical|=weights[:,list(CHAINS[0])+list(CHAINS[partner])].sum(-1)>=.5
    before,before_j=m.model.official_rest(previous[:10],previous[46:89])
    raw,raw_j=m.model.official_rest(proposal[:10],proposal[46:89],apply_residual=False)
    assert (raw_j[0]-before_j[0]).norm()>0
    accepted,valid=spatial_ema_step(m.model,prior,previous,proposal,frozen,critical)
    assert valid and torch.equal(accepted,proposal)
    actual,actual_j=m.model.official_rest(accepted[:10],accepted[46:89])
    torch.testing.assert_close(actual[critical],raw[critical],atol=0,rtol=0)
    expected=(before-before_j[0]).lerp(raw-raw_j[0],.05)
    torch.testing.assert_close((actual-actual_j[0])[~critical],expected[~critical],atol=1e-8,rtol=1e-5)


def test_soft_shape_loss_increases_for_thin_fingers_and_restores_thickness():
    from moxian_web.mano_shape_prior import ShapePrior
    m=ManoSurface(DEFAULT_MANO,device='cpu');prior=ShapePrior(m);prior.soft_constraints=True
    beta=torch.zeros(10);values=[]
    for depth in (0.,-.15,-.4):
        geometry=torch.zeros(43);geometry[20:25]=depth;geometry.requires_grad_()
        penalty=prior.residuals(beta,geometry)['shape_soft_constraints'].square().sum()
        gradient=torch.autograd.grad(penalty,geometry)[0]
        assert torch.isfinite(penalty) and torch.isfinite(gradient).all()
        values.append(float(penalty.detach()))
        if depth==-.4:
            vertices,_=m.model.official_rest(beta,geometry)
            assert not prior.valid_surface(vertices)
            # Gradient descent increases depth towards an anatomical section.
            assert gradient[20:25].sum()<0
            offset=torch.zeros_like(geometry);offset[21]=1e-3
            def value(g):return prior.residuals(beta,g)['shape_soft_constraints'].square().sum()
            numeric=(value(geometry+offset)-value(geometry-offset))/(2e-3)
            torch.testing.assert_close(gradient[21],numeric,rtol=.02,atol=1e-3)
    assert values[0]<values[1]<values[2]


def test_soft_shape_step_keeps_finite_proposal_outside_anatomical_region():
    from moxian_web.mano_shape_prior import ShapePrior
    from moxian_web.mano_local_geometry import spatial_ema_step
    m=ManoSurface(DEFAULT_MANO,device='cpu');prior=ShapePrior(m)
    previous=torch.zeros(89);proposal=previous.clone();proposal[66:71]=-.4
    frozen=torch.zeros(89,dtype=torch.bool)
    critical=m.model.backend.lbs_weights[:,list(CHAINS[0])].sum(-1)>=.5
    before,before_j=m.model.official_rest(previous[:10],previous[46:89])
    raw,raw_j=m.model.official_rest(proposal[:10],proposal[46:89],apply_residual=False)
    updated,accepted=spatial_ema_step(m.model,prior,previous,proposal,frozen,critical,soft_constraints=True)
    assert accepted and torch.equal(updated,proposal)
    m.model.surface_residual_enabled=True
    actual,actual_j=m.model.official_rest(updated[:10],updated[46:89])
    assert not prior.valid_surface(actual)  # Accepted for learning, not deployment.
    torch.testing.assert_close(actual[critical],raw[critical],atol=0,rtol=0)
    expected=(before-before_j[0]).lerp(raw-raw_j[0],.05)
    torch.testing.assert_close((actual-actual_j[0])[~critical],expected[~critical],atol=1e-8,rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_mixed_gpu_contact_gradient_and_deployment_match():
    if not SOURCE.exists():pytest.skip('Retained capture required')
    from moxian_web.mano_mixed_training import MixedGPUManoObjective,solve_mixed
    from moxian_web.official_mano import reference_from_frame
    from moxian_web.mano_fit import choose_rows
    from moxian_web.mano_objective import SolverSettings
    full=json.loads(SOURCE.read_text())['rows'];coverage=choose_rows(full)[0]
    opened=next(r for r in coverage if r['stage']=='open')
    # Keep a real mixed buffer, including four independently evidenced contacts.
    from moxian_web.motion_capture import confirmed_touch,bilateral
    rows=[opened]+[next(r for r in coverage if r['stage']=='pinch_'+f and confirmed_touch(r) and bilateral(r,f)) for f in ('index','middle','ring','pinky')]
    m=ManoSurface(DEFAULT_MANO.with_name('MANO_RIGHT.pkl'),side='right');m.enable_official_fk();m.build_binding(opened['mapping'])
    m.install_flat_reference(reference_from_frame(opened,m.weights_hash,'test','mixed-test'),allow_coarse=True)
    values=[m.official_fk.parameters(m,row,CHAINS) for row in rows]
    o=MixedGPUManoObjective(m,rows,np.asarray([v[0] for v in values]),np.asarray([v[1] for v in values]),training=True,shared=True,tactile=True,fixed_forward=True)
    x=o.initial.clone().requires_grad_();residual=o.residual(x,record=True)
    # One indexed contact residual may not alter unrelated geometry or beta.
    v=o.surface(x)[1];p,_=o.material_contacts(v)
    grad=torch.autograd.grad((p[0,0]-p[0,1]).square().sum(),x)[0]
    assert grad[46:89].abs().sum()>0 and not grad[:10].any()
    assert not grad[52:61].any() and not grad[63:66].any() and not grad[68:71].any()
    for flash in (False,True):
        o.settings=SolverSettings(adam_steps=2,lm_steps=0)
        report=solve_mixed(o,threading.Event(),lambda _:None,max_seconds=30,compile_loss=False,flash=flash)
        assert [p['phase'] for p in report['phases']]==['mixed']
        assert report['replay']['frames']==5 and report['replay']['all_frames_used_every_step']
        assert not report['stage_gating'] and not report['pair_rotation']
        assert report['phases'][0]['steps']==2
        assert report['local_geometry']['rejected_surface_steps']==0
        assert report['local_geometry']['accepted_surface_steps']==2
        assert not report['shape_soft_constraints']['training_step_rejection']
        assert report['cost_terms']['shape_soft_constraints']>0
        assert not hasattr(o,'phase_stages') and not hasattr(o,'phase_active')
        vector=o.t(report['parameter_snapshot']['vector']);o.install(vector)
        with torch.no_grad():
            vv,jj,_,_=m.decode(o.seed_pose,o.root);_,expected_v,expected_j,_,_=o.surface(vector)
        torch.testing.assert_close(vv,expected_v,atol=2e-7,rtol=1e-5)
        torch.testing.assert_close(jj,expected_j,atol=2e-7,rtol=1e-5)
    cancel=threading.Event();cancel.set()
    with pytest.raises(InterruptedError):solve_mixed(o,cancel,lambda _:None,compile_loss=False)
