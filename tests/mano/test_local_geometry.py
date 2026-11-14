"""Local identification must freeze parameters, not only mask observations."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from moxian_web.hand_model import DEFAULT_MANO
from moxian_web.mano_runtime import ManoSurface
from moxian_web.mano_shape_prior import ShapePrior,GEOMETRY_UPPER
from moxian_web.mano_local_geometry import finger_coordinates,PairLocalRefinement,phase_stages
from moxian_web.mano_calibrate_flashsac import make_update


def objective(side='right'):
    path=DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl')
    if not path.exists():pytest.skip('Licensed MANO required')
    m=ManoSurface(path,device='cpu',side=side)
    rows=[dict(stage='pinch_'+f,partition='train',motion_phase=p)
          for f in ('index','middle','ring','pinky') for p in ('approach','touch','release')]
    o=SimpleNamespace(model=m,rows=rows)
    o.t=lambda value:torch.tensor(value,dtype=torch.float32)
    o.shape_prior=ShapePrior(m);o.weights=torch.full((len(rows),5),1/(5*len(rows)))
    o.lower=torch.cat((torch.full((46,),-3.),-torch.tensor(GEOMETRY_UPPER,dtype=torch.float32)))
    o.upper=-o.lower;o.fixed_forward=True;o.shared_size=89
    return o


@pytest.mark.parametrize('side',['left','right'])
def test_pair_blocks_leave_forward_identical_and_inactive_geometry_near_frozen(side):
    o=objective(side);x=torch.nn.Parameter(torch.zeros(89));ema=x.detach().clone()
    local=PairLocalRefinement(o,x,torch.zeros(89,dtype=torch.bool),o.weights.clone())
    local.select(0,x,ema)
    assert local.frozen[:10].all() and local.frozen[40:46].all()
    assert not (finger_coordinates(2)&~local.frozen).any()
    assert not (finger_coordinates(3)&~local.frozen).any()
    assert not (finger_coordinates(4)&~local.frozen).any()
    # Approach/release retain EMF weights; contact evidence is gated elsewhere.
    assert torch.equal(o.phase_active,torch.tensor([1.]*3+[0.]*9))
    assert o.weights[:3].sum()>0 and o.weights[3:].sum()==0
    before,joints_before=o.model.model.official_rest(x[:10],x[46:])
    q=torch.eye(3).expand(2,15,3,3)
    values=o.model.model.official_matrix_forward(x[:10],q,x[46:])
    local.select(1,x,ema)
    after=o.model.model.official_matrix_forward(x[:10],q,x[46:])
    for a,b in zip(values,after):torch.testing.assert_close(a,b,atol=0,rtol=0)
    with torch.no_grad():
        proposal=x.detach().clone();proposal[~local.frozen]=.001
        ema.copy_(local.update_ema(ema,proposal))
    assert torch.equal(ema[local.frozen],local.fixed[local.frozen])
    local.record_block(ema,32)
    assert local.report()['inactive_parameter_max_drift']==0
    assert local.report()['inactive_core_vertex_max_drift_mm']['pinch_middle']<.01
    v,joints_after=o.model.model.official_rest(ema[:10],ema[46:])
    from moxian_web.glove_geometry import CHAINS
    inactive_joints=[0]+[j for f in (1,3,4) for j in CHAINS[f]]
    torch.testing.assert_close(joints_after[inactive_joints],joints_before[inactive_joints],rtol=0,atol=0)
    assert (v-before).abs().max()>0
    local.close();assert not hasattr(o,'phase_active')


def test_dynamic_freeze_blocks_adam_momentum_and_ema_drift():
    o=objective();x=torch.nn.Parameter(torch.zeros(89));ema=x.detach().clone()
    local=PairLocalRefinement(o,x,torch.zeros(89,dtype=torch.bool),o.weights.clone())
    local.select(0,x,ema)
    o.residual=lambda z:z-torch.full_like(z,.02)
    update=make_update(o,x,None,local.frozen,local.fixed,compile_step=False)
    for block in range(8):
        local.select(block,x,ema)
        fixed=ema.clone()
        for _ in range(5):
            update();ema.copy_(local.update_ema(ema,x))
            assert torch.equal(x.detach()[local.frozen],fixed[local.frozen])
            assert torch.equal(ema[local.frozen],fixed[local.frozen])
        local.record_block(ema,5)
    assert local.report()['inactive_parameter_max_drift']==0
    assert set(local.counts.values())=={10}
    # The ten global shape parameters and wrist gauge never move.
    assert torch.equal(ema[:10],torch.zeros(10)) and torch.equal(ema[40:46],torch.zeros(6))


@pytest.mark.parametrize('side',['left','right'])
def test_critical_fingers_are_direct_and_only_noncritical_surface_uses_position_ema(side):
    o=objective(side);accepted=torch.zeros(89);x=accepted.clone()
    local=PairLocalRefinement(o,x,torch.zeros(89,dtype=torch.bool),o.weights.clone())
    local.select(0,x,accepted)
    critical=local.critical_vertices[0]
    assert critical.any() and (~critical).any()
    for step in range(3):
        previous,j_before=o.model.model.official_rest(accepted[:10],accepted[46:89])
        proposal=accepted.clone();proposal[~local.frozen]+=.001
        # Frozen coordinates cannot sneak in through optimizer momentum.
        proposal[local.frozen]=.1
        masked=torch.where(local.frozen,accepted,proposal)
        raw,_=o.model.model.official_rest(masked[:10],masked[46:89],apply_residual=False)
        accepted=local.update_ema(accepted,proposal)
        v,j_after=o.model.model.official_rest(accepted[:10],accepted[46:89])
        torch.testing.assert_close(accepted,masked,atol=0,rtol=0)
        torch.testing.assert_close(v[critical],raw[critical],atol=1e-9,rtol=0)
        torch.testing.assert_close(v[~critical],previous[~critical].lerp(raw[~critical],.05),atol=1e-9,rtol=0)
        coupled=(raw-previous)[~critical].norm(dim=-1)
        assert coupled.max()>1e-7
        torch.testing.assert_close((v-previous)[~critical],.05*(raw-previous)[~critical],atol=5e-9,rtol=1e-4)
        # Wrist and all other fingers' bones remain exactly fixed.
        from moxian_web.glove_geometry import CHAINS
        inactive=[0]+[j for f in (2,3,4) for j in CHAINS[f]]
        torch.testing.assert_close(j_before[inactive],j_after[inactive],atol=0,rtol=0)
    assert int(local.accepted_steps)==3 and int(local.rejected_steps)==0
    # Switching pairs also switches which fingers are allowed to move directly.
    local.select(1,x,accepted)
    assert not torch.equal(critical,local.critical_vertices[1])
    previous=o.model.model.rest_vertex_residual.clone()
    invalid=accepted.clone();invalid[66]=-2.
    invalid[10]+=.01
    rejected=local.update_ema(accepted,invalid)
    torch.testing.assert_close(rejected[46:89],accepted[46:89],atol=0,rtol=0)
    torch.testing.assert_close(o.model.model.rest_vertex_residual,previous,atol=0,rtol=0)
    assert rejected[10]==invalid[10] and int(local.rejected_steps)==1


@pytest.mark.parametrize('side',['left','right'])
def test_fixed_surface_ema_round_trip_and_reset_preserve_posed_geometry(side):
    o=objective(side);x=torch.zeros(89)
    local=PairLocalRefinement(o,x,torch.zeros(89,dtype=torch.bool),o.weights.clone())
    local.select(0,x,x.clone())
    proposal=x.clone();proposal[~local.frozen]=.001
    accepted=local.update_ema(x,proposal)
    source=o.model
    source.set_glove_geometry(accepted[46:89].numpy());source.set_shape(accepted[:10].numpy())
    residual=source.model.rest_vertex_residual.numpy().tolist()
    restored=objective(side).model
    original_version=restored.shape_version
    restored.set_rest_vertex_residual(residual)
    restored.set_glove_geometry(accepted[46:89].numpy());restored.set_shape(accepted[:10].numpy())
    assert source.shape_version==restored.shape_version!=original_version
    torch.testing.assert_close(source.rest,restored.rest,atol=0,rtol=0)
    from moxian_web.card_math import exp_so3
    rotations=exp_so3(torch.randn(2,15,3,generator=torch.Generator().manual_seed(7))*.2)
    a=source.model.official_matrix_forward(accepted[:10],rotations,accepted[46:89])
    b=restored.model.official_matrix_forward(accepted[:10],rotations,accepted[46:89])
    for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,atol=0,rtol=0)
    # Old checkpoints without this field clear all compensation.
    restored.set_rest_vertex_residual()
    assert not restored.model.rest_vertex_residual.any()
    raw,_=restored.model.official_rest(accepted[:10],accepted[46:89],apply_residual=False)
    torch.testing.assert_close(restored.rest,raw,atol=0,rtol=0)
    with pytest.raises(ValueError,match='EMA'):
        restored.set_rest_vertex_residual([[float('nan')]*3]*778)
    with pytest.raises(ValueError,match='EMA'):
        restored.set_rest_vertex_residual(np.ones((778,3)))


def test_block_selection_rejects_heldout_and_keeps_contacts_out_of_global_geometry_stage():
    o=objective();x=torch.zeros(89)
    o.rows[0]['partition']='validation'
    with pytest.raises(ValueError,match='训练分区'):
        PairLocalRefinement(o,x,torch.zeros(89,dtype=torch.bool),o.weights)
    assert phase_stages(o.rows,'geometry')==set()
    assert len(phase_stages(o.rows,'pinches'))==4
    from moxian_web.mano_gpu_solver import solve_calibration_gpu
    import threading
    with pytest.raises(ValueError,match='留出数据不能参与参数更新'):
        solve_calibration_gpu(o,threading.Event(),lambda _:None)


def test_contact_gradient_only_updates_owned_geometry_and_matches_finite_difference():
    from scripts.calibrate_wuji_mano import TIPS
    o=objective();x=torch.zeros(89,requires_grad=True)
    local=PairLocalRefinement(o,x,torch.zeros(89,dtype=torch.bool),o.weights.clone())
    local.select(0,x,x.detach().clone())
    def loss(value):
        v,_=o.model.model.official_rest(value[:10],value[46:])
        return (v[TIPS[0]]-v[TIPS[1]]).square().sum()
    g=torch.autograd.grad(loss(x),x)[0].masked_fill(local.frozen,0.)
    assert g[46:89].abs().sum()>0 and not g[local.frozen].any()
    k=46+3  # index proximal length
    epsilon=1e-3;offset=torch.zeros_like(x);offset[k]=epsilon
    numeric=(loss(x+offset)-loss(x-offset))/(2*epsilon)
    torch.testing.assert_close(g[k],numeric,rtol=.01,atol=1e-7)


def test_torch_adam_old_momentum_cannot_move_frozen_fingers():
    o=objective();x=torch.nn.Parameter(torch.zeros(89));ema=x.detach().clone()
    local=PairLocalRefinement(o,x,torch.zeros(89,dtype=torch.bool),o.weights.clone())
    optimizer=torch.optim.Adam([x],lr=.01)
    for block in range(4):
        local.select(block,x,ema)
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            (x-.01).square().sum().backward();x.grad[local.frozen]=0
            optimizer.step()  # deliberately retain old nonzero momentum
            with torch.no_grad():
                x[local.frozen]=local.fixed[local.frozen]
                ema.copy_(local.update_ema(ema,x))
            assert torch.equal(ema[local.frozen],local.fixed[local.frozen])
        local.record_block(ema,4)
    assert local.report()['inactive_parameter_max_drift']==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_gpu_contact_and_observation_losses_follow_current_pair_only():
    import json,math
    from pathlib import Path
    from moxian_web.mano_fit import choose_rows
    from moxian_web.official_mano import reference_from_frame
    from moxian_web.mano_gpu_objective import GPUManoObjective
    from scripts.calibrate_wuji_mano import CHAINS
    source=Path(__file__).resolve().parents[2]/'output/moxian/hand_model/348640923d9e2f3f/fit-wrist-targets.json'
    if not source.exists():pytest.skip('Retained training data required')
    rows,_=choose_rows(json.loads(source.read_text())['rows'])
    m=ManoSurface(DEFAULT_MANO.with_name('MANO_RIGHT.pkl'),side='right');m.enable_official_fk();m.build_binding(rows[0]['mapping'])
    m.install_flat_reference(reference_from_frame(next(r for r in rows if r['stage']=='open'),m.weights_hash,'unit-test','local-gradient'),allow_coarse=True)
    values=[m.official_fk.parameters(m,row,CHAINS) for row in rows]
    o=GPUManoObjective(m,rows,np.array([v[0] for v in values]),np.array([v[1] for v in values]),training=True,shared=True,fixed_forward=True)
    o.phase_stages=phase_stages(rows,'pinches')
    x=o.initial.clone().requires_grad_();local=PairLocalRefinement(o,x,torch.zeros_like(x,dtype=torch.bool),o.weights.clone())
    local.select(0,x,x.detach().clone())
    o.rawp.requires_grad_(True)
    loss=o.residual(x,record=True).square().sum();gradient,observation_gradient=torch.autograd.grad(loss,(x,o.rawp))
    assert torch.count_nonzero(observation_gradient[o.phase_active==0])==0
    with torch.no_grad():
        v=o.surface(x)[1];p,n=o.material_contacts(v)
        weight=o.phase_active[o.contact_rows]*o.contact_action_weight*o.contact_valid
        scale=(weight/(3*weight.sum().clamp_min(1e-12))).sqrt()[:,None]
        expected=((p[:,0]-p[:,1])/.002*scale*math.sqrt(320)).square().sum()
        from moxian_web.mano_contact_loss import normal_residual
        expected_normal=(normal_residual(n[:,0],n[:,1])*scale*math.sqrt(40)).square().sum()
    assert o.last_terms['contact_position']==pytest.approx(float(expected),rel=2e-6)
    assert o.last_terms['contact_normal']==pytest.approx(float(expected_normal),rel=2e-6)
    assert o.last_training_metrics['contact_normal_error_deg'] is not None
    with torch.no_grad():o.rawp[o.phase_active==0]+=1.
    changed=o.residual(x).square().sum();other=torch.autograd.grad(changed,x)[0]
    torch.testing.assert_close(loss,changed,rtol=0,atol=0)
    # Area-weighted CUDA index_add has FP32 atomic reduction roundoff; the
    # excluded observations' own gradients are required to be exactly zero.
    torch.testing.assert_close(gradient,other,rtol=2e-6,atol=2e-5)
    from moxian_web.mano_contact_loss import gradient_report
    diagnostics=gradient_report(o,x.detach())
    assert set(diagnostics['terms'])=={'contact_position','contact_normal','contact_tip_emf_orientation'}
    assert all(t['finite'] and t['shared_gradient_norm']>0 for t in diagnostics['terms'].values())
    # Execute the compiled Flash update against the real captured GPU buffer.
    # The second update must read the new surface buffer, not a baked-in zero.
    accepted=x.detach().clone()
    update=make_update(o,x,None,local.frozen,local.fixed,compile_step=True)
    for _ in range(2):
        previous,_=m.model.official_rest(accepted[:10],accepted[46:89])
        _,finite,_=update();assert bool(finite)
        proposal=x.detach().clone()
        raw,_=m.model.official_rest(proposal[:10],proposal[46:89],apply_residual=False)
        accepted=local.update_ema(accepted,proposal)
        actual,_=m.model.official_rest(accepted[:10],accepted[46:89])
        mask=local.critical_vertices[local.index]
        torch.testing.assert_close(accepted,proposal,atol=0,rtol=0)
        torch.testing.assert_close(actual[mask],raw[mask],atol=1e-9,rtol=0)
        torch.testing.assert_close(actual[~mask],previous[~mask].lerp(raw[~mask],.05),atol=1e-9,rtol=0)
        with torch.no_grad():x.copy_(accepted)
    o.install(accepted)
    with torch.no_grad():
        vv,jj,_,_=m.decode(o.seed_pose,o.root)
        _,expected_v,expected_j,_,_=o.surface(accepted)
    torch.testing.assert_close(vv,expected_v,atol=2e-7,rtol=1e-5)
    torch.testing.assert_close(jj,expected_j,atol=2e-7,rtol=1e-5)
