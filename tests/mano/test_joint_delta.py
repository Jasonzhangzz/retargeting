import copy,json,threading
from pathlib import Path
import numpy as np
import pytest
import torch
from moxian_web.hand_model import DEFAULT_MANO
from moxian_web.mano_runtime import ManoSurface
from moxian_web.mano_joint_delta import JointDeltaPolicy,FEATURES,PoseChart,observable_inputs
from moxian_web.mano_shape_prior import project,ShapePrior
from moxian_web.glove_geometry import validate

SOURCE=Path(__file__).resolve().parents[2]/'output/moxian/hand_model/348640923d9e2f3f/fit-wrist-targets.json'


def test_unbounded_shape_projection_and_roundtrip_keep_large_finite_parameters():
    x=torch.zeros(89);x[:10]=2.;x[46:71]=.6;x[71:83]=1.
    low=torch.full_like(x,-3.);high=-low
    low[:10]=-float('inf');high[:10]=float('inf');low[46:]=-float('inf');high[46:]=float('inf')
    assert torch.equal(project(x,low,high,89,unbounded_shape=True),x)
    np.testing.assert_array_equal(validate(x[46:].numpy()),x[46:].numpy())
    m=ManoSurface(DEFAULT_MANO,device='cpu');m.set_shape(x[:10].numpy());m.set_glove_geometry(x[46:].numpy())
    assert np.linalg.norm(m.shape[:10])>1.5 and np.isfinite(m.rest_np).all()
    prior=ShapePrior(m);prior.soft_constraints=True;prior.unbounded_shape=True
    z=x.clone().requires_grad_();loss=sum(v.square().mean() for v in prior.residuals(z[:10],z[46:]).values())
    g=torch.autograd.grad(loss,z)[0];assert torch.isfinite(g).all() and g[:10].norm()>0


@pytest.mark.parametrize('side',['left','right'])
def test_joint_policy_zero_head_exactly_preserves_ik_and_has_gradient(side):
    from moxian_web.mano_objective import neutral_rotations
    m=ManoSurface(DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl'),device='cpu',side=side)
    net=JointDeltaPolicy();x=torch.randn(3,FEATURES);delta=net(x)
    seed=torch.randn(3,15,3)*.1
    ref=dict(basis=m.anatomy.basis.tolist(),neutral=neutral_rotations(m).tolist(),low=m.anatomy.lower.tolist(),high=m.anatomy.upper.tolist())
    chart=PoseChart(m,[{}]*3,seed.numpy(),reference=ref)
    q,d=chart.correct(seed,delta,torch.ones(3,5))
    assert torch.equal(q,seed) and not d.any()
    q.square().sum().backward();assert net.net[-1].weight.grad.norm()>0
    invalid,_=chart.correct(seed,torch.ones_like(delta),torch.zeros(3,5));assert torch.equal(invalid,seed)


def test_joint_observations_keep_unknown_touch_unknown_and_ignore_labels():
    rows=[dict(timestamp_ms=100,tactile_timestamp_ms=100,contact=dict(ready=False,patches=[dict(id='thumb_distal',response=1)]))]
    a=observable_inputs(rows,'cpu');assert not a[0].any()
    other=copy.deepcopy(rows);other[0].update(stage='pinch_index',cycle=99,partition='validation',future_pose=[99]*45,contact_confirmed=True)
    for x,y in zip(a,observable_inputs(other,'cpu')):assert torch.equal(x,y)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('optimizer',['adam','muon'])
def test_joint_training_changes_geometry_and_network_and_matches_fixed_inference(optimizer):
    from moxian_web.mano_joint_delta import JointGPUManoObjective,infer
    from moxian_web.mano_mixed_training import solve_mixed
    from moxian_web.mano_calibrate_delta import load_policy
    from moxian_web.mano_objective import SolverSettings
    from moxian_web.official_mano import reference_from_frame
    from moxian_web.motion_capture import confirmed_touch,bilateral
    from scripts.calibrate_wuji_mano import CHAINS
    rows=json.loads(SOURCE.read_text())['rows'];train=[r for r in rows if r['partition']=='train']
    selected=[next(r for r in train if r['stage']=='open')]+[next(r for r in train if r['stage']=='pinch_'+f and confirmed_touch(r) and bilateral(r,f)) for f in ('index','middle','ring','pinky')]
    m=ManoSurface(DEFAULT_MANO.with_name('MANO_RIGHT.pkl'),side='right');m.enable_official_fk();m.build_binding(selected[0]['mapping'])
    m.install_flat_reference(reference_from_frame(selected[0],m.weights_hash,'test','joint-test'),allow_coarse=True)
    values=[m.official_fk.parameters(m,r,CHAINS) for r in selected];poses=np.asarray([v[0] for v in values]);roots=np.asarray([v[1] for v in values])
    o=JointGPUManoObjective(m,selected,poses,roots,training=True,shared=True,tactile=True,fixed_forward=True,settings=SolverSettings(adam_steps=3,lm_steps=0,optimizer=optimizer))
    assert torch.isinf(o.lower[:10]).all() and torch.isinf(o.upper[46:]).all()
    assert len(o.initial)==89 and not o.free.any()
    x=o.initial.detach().clone();before=o.joint_policy.net[-1].weight.detach().clone()
    with torch.no_grad():q=o.surface(x)[0];assert torch.equal(q,o.seed_pose)
    report=solve_mixed(o,threading.Event(),lambda _:None,max_seconds=60,compile_loss=False)
    assert report['optimizer']==('Muon+Adam' if optimizer=='muon' else 'Adam')
    assert report['joint_geometry_policy'] and not report['geometry_frozen_during_policy_training']
    assert report['local_geometry']['rejected_surface_steps']==0 and report['policy_parameter_count']>0
    assert not torch.equal(o.joint_policy.net[-1].weight,before)
    saved=o.t(report['parameter_snapshot']['vector']);assert not torch.equal(saved[:10],x[:10]) and not torch.equal(saved[46:83],x[46:83])
    m.delta_policy=load_policy(m.delta_artifact,m.device)
    with torch.no_grad():expected=o.surface(saved)
    actual=infer(m,selected,poses,roots)
    torch.testing.assert_close(actual,expected[0],atol=2e-6,rtol=2e-5)
    for i in range(len(selected)):
        single=infer(m,[selected[i]],poses[i:i+1],roots[i:i+1])
        torch.testing.assert_close(single,actual[i:i+1],atol=2e-6,rtol=2e-5)
    modified=copy.deepcopy(selected)
    for r in modified:r.update(stage='anything',cycle=99,partition='validation',future_pose=[99]*45,contact_confirmed=False)
    torch.testing.assert_close(infer(m,modified,poses,roots),actual,atol=0,rtol=0)
    with torch.no_grad():v,j,_,_=m.decode(actual,torch.as_tensor(roots,device=m.device))
    torch.testing.assert_close(v,expected[1],atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(j,expected[2],atol=2e-6,rtol=2e-5)
    # A second objective must start at both learned geometry and the same
    # nonzero policy, including normalization and the physical joint chart.
    from moxian_web.mano_fit import restore_training_snapshot
    parent=dict(shape=m.shape.tolist(),flat_reference=copy.deepcopy(m.flat_reference),
        glove_parameters=m.glove_parameters.tolist(),rest_vertex_residual_m=m.model.rest_vertex_residual.cpu().tolist(),
        sensor_joint_position=m.sensor_offsets().tolist(),mount_rot=m.mount_rot.tolist(),
        raw_rotation=m.raw_rotation.tolist(),raw_translation=m.raw_translation.tolist(),
        report=dict(solver=report),delta_policy=copy.deepcopy(m.delta_artifact))
    fresh=ManoSurface(DEFAULT_MANO.with_name('MANO_RIGHT.pkl'),side='right');fresh.enable_official_fk();fresh.build_binding(selected[0]['mapping'])
    restore_training_snapshot(fresh,parent)
    seeds=[fresh.official_fk.parameters(fresh,row,CHAINS) for row in selected]
    continued=JointGPUManoObjective(fresh,selected,np.asarray([s[0] for s in seeds]),np.asarray([s[1] for s in seeds]),
        training=True,shared=True,tactile=True,fixed_forward=True,initial_policy=parent['delta_policy'],
        settings=SolverSettings(adam_steps=2,lm_steps=0,optimizer=optimizer))
    continued.initial=continued.t(report['parameter_snapshot']['vector'])
    continued.resume_info=dict(previous_steps=report['total_training_steps'],parent_run_id='test-parent')
    continued.refresh(continued.initial)
    for name,value in o.joint_policy.state_dict().items():
        torch.testing.assert_close(continued.joint_policy.state_dict()[name],value,atol=0,rtol=0)
    with torch.no_grad():
        start=continued.surface(continued.initial)
        torch.testing.assert_close(start[0],expected[0],atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(start[1],expected[1],atol=2e-6,rtol=2e-5)
    old_head=continued.joint_policy.net[-1].weight.detach().clone()
    next_report=solve_mixed(continued,threading.Event(),lambda _:None,max_seconds=60,compile_loss=False)
    assert next_report['total_training_steps']==5
    assert next_report['training_trace']['points'][0]['previous_training_steps']==3
    assert not torch.equal(continued.joint_policy.net[-1].weight,old_head)
    m.delta_artifact['surface_version']='changed'
    torch.testing.assert_close(infer(m,selected,poses,roots),o.seed_pose,atol=0,rtol=0)
    assert m._delta_disabled_reason
