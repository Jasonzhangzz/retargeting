import copy,json,threading
from pathlib import Path
import numpy as np
import pytest
import torch
from moxian_web.mano_calibrate_delta import DeltaPolicy,FEATURES,load_policy,REVISION
from moxian_web.mano_calibration import fit_capture
from moxian_web.mano_gpu_runtime import project_sequence


def test_delta_starts_at_existing_ik_and_roundtrips():
    net=DeltaPolicy();x=torch.randn(7,FEATURES)
    assert torch.equal(net(x),torch.zeros(7,15,3))
    artifact=dict(revision=REVISION,features=FEATURES,state={k:v.tolist() for k,v in net.state_dict().items()})
    torch.testing.assert_close(load_policy(artifact,'cpu')(x),net(x),rtol=0,atol=0)
    artifact['state']['scale'][0]=0
    with pytest.raises(ValueError,match='归一化'):load_policy(artifact,'cpu')


def test_public_method_validation():
    with pytest.raises(ValueError,match='parameters'):fit_capture(None,{},None,[],lambda _:None,threading.Event(),method='unknown')


def test_causal_projection_chunking_and_gap_reset():
    angles=torch.ones(5,15,3);seed=torch.zeros_like(angles);low=torch.full((15,3),-2.);high=-low
    rows=[{'timestamp_ms':x} for x in (0,20,40,500,520)]
    full,_=project_sequence(angles,seed,low,high,rows,previous=(torch.zeros(15,3),-20),rate_deg_s=90)
    a,old=project_sequence(angles[:2],seed[:2],low,high,rows[:2],previous=(torch.zeros(15,3),-20),rate_deg_s=90)
    b,_=project_sequence(angles[2:],seed[2:],low,high,rows[2:],previous=old,rate_deg_s=90)
    torch.testing.assert_close(full,torch.cat((a,b)),rtol=0,atol=0)
    assert full[0].max()<.04 and torch.equal(full[3],angles[3])


@pytest.mark.parametrize('side',['left','right'])
def test_batched_original_triangle_collision_matches_single(side):
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.mano_runtime import ManoSurface
    from moxian_web.mano_gpu_collision import BatchedSurfaceCollision
    from moxian_web.mano_objective import decode
    path=DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl')
    if not torch.cuda.is_available() or not path.exists():pytest.skip('CUDA / licensed MANO required')
    m=ManoSurface(path,side=side);device=m.device
    q=torch.zeros(2,15,3,device=device);q[1,:,0]=.7
    _,v,_,_=decode(m,torch.zeros(10,device=device),q,torch.eye(3,device=device)[None].expand(2,-1,-1),torch.as_tensor(m.anatomy.basis,device=device,dtype=torch.float32))
    c=BatchedSurfaceCollision(m,2);w=c.query(v)
    assert bool(w['valid'])
    for i in range(2):
        reference=c.reference.query(v[i].detach().cpu().numpy())[1]
        assert abs(reference['max_penetration_m']-float(w['maximum'][i]))<2e-5
    z=v.detach().requires_grad_();loss=c.residual(z,w).square().sum();grad=torch.autograd.grad(loss,z)[0]
    assert torch.isfinite(grad).all()


def test_gpu_geometry_lbs_matches_upstream_transform():
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.mano_runtime import ManoSurface
    from smplx.lbs import batch_rigid_transform
    from moxian_web.card_math import exp_so3
    if not DEFAULT_MANO.exists():pytest.skip('Licensed MANO required')
    m=ManoSurface(DEFAULT_MANO,device='cpu');model=m.model
    local=exp_so3(torch.randn(3,16,3)*.2);j=torch.randn(3,16,3)*.03
    actual,t=model.rigid_transforms(local,j)
    expected,old=batch_rigid_transform(local,j,model.backend.parents)
    torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-6);torch.testing.assert_close(t,old,atol=2e-7,rtol=2e-6)


def test_so3_forward_derivative_stays_fp32_at_identity():
    from moxian_web.card_math import exp_so3
    x=torch.zeros(3);v=torch.tensor([.1,.2,.3]);_,j=torch.func.jvp(exp_so3,(x,),(v,))
    assert j.dtype==torch.float32 and torch.isfinite(j).all()
    fd=(exp_so3(x+v*.001)-exp_so3(x-v*.001))/.002
    torch.testing.assert_close(j,fd,atol=1e-6,rtol=1e-5)


def retained_model_and_rows():
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.mano_runtime import ManoSurface
    from moxian_web.official_mano import reference_from_frame
    root=Path(__file__).resolve().parents[2]
    source=root/'output/moxian/hand_model/348640923d9e2f3f/fit-wrist-targets.json'
    path=DEFAULT_MANO.with_name('MANO_RIGHT.pkl')
    if not torch.cuda.is_available() or not source.exists() or not path.exists():pytest.skip('Local retained observations and CUDA required')
    rows=json.loads(source.read_text())['rows'];m=ManoSurface(path,side='right');m.enable_official_fk();m.build_binding(rows[0]['mapping'])
    m.install_flat_reference(reference_from_frame(next(r for r in rows if r['stage']=='open'),m.weights_hash,'2DDDF3AC0DA5','unit-training-reference'),allow_coarse=True)
    return m,rows


def test_policy_features_do_not_read_labels_or_future():
    from moxian_web.mano_gpu_objective import GPUManoObjective
    from moxian_web.mano_calibrate_delta import features
    from scripts.calibrate_wuji_mano import CHAINS
    m,rows=retained_model_and_rows();rows=copy.deepcopy(rows[:2]);values=[m.official_fk.parameters(m,r,CHAINS) for r in rows]
    o=GPUManoObjective(m,rows,np.asarray([v[0] for v in values]),np.asarray([v[1] for v in values]),training=True,shared=True,tactile=False)
    before=features(o,o.initial)
    for r in o.rows:r.update(stage='invented',cycle=99,partition='validation',guided_contact=True,contact_confirmed=True,future_pose=[99]*45)
    after=features(o,o.initial)
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    assert before.shape==(2,FEATURES)


def test_delta_training_rejects_holdout():
    from moxian_web.mano_calibrate_delta import train_policy
    class O:rows=[dict(partition='validation')]
    with pytest.raises(ValueError,match='训练分区'):train_policy(O(),None,threading.Event(),lambda _:None)


def test_gpu_objective_gradient_and_repeated_rows():
    from moxian_web.mano_gpu_objective import GPUManoObjective
    from scripts.calibrate_wuji_mano import CHAINS
    m,rows=retained_model_and_rows();rows=[copy.deepcopy(next(r for r in rows if r['stage']=='open'))]
    def make(r):
        values=[m.official_fk.parameters(m,row,CHAINS) for row in r]
        return GPUManoObjective(m,r,np.asarray([v[0] for v in values]),np.asarray([v[1] for v in values]),training=True,shared=True,tactile=True)
    o=make(rows);x=o.initial.detach().clone().requires_grad_();loss=o.residual(x,measurement_only=True).square().sum();g=torch.autograd.grad(loss,x)[0]
    delta=torch.zeros_like(x);delta[3]=.001
    fd=(o.residual(x+delta,measurement_only=True).square().sum()-o.residual(x-delta,measurement_only=True).square().sum())/.002
    torch.testing.assert_close(g[3],fd,atol=.03,rtol=.02)
    other=make(rows*3);duplicated=other.residual(other.initial,measurement_only=True).square().sum()
    torch.testing.assert_close(duplicated,loss,atol=.03,rtol=1e-5)


def test_runtime_batch_matches_single_and_ignores_labels(monkeypatch):
    from moxian_web import mano_gpu_runtime as runtime
    from scripts.calibrate_wuji_mano import CHAINS
    monkeypatch.setattr(runtime,'pose_loss',runtime._pose_loss)
    m,source=retained_model_and_rows();rows=copy.deepcopy(source[:2])
    rows[1]['timestamp_ms']=rows[0]['timestamp_ms']+1000 # independent reset
    rows[1]['skeleton']['timestamp_ms']=rows[1]['timestamp_ms']
    values=[m.official_fk.parameters(m,r,CHAINS) for r in rows];poses=np.asarray([v[0] for v in values]);roots=np.asarray([v[1] for v in values])
    q,_,decoded,_,_=runtime.solve_batch(m,rows,poses,roots,tactile=False,iterations=2)
    for i in range(2):
        altered=copy.deepcopy(rows[i]);altered.update(stage='pinch_pinky',guided_contact=True,contact_confirmed=True,cycle=3,partition='validation')
        one,_,single,_,_=runtime.solve_batch(m,[altered],poses[i:i+1],roots[i:i+1],tactile=False,iterations=2)
        # The translated BVH has FP32 rounding; require <20 micrometres in the
        # actual surface, well below the unchanged millimetre acceptance gates.
        torch.testing.assert_close(q[i],one[0],atol=5e-5,rtol=5e-5)
        torch.testing.assert_close(decoded[0][i],single[0][0],atol=2e-5,rtol=1e-5)


@pytest.mark.parametrize('side',['left','right'])
def test_signed_ray_matches_winding_for_deep_articulation(side):
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.mano_runtime import ManoSurface
    from moxian_web.mano_gpu_collision import BatchedSurfaceCollision
    from moxian_web.mano_objective import decode
    path=DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl')
    if not torch.cuda.is_available() or not path.exists():pytest.skip('CUDA / licensed MANO required')
    m=ManoSurface(path,side=side);torch.manual_seed(12)
    q=torch.randn(12,15,3,device=m.device)*.7
    q[0,:,0]=1.6;q[0,:,1:]=0
    _,v,_,_=decode(m,torch.zeros(10,device=m.device),q,torch.eye(3,device=m.device)[None].expand(len(q),-1,-1),torch.as_tensor(m.anatomy.basis,device=m.device,dtype=torch.float32))
    c=BatchedSurfaceCollision(m,len(q));positive=False
    for vertices in (v,v.flip(0)):
        w=c.query(vertices);assert bool(w['valid'])
        for i in range(len(q)):
            reference=c.reference.query(vertices[i].detach().cpu().numpy())[1]['max_penetration_m']
            positive |= reference>.001
            assert abs(reference-float(w['maximum'][i]))<2e-5
    assert positive # exercise signs, not just separated fingers


def test_runtime_really_uses_matching_delta_and_rejects_stale_artifact():
    from moxian_web import mano_gpu_runtime as runtime
    from scripts.calibrate_wuji_mano import CHAINS
    m,source=retained_model_and_rows();row=copy.deepcopy(source[0])
    pose,root=m.official_fk.parameters(m,row,CHAINS)[:2]
    kwargs=dict(tactile=False,iterations=0)
    baseline=runtime.solve_batch(m,[row],pose[None],root[None],**kwargs)[0]
    policy=DeltaPolicy().to(m.device)
    with torch.no_grad():policy.net[-1].bias.fill_(.3)
    m.delta_policy=policy
    m.delta_artifact=dict(binding_version=m.binding_version,mapping_version=m.mapping.get('version'),calibration_fingerprint=row['calibration_fingerprint'])
    active=runtime.solve_batch(m,[row],pose[None],root[None],**kwargs)[0]
    assert not torch.allclose(active,baseline,atol=1e-4,rtol=0)
    m.delta_artifact['calibration_fingerprint']='stale-capture'
    stale=runtime.solve_batch(m,[row],pose[None],root[None],**kwargs)[0]
    torch.testing.assert_close(stale,baseline,atol=2e-6,rtol=0)
