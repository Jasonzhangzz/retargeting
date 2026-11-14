import json,threading
from types import SimpleNamespace
import numpy as np
import pytest,torch
from moxian_web.training_options import options,training_rows
from moxian_web.training_trace import TrainingTrace
from moxian_web.mano_gpu_objective import tip_emf_directions
from moxian_web.card_math import exp_so3


def test_tip_orientation_respects_mount_and_wrist_rotations_and_has_geometry_gradients():
    rest=torch.tensor([[0.,0.,0.],[1.,.1,0.]],requires_grad=True)
    rest_joints=torch.zeros(1,3)
    world=exp_so3(torch.tensor([.2,.3,-.4]));mount=exp_so3(torch.tensor([-.4,.1,.7]))[None]
    align=exp_so3(torch.tensor([-.3,.2,.1]))
    actual=rest[None]@world.T;joints=torch.zeros(1,1,3)
    raw=(align.T@world@mount[0])[None,None]
    a,b=tip_emf_directions(actual,joints,rest,rest_joints,[0],[1],raw,align,mount)
    torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-6)
    wrong=raw@exp_so3(torch.tensor([0.,0.,.5]))
    a,b=tip_emf_directions(actual,joints,rest,rest_joints,[0],[1],wrong,align,mount)
    loss=(a-b).square().sum();gradient=torch.autograd.grad(loss,rest)[0]
    assert float(loss.detach())>.01 and torch.isfinite(gradient).all() and gradient.abs().sum()>0


def test_training_trace_incremental_reset_ring_and_nonfinite_rejection():
    trace=TrainingTrace(2)
    def fit(run,n,loss=1.):return dict(run_id=run,state='running',method='parameters',training_sample=dict(sample_id=n,step=n,source='train_only',raw={'loss':loss}))
    trace.ingest(fit('first',1));trace.ingest(fit('first',1));trace.ingest(fit('first',2));trace.ingest(fit('first',3))
    result=trace.snapshot(0,'first');assert result['reset'] and [p['step'] for p in result['points']]==[2,3]
    result=trace.snapshot(2,'first');assert not result['reset'] and len(result['points'])==1
    trace.ingest(fit('first',4,float('nan')));assert trace.snapshot(3,'first')['points']==[]
    trace.ingest(fit('second',1));assert trace.snapshot(3,'first')['reset']
    json.dumps(trace.snapshot(),allow_nan=False)


def test_worker_sample_survives_progress_event_but_not_pose_polling():
    from moxian_web.hand_model import public_fit_report
    report=dict(run_id='live',state='running',method='parameters',training_sample=dict(sample_id=1,step=4,source='train_only',raw={'loss':2.}),solver={'training_trace':{'points':[1,2]},'parameter_snapshot':{'large':True}})
    wire=public_fit_report(report,include_training_sample=True)
    trace=TrainingTrace();trace.ingest(wire)
    assert trace.snapshot()['points'][0]['step']==4
    assert 'training_sample' not in public_fit_report(wire)
    assert 'training_trace' not in wire['solver'] and 'parameter_snapshot' not in wire['solver']


def test_batch_expansion_keeps_unique_train_only_coverage_and_does_not_mutate_source():
    from pathlib import Path
    source=Path(__file__).resolve().parents[2]/'output/moxian/hand_model/348640923d9e2f3f/motion-capture.json'
    if not source.exists():pytest.skip('Retained motion data required')
    rows=json.loads(source.read_text())['rows'];before=json.dumps(rows,sort_keys=True)
    full=rows[0].get('capture_protocol')=='contact_motion_v3'
    normal=training_rows(rows,full_buffer=full);dense=training_rows(rows,512,full_buffer=full)
    assert len(dense)==(len(normal) if full else 512) and all(r['partition']=='train' for r in dense)
    key=lambda r:(r['timestamp_ms'],r['stage'],r.get('cycle',-1))
    assert len({key(r) for r in dense})==len(dense)
    assert {key(r) for r in normal}<={key(r) for r in dense}
    assert json.dumps(rows,sort_keys=True)==before
    dense[0]['stage']='changed';assert rows[0]['stage']!='changed'
    for bad in ({'adam_steps':0},{'batch_size':True},{'max_seconds':float('inf')},{'extra':1}):
        with pytest.raises(ValueError):options(bad)


def test_chunked_forward_jacobian_matches_exact_autograd():
    from moxian_web.mano_calibrate_flashsac import chunked_jacobian
    x=torch.tensor([.1,.3,-.2,.5,.8])
    f=lambda z:torch.cat((z.square(),z.sin(),(z[:3]*z[2:]).sum()[None]))
    torch.testing.assert_close(chunked_jacobian(f,x,2),torch.func.jacfwd(f)(x),rtol=0,atol=0)


def test_telemetry_prior_kl_is_explicit_and_never_reads_holdout():
    from moxian_web.mano_training_metrics import TrainingMetrics
    with pytest.raises(ValueError,match='留出'):
        TrainingMetrics(SimpleNamespace(rows=[{'partition':'validation'}]),lambda _:None)


def test_fixed_pose_constants_do_not_hide_geometry_convergence():
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.mano_runtime import ManoSurface
    from moxian_web.mano_objective import ManoObjective
    if not DEFAULT_MANO.exists():pytest.skip('Licensed MANO required')
    m=ManoSurface(DEFAULT_MANO,device='cpu')
    o=ManoObjective(m,[dict(stage='mcp90',capture_protocol='geort_motion_v2',timestamp_ms=0,partition='train')],np.zeros((1,15,3)),np.eye(3)[None],shared=True,training=True,tactile=False,fixed_forward=True)
    a=o.residual(o.initial).square().sum()
    o.seed_angles=o.seed_angles+.5
    b=o.residual(o.initial).square().sum()
    torch.testing.assert_close(a,b,atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_collision_graph_capture_can_coexist_with_default_stream_display():
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.mano_runtime import ManoSurface
    from moxian_web.mano_gpu_collision import BatchedSurfaceCollision
    if not DEFAULT_MANO.exists():pytest.skip('Licensed MANO required')
    m=ManoSurface(DEFAULT_MANO);q=torch.zeros(2,15,3,device=m.device);root=torch.eye(3,device=m.device)[None].expand(2,-1,-1)
    with torch.no_grad():v=m.decode(q,root)[0]
    collision=BatchedSurfaceCollision(m,2);failures=[];ready=threading.Event()
    def query():
        try:
            ready.set()
            for _ in range(3):assert bool(collision.query(v)['valid'])
        except Exception as exc:failures.append(exc)
    thread=threading.Thread(target=query);thread.start();assert ready.wait(5)
    try:
        for _ in range(25):
            with torch.no_grad():m.decode(q,root)
        torch.cuda.synchronize()
    finally:thread.join(timeout=30)
    assert not thread.is_alive() and not failures and collision.query_graph is not None
