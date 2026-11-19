"""Scalar logging uses actual optimizer steps and an explicit self-host URL."""
import sys
from types import SimpleNamespace
from moxian_web.cart_tracking import CartTracking


def test_disabled_tracking_never_imports_wandb(monkeypatch,tmp_path):
    monkeypatch.setitem(sys.modules,'wandb',None)
    tracker=CartTracking(tmp_path,dict(run_id='local'))
    tracker.log('retargeting',{});tracker.finish(error='cancelled')
    assert not list(tmp_path.iterdir())


def test_online_curves_resume_and_final_acceptance(monkeypatch,tmp_path):
    calls=[];logs=[];definitions=[];finished=[]
    run=SimpleNamespace(id='run',url='https://wandb.example/user/cart/run',summary={'error':'previous OOM'},
        define_metric=lambda *a,**kw:definitions.append((a,kw)),log=logs.append,
        finish=lambda **kw:finished.append(kw))
    def init(**kwargs):calls.append(kwargs);return run
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=init,Settings=lambda **kw:kw))
    tracker=CartTracking(tmp_path,dict(run_id='run'),project='cart',base_url='https://wandb.example/',resume=True)
    assert calls[0]['resume']=='must' and calls[0]['settings']['base_url']=='https://wandb.example'
    assert calls[0]['allow_val_change'] is True
    assert 'error' not in run.summary
    assert calls[0]['save_code'] is False and len(definitions)==6
    tracker.log('batch_compute',dict(compute_stage='collect'))
    assert not logs
    tracker.log('retargeting',dict(batch_losses=dict(total=2.,contact=1.),optimizer_step=7,epoch=2,
        parallel_batch_size=1326,transitions_per_s=400.,timings=dict(policy_s=44.),sdf_losses=dict(total=.3),
        sdf_updates=2014,elapsed_s=90.))
    assert logs[-1]['train/optimizer_step']==7 and logs[-1]['geometry/update']==2014
    assert logs[-1]['train/total']==2. and logs[-1]['timing/policy_s']==44.
    tracker.log('validation',dict(epoch=2,metrics=dict(feasible_rate=.9),sdf_validation=dict(distance_p95_mm=.7),
        validation_loss=3.,best_validation_loss=2.,elapsed_s=94.))
    assert logs[-1]['validation/epoch']==2 and logs[-1]['validation/sdf_distance_p95_mm']==.7
    tracker.finish(report=dict(accepted=False,rejection_reasons=['penetration']))
    assert run.summary['final/accepted'] is False and finished==[dict(exit_code=0)]
    assert 'https://wandb.example' in (tmp_path/'wandb_run.json').read_text()


def test_resume_clears_real_sdk_summary_error(monkeypatch,tmp_path):
    import pytest
    wandb=pytest.importorskip('wandb')
    run=wandb.init(project='cart-sdk-test',mode='offline',dir=str(tmp_path))
    try:
        run.summary['error']='previous OOM'
        monkeypatch.setattr(wandb,'init',lambda **kwargs:run)
        tracker=CartTracking(tmp_path,dict(run_id=run.id),project='cart-sdk-test',mode='offline',resume=True)
        assert tracker.run is run and run.summary.get('error') is None
    finally:
        run.finish()
