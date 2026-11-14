import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from moxian_web import mano_mjwarp_check as check


def test_native_failure_returns_invalid_result_instead_of_losing_candidate(monkeypatch):
    monkeypatch.setattr(check,'_scene',lambda *a:('<mujoco/>',np.zeros((2,112))))
    process=SimpleNamespace(returncode=13,poll=lambda:13)
    monkeypatch.setattr(check.subprocess,'Popen',lambda *a,**k:process)
    result=check.cross_check(SimpleNamespace(device='cuda:0'),[0,1],None)
    assert not result['query_valid'] and result['process_isolated'] and '13' in result['error']


@pytest.mark.parametrize('cancelled',[False,True])
def test_timeout_and_cancel_always_stop_diagnostic_child(monkeypatch,cancelled):
    monkeypatch.setattr(check,'_scene',lambda *a:('<mujoco/>',np.zeros((2,112))))
    class Process:
        stopped=False;returncode=None
        def poll(self):return self.returncode
        def terminate(self):self.stopped=True;self.returncode=-15
        def wait(self,**kwargs):return self.returncode
    process=Process();monkeypatch.setattr(check.subprocess,'Popen',lambda *a,**k:process)
    cancel=threading.Event()
    if cancelled:cancel.set()
    if cancelled:
        with pytest.raises(InterruptedError):check.cross_check(SimpleNamespace(device='cuda:0'),[0,1],None,cancel,timeout_s=0)
    else:assert not check.cross_check(SimpleNamespace(device='cuda:0'),[0,1],None,cancel,timeout_s=0)['query_valid']
    assert process.stopped


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_real_isolated_mjwarp_checks_padded_batch_without_corrupting_parent_cuda():
    tetra=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.],[0.,0.,.01]],dtype=np.float32)
    class Model:
        device=torch.device('cuda:0')
        rest_j_np=np.array([[i*.04,0,0] for i in range(16)],dtype=np.float32)
        labels=np.repeat(np.arange(16),4)
        rest_np=np.concatenate([tetra+np.array([i*.04,0,0]) for i in range(16)])
        def decode(self,poses,roots):
            j=torch.as_tensor(self.rest_j_np,device=self.device)[None].expand(len(poses),-1,-1)
            r=torch.eye(3,device=self.device)[None,None].expand(len(poses),16,-1,-1)
            return None,j,r,None
    # 35 worlds exercises a padded final chunk after the initial 32.
    result=check.cross_check(Model(),list(range(35)),None)
    assert result['query_valid'],result.get('error')
    assert result['frames']==35 and result['process_isolated'] and not result['authoritative']
    assert float(torch.ones(2,device='cuda').sum())==2


def test_failed_proxy_check_cannot_be_applied_as_successful_calibration():
    from moxian_web.hand_model import calibration_report_valid,MANO_FIT_REVISION,MANO_SOLVER_REVISION
    report={k:True for k in ('accepted','contact_validation_passed','together_validation_passed','flat_validation_passed','penetration_validation_passed','runtime_validation_passed','per_stage_validation_passed','holdout_validation_passed','performance_passed','frozen_forward')}
    report.update(gpu_backend='torch_warp_v1',online_optimization=False,method='parameters',shape_validation={'accepted':True},fit_revision=MANO_FIT_REVISION,solver_revision=MANO_SOLVER_REVISION,validation_pose_source='runtime_mano_sensor_only',capture_protocol='official_eight_v1',max_penetration_mm=0.,runtime_max_penetration_mm=0.,mjwarp_cross_check={'query_valid':True})
    assert calibration_report_valid(report)
    report['mjwarp_cross_check']['query_valid']=False
    assert not calibration_report_valid(report)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_original_mano_hulls_with_flexed_fingers_pass_isolated_check():
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.mano_runtime import ManoSurface
    if not DEFAULT_MANO.exists():pytest.skip('Licensed MANO required')
    m=ManoSurface(DEFAULT_MANO.with_name('MANO_RIGHT.pkl'),side='right')
    q=torch.zeros(35,15,3,device=m.device);q[::2,:,0]=.7;q[::3,:,0]=1.2
    roots=torch.eye(3,device=m.device)[None].expand(len(q),-1,-1)
    report=check.cross_check(m,q,roots)
    assert report['query_valid'],report.get('error')
    assert report['contact_records']>0
    assert float(torch.ones(1,device=m.device).sum())==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_display_warp_collision_can_coexist_with_training_query_capture():
    import json
    from pathlib import Path
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.mano_runtime import ManoSurface,CollisionSurface
    from moxian_web.mano_gpu_collision import BatchedSurfaceCollision
    if not DEFAULT_MANO.exists():pytest.skip('Licensed MANO required')
    source=Path(__file__).resolve().parents[2]/'output/moxian/hand_model/348640923d9e2f3f/fit-wrist-targets.json'
    if not source.exists():pytest.skip('Retained mapping required')
    m=ManoSurface(DEFAULT_MANO);m.build_binding(json.loads(source.read_text())['rows'][0]['mapping'])
    q=torch.zeros(2,15,3,device=m.device)
    roots=torch.eye(3,device=m.device)[None].expand(2,-1,-1)
    with torch.no_grad():v,j,r,_=m.decode(q,roots)
    vv,jj,rr=(x[0].cpu().numpy() for x in (v,j,r))
    failures=[];started=threading.Event()
    def training():
        try:
            started.set();collision=BatchedSurfaceCollision(m,2)
            for _ in range(8):assert bool(collision.query(v)['valid'])
        except Exception as exc:failures.append(exc)
    worker=threading.Thread(target=training);worker.start();assert started.wait(5)
    try:
        display=CollisionSurface(m)
        for _ in range(8):assert not display.query(vv,jj,rr)['overflow']
    finally:worker.join(timeout=30)
    assert not worker.is_alive() and not failures
    assert float(torch.ones(1,device=m.device).sum())==1
