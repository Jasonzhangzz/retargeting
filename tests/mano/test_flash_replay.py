import threading
import pytest
import torch
from moxian_web.mano_calibrate_flashsac import CalibrationReplayBuffer,make_update


def test_replay_is_immutable_aligned_and_train_only():
    source=torch.arange(24,dtype=torch.float32).reshape(6,4)
    buffer=CalibrationReplayBuffer([dict(partition='train')]*6,{'emf':source,'tactile':source[:,1:]})
    source.zero_()
    ids,batch=buffer.sample(20,generator=torch.Generator().manual_seed(3))
    assert batch['emf'].device==source.device
    torch.testing.assert_close(batch['emf'][:,1:],batch['tactile'])
    torch.testing.assert_close(batch['emf'][:,0],4*ids.float())
    with pytest.raises(ValueError,match='训练分区'):
        CalibrationReplayBuffer([dict(partition='validation')],{'emf':source[:1]})
    with pytest.raises(ValueError,match='帧数'):
        CalibrationReplayBuffer([dict(partition='train')]*6,{'emf':source[:2]})


def test_direct_update_freezes_geometry_and_respects_bounds():
    class Objective:
        lower=torch.tensor([-.2,-.2]);upper=-lower
        def residual(self,vector):return vector-torch.tensor([1.,-1.])
    vector=torch.nn.Parameter(torch.tensor([.1,0.]))
    opt=torch.optim.Adam([vector],lr=.08)
    update=make_update(Objective(),vector,opt,torch.tensor([True,False]),vector.detach().clone(),compile_step=False)
    initial=float(Objective().residual(vector).square().sum().detach())
    for _ in range(10):value,finite,before=update()
    assert bool(finite) and float(value)<initial
    torch.testing.assert_close(vector,torch.tensor([.1,-.2]))
    assert not before.requires_grad


def test_public_flash_entry_is_independent(monkeypatch):
    from moxian_web import mano_calibration,mano_calibrate_flashsac
    calls=[]
    monkeypatch.setattr(mano_calibration,'run_backend',lambda method,*a,**kw:calls.append(method))
    mano_calibrate_flashsac.fit_capture(None,{},None,[],lambda _:None,threading.Event())
    assert calls==['flashsac']


def test_missing_source_evidence_rejects_without_nonfinite_json():
    import json
    from moxian_web.mano_gpu_validation import skeleton_report
    from moxian_web.contact import FINGERS
    result=skeleton_report([{'skeleton':{}}],[{'source_tracking_errors':{f:{'position_error_mm':1.} for f in FINGERS}}])
    assert not result['accepted']
    assert result['per_finger']['thumb']['baseline_rmse_mm'] is None
    json.dumps(result,allow_nan=False)


def test_tensor_adam_matches_torch_adam_before_bounds():
    class Objective:
        lower=torch.full((3,),-10.);upper=-lower
        def residual(self,x):return torch.stack((x[0]*x[1]-1.,x[1]+x[2]-.2,x[2]-.5))
    actual=torch.nn.Parameter(torch.tensor([.3,.1,-.4]));expected=torch.nn.Parameter(actual.detach().clone())
    reference=torch.optim.Adam([expected],lr=.01)
    update=make_update(Objective(),actual,None,torch.zeros(3,dtype=torch.bool),actual.detach().clone(),compile_step=False)
    for _ in range(20):
        update();reference.zero_grad(set_to_none=True);Objective().residual(expected).square().sum().backward();reference.step()
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-6)
