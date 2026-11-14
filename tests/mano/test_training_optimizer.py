"""Optimizer ownership, legacy Adam parity and real Muon joint updates."""
import copy
import pytest
import torch
from moxian_web.mano_joint_delta import JointDeltaPolicy
from moxian_web.training_optimizer import TrainingOptimizer
from moxian_web.training_options import options,validate_optimizer


def test_muon_owns_only_hidden_matrices_and_updates_both_optimizers():
    torch.manual_seed(13)
    geometry=torch.nn.Parameter(torch.zeros(89))
    policy=JointDeltaPolicy()
    optimizer=TrainingOptimizer(geometry,policy,'muon')
    assert optimizer.assignment['policy_muon']==['net.0.weight','net.2.weight']
    assert optimizer.assignment['geometry_reshaped'] is False
    adam={id(p) for g in optimizer.adam.param_groups for p in g['params']}
    muon={id(p) for g in optimizer.muon.param_groups for p in g['params']}
    assert not adam&muon
    assert adam|muon=={id(geometry),*(id(p) for p in policy.parameters())}
    assert id(policy.net[-1].weight) in adam and id(geometry) in adam
    hidden=policy.net[0].weight.detach().clone()
    from moxian_web.mano_calibrate_delta import FEATURES
    features=torch.randn(8,FEATURES)
    for _ in range(3):
        optimizer.zero_grad()
        loss=(policy(features)-.1).square().mean()+(geometry-.1).square().mean()
        loss.backward();optimizer.step()
    assert torch.isfinite(loss) and torch.isfinite(geometry).all()
    assert geometry.abs().sum()>0
    assert not torch.equal(policy.net[0].weight,hidden)
    assert all(torch.isfinite(p).all() for p in policy.parameters())


def test_adam_default_is_numerically_identical_to_previous_groups():
    torch.manual_seed(14)
    geometry=torch.nn.Parameter(torch.randn(89));policy=JointDeltaPolicy()
    reference=torch.nn.Parameter(geometry.detach().clone());network=copy.deepcopy(policy)
    optimizer=TrainingOptimizer(geometry,policy)
    previous=torch.optim.Adam([dict(params=[reference],lr=.01),dict(params=network.parameters(),lr=.001)])
    from moxian_web.mano_calibrate_delta import FEATURES
    features=torch.randn(4,FEATURES)
    for _ in range(3):
        for g,p,o in ((geometry,policy,optimizer),(reference,network,previous)):
            o.zero_grad();loss=g.square().mean()+(p(features)-.1).square().mean();loss.backward();o.step()
    torch.testing.assert_close(geometry,reference,rtol=0,atol=0)
    for actual,expected in zip(policy.parameters(),network.parameters()):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_optimizer_controls_preserve_old_requests_and_reject_unsupported_modes():
    assert options(dict(adam_steps=480,max_seconds=45,batch_size=0))['optimizer']=='adam'
    settings=options(dict(optimizer='muon'))
    validate_optimizer('delta',settings)
    for method in ('parameters','flashsac'):
        with pytest.raises(ValueError,match='Delta'):validate_optimizer(method,settings)
    with pytest.raises(ValueError):options(dict(optimizer='unknown'))
    with pytest.raises(ValueError,match='Delta'):
        TrainingOptimizer(torch.nn.Parameter(torch.zeros(89)),selection='muon')


def test_public_backend_forwards_optimizer_and_legacy_default(monkeypatch):
    from moxian_web import mano_calibration,mano_fit,gpu_memory
    from contextlib import nullcontext
    from types import SimpleNamespace
    captured=[]
    monkeypatch.setattr(mano_calibration,'prepare_rows',lambda *args:([],{}))
    monkeypatch.setattr(gpu_memory,'DeviceMemoryMonitor',lambda:nullcontext(SimpleNamespace(peak=0)))
    def fit(*args,**kwargs):
        captured.append(kwargs['solver_settings'])
        return {},dict(accepted=False,rejection_reasons=[])
    monkeypatch.setattr(mano_fit,'fit_capture',fit)
    for settings in (dict(optimizer='muon'),{}):
        mano_calibration.run_backend('delta',None,{},None,[],lambda _:None,None,training_options=settings)
    assert [s.optimizer for s in captured]==['muon','adam']
