"""Bounded CARD objective, its gradient, and actual optimizer instrumentation."""
from types import SimpleNamespace
import pytest
import torch
from moxian_web.cart_loss_scaling import normalized_policy_cost,policy_cost_scaling_spec


def test_complete_cost_scaling_preserves_relative_gradient_and_is_bounded():
    x=torch.tensor([2.,3.],dtype=torch.float64,requires_grad=True)
    raw=(x.square()*torch.tensor([1.e6,4.e6])).sum()
    scaled=normalized_policy_cost(raw,1.e6)
    gr=torch.autograd.grad(raw,x,retain_graph=True)[0]
    gs=torch.autograd.grad(scaled,x)[0]
    torch.testing.assert_close(gs,gr*1.e6/(raw.detach()+1.e6).square())
    assert 0<float(scaled.detach())<1
    assert torch.autograd.gradcheck(lambda v:normalized_policy_cost(v.square().sum()*1.e6,1.e6),(x,))
    vals=normalized_policy_cost(torch.tensor([0.,1.e6,1.e9],dtype=torch.float64),1.e6)
    assert vals[0]==0 and vals[1]==.5 and vals[2]<1 and (vals.diff()>0).all()


@pytest.mark.parametrize('scale',[-1.,float('inf'),float('nan'),True])
def test_invalid_scales_are_rejected(scale):
    with pytest.raises(ValueError):policy_cost_scaling_spec(scale)


def test_zero_scale_preserves_raw_tensor_identity():
    v=torch.tensor(4.,requires_grad=True)
    assert normalized_policy_cost(v,0.) is v


def test_policy_update_records_actual_clipping_and_parameter_change(monkeypatch):
    import moxian_web.cart_training as training
    model=torch.nn.Linear(2,1,bias=False)
    with torch.no_grad():model.weight.fill_(1.)
    class Trajectory:
        def collect(self,*a):pass
        def loss(self,*a,**kw):
            raw=1.e6*model.weight.square().sum()
            return raw,dict(contact_position=raw),[],[]
    monkeypatch.setattr(training,'predict_trajectory',lambda *a:Trajectory())
    monkeypatch.setattr(training,'perturb_initial_states',lambda batch,*a:batch)
    monkeypatch.setattr(training,'group_normalizers',lambda batch:None)
    objective=SimpleNamespace(hand=SimpleNamespace(device='cpu'),collision=SimpleNamespace(model=torch.nn.Linear(1,1)),policy_loss_scale=1.e6)
    learner=SimpleNamespace(collect=lambda:None,update=lambda:{})
    optimizer=torch.optim.AdamW(model.parameters(),lr=1.e-4)
    before=model.weight.detach().clone()
    loss,parts,_,_=training.policy_update(model,objective,learner,optimizer,[{}],1)
    torch.testing.assert_close(loss,torch.tensor(2./3))
    assert parts['raw_cost']==2.e6
    assert parts['gradient_norm_before_clip']<10
    assert parts['gradient_clip_multiplier']==1
    torch.testing.assert_close(parts['parameter_update_norm'],(model.weight.detach()-before).norm())
    assert parts['parameter_update_norm']>0
