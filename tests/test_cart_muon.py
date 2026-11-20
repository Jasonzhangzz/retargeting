"""Parameter routing and exact resume of the Muon/SGD composition."""
import copy
import pytest
import torch
from moxian_web.cart_optim import MuonSGD,cosine_multiplier,scale_learning_rates


def network():
    return torch.nn.Sequential(torch.nn.Embedding(7,4),torch.nn.Linear(4,4),torch.nn.LayerNorm(4),torch.nn.Linear(4,2))


def update(model,optimizer):
    optimizer.zero_grad();model(torch.tensor([0,2,3])).square().sum().backward();optimizer.step()


def test_muon_routes_every_parameter_once_and_resumes_identically():
    torch.manual_seed(714);model=network();optimizer=MuonSGD(model)
    assert optimizer.assignment['0.weight']=='sgd'
    assert optimizer.assignment['1.weight']=='muon'
    assert optimizer.assignment['1.bias']=='sgd'
    assert optimizer.assignment['2.weight']=='sgd'
    ids=[id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids)==len(set(ids))==len(list(model.parameters()))
    assert all(isinstance(o,(torch.optim.Muon,torch.optim.SGD)) for o in optimizer.optimizers.values())
    update(model,optimizer)
    snapshot=copy.deepcopy(model.state_dict());state=copy.deepcopy(optimizer.state_dict())
    other=network();other.load_state_dict(snapshot);other_opt=MuonSGD(other);other_opt.load_state_dict(state)
    update(model,optimizer);update(other,other_opt)
    for a,b in zip(model.parameters(),other.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    with pytest.raises(ValueError):other_opt.load_state_dict(dict(kind='adamw'))


def test_geometry_groups_keep_pretrained_distance_lr_separate():
    model=torch.nn.Module();model.pretrained=torch.nn.Linear(4,4);model.contact_field=torch.nn.Linear(4,3);model.links=torch.nn.Embedding(4,4)
    optimizer=MuonSGD(model,pretrained_lr=1.e-5,contact_lr=1.e-3)
    groups={id(p):g['lr'] for g in optimizer.param_groups for p in g['params']}
    assert groups[id(model.pretrained.weight)]==1.e-5
    assert groups[id(model.contact_field.weight)]==1.e-3
    assert groups[id(model.links.weight)]==1.e-3


def test_schedule_changes_step_size_and_does_not_compound_rates():
    assert cosine_multiplier(0,140)==.2
    assert cosine_multiplier(4,140)==1
    assert cosine_multiplier(139,140)==pytest.approx(.1)
    optimizer=MuonSGD(network(),lr=.001)
    scale_learning_rates(optimizer,.5);scale_learning_rates(optimizer,.25)
    assert all(group['lr']==.00025 for group in optimizer.param_groups)
