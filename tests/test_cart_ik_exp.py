"""IK reference semantics and exponential task costs, independent of loss display."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from moxian_web.cart_reward import exp_cost
from moxian_web.cart_network import CartNetwork
from moxian_web.cart_observations import pack_future
from moxian_web.cart_training import full_session
from moxian_web.cart_objective import CartHand,CartObjective,policy_settings
from moxian_web.cart_collision import CollisionModel,frozen


@pytest.fixture(autouse=True)
def one_thread():torch.set_num_threads(1)


def observations(batch=2):
    return dict(future_ik=torch.full((batch,10,20),.55),ik_valid=torch.ones(batch,10,dtype=torch.bool),
        future_dt=torch.arange(1,11)[None].expand(batch,-1)*.02,
        tactile_positions=torch.zeros(batch,10,460,3),tactile_state=torch.zeros(batch,10,460),
        tactile_valid=torch.ones(batch,10,460,dtype=torch.bool))


@pytest.mark.parametrize('architecture',['attention','fsq'])
def test_fresh_policy_tracks_ik_and_trained_residual_preserves_bptt(architecture):
    torch.manual_seed(8);model=CartNetwork(torch.zeros(20),torch.ones(20),architecture)
    q=torch.full((2,20),.5);args=observations()
    delta=model(q,**args)
    torch.testing.assert_close(model.apply_delta(q,delta),args['future_ik'][:,0])
    with torch.no_grad():model.decoder[-2].weight.normal_(std=.02)
    action=model(q,**args);action.retain_grad()
    q1=model.apply_delta(q,action)
    q2=model.apply_delta(q1,model(q1,**args))
    q2.square().mean().backward()
    assert action.grad.isfinite().all() and action.grad.abs().sum()>0
    # The learned correction is at most .1 in normalized joint coordinates.
    assert (model.normalize(q1)-model.normalize(args['future_ik'][:,0])).abs().max()<=.100001


@pytest.mark.parametrize('architecture',['attention','fsq'])
def test_no_emf_input_ik_masks_and_exact_ten_step_boundary(architecture):
    model=CartNetwork(torch.zeros(20),torch.ones(20),architecture)
    with torch.no_grad():model.decoder[-2].weight.normal_(std=.02)
    q=torch.full((2,20),.5);args=observations()
    args['ik_valid'][:,-1]=False
    a=model(q,**args);args['future_ik'][:,-1]=float('nan')
    torch.testing.assert_close(a,model(q,**args))
    args['ik_valid'].fill_(False);args['future_ik'].fill_(float('nan'))
    assert model(q,**args).isfinite().all()
    with pytest.raises(ValueError):model(q,**dict(args,future_ik=torch.zeros(2,11,20)))
    rows=[dict(timestamp_ms=i*20.,ik_valid=True,ik_q_rad=[i/100.]*20,
        tactile_positions_m=np.zeros((460,3)),tactile_state=[0]*460,tactile_valid=[True]*460) for i in range(12)]
    packed=pack_future(0.,rows[1:11])
    torch.testing.assert_close(packed['future_ik'][:,0],torch.arange(1,11)/100.)
    rows[11]['ik_q_rad']=[999.]*20
    torch.testing.assert_close(packed['future_ik'],pack_future(0.,rows[1:11])['future_ik'])
    del rows[1]['ik_valid']
    with pytest.raises(ValueError):pack_future(0.,rows[1:11])


def test_exponential_cost_is_bounded_accurate_and_has_recovery_gradients():
    error=torch.tensor([0.,1.e-7,.01,.04,.1],dtype=torch.float64,requires_grad=True)
    cost=exp_cost(error.square(),.01,4.)
    assert cost[0]==0 and (cost>=0).all() and (cost<=1).all()
    gradient=torch.autograd.grad(cost.sum(),error)[0]
    assert gradient[3]>1 and gradient[4]>.01  # 40 / 100 mm retain usable gradients
    assert gradient.isfinite().all()
    assert torch.autograd.gradcheck(lambda x:exp_cost(x.square(),.01,4.),(error,),eps=1.e-7,atol=1.e-5)


def test_policy_loss_bound_and_native_metric_units():
    hand=CartHand('cpu');collision=CollisionModel(hand);settings=policy_settings()
    q=((hand.low+hand.high)/2)[None].requires_grad_();p,r,j,*_=hand.frames(q)
    tri=hand.tensor(hand.surface.contact_surface(8)['triangles'][0]).mean(0)
    point=(r[0,8]@tri+p[0,8]).detach()+torch.tensor([0.,0.,.03])
    row=dict(timestamp_ms=20.,stream_epoch=0,joints_m=j[0].detach().tolist(),
        tactile_positions_m=np.zeros((460,3)),tactile_valid=[False]*460,tactile_state=[0]*460,
        contacts=dict(tactile_ready=True,self_pairs=[],object_contacts=[dict(region='index_distal',
            position_m=point.tolist(),normal=[1.,0.,0.],force=1.)]))
    obj=CartObjective(hand,settings,collision)
    with frozen(collision.model):
        loss,parts,_=obj.batch(q,[row],q.detach())
        loss.backward()
    assert loss.isfinite() and 0<=loss<10 and q.grad.isfinite().all()
    assert 0<=parts['contact']<=2.25 and 0<=parts['penetration']<=1
    assert settings.contact_scale_m==pytest.approx(.01)
    assert settings.contact_limit_m==settings.penetration_limit_m==.0005


def test_complete_session_initializes_from_current_ik_instead_of_neutral():
    seen=[]
    class Model:
        def __call__(self,q,**args):seen.append(q.clone());return torch.zeros_like(q)
        def apply_delta(self,q,delta):return q+delta
    rows=[dict(timestamp_ms=i*20.,ik_valid=True,ik_q_rad=[.4+i*.001]*20,
        tactile_positions_m=np.zeros((460,3)),tactile_state=[0]*460,tactile_valid=[True]*460) for i in range(13)]
    objective=SimpleNamespace(hand=SimpleNamespace(device='cpu',tensor=lambda x:torch.tensor(x)),
        collision=SimpleNamespace(auditing=False),batch=lambda *a,**k:(torch.tensor(0.),{},[]))
    full_session(Model(),objective,[(dict(id='test'),rows,{})])
    assert len(seen)==3
    for state in seen:torch.testing.assert_close(state,torch.full((1,20),.4))
