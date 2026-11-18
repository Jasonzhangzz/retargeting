import copy
import torch
import pytest
from moxian_web.cart_pretrained_sdf import ConfigurationDistance,geometry_spec,load_pretrained,restore_geometry
from moxian_web.cart_collision import CollisionModel,SDFLearner,LinkSDF,frozen
from moxian_web.cart_objective import CartHand


@pytest.fixture(autouse=True)
def one_thread():torch.set_num_threads(1)


@pytest.fixture(scope='module')
def hand():return CartHand('cpu')


def checkpoint(hand,path):
    geometry=geometry_spec(hand)
    config=dict(lower=hand.domain['lower'],upper=hand.domain['upper'],joint_mask=geometry['joint_mask'],width=32,depth=2,embedding=16)
    torch.manual_seed(145)
    teacher=ConfigurationDistance(**config)
    saved=dict(model_config=config,geometry=geometry,model=teacher.state_dict(),epoch=30,fingerprint='test_geometry')
    torch.save(saved,path)
    return teacher,saved


def test_actual_q_native_frame_parity_gradients_and_checkpoint_roundtrip(hand,tmp_path):
    path=tmp_path/'sdf.pt';teacher,_=checkpoint(hand,path)
    model,spec=load_pretrained(hand,path);collision=CollisionModel(hand,model)
    q=((hand.low+hand.high)*.5)[None].repeat(2,1).requires_grad_()
    q.data[1,4]+=.2
    points=torch.tensor([[[.03,.02,.08]],[[.03,.02,.08]]],requires_grad=True)
    links=torch.tensor([[8],[8]])
    frames=hand.frames(q)
    expected=teacher(q[:,None],points@hand.basis.T+frames[3][:,hand.ids[0],None],links)
    actual=model.query(hand,q,points,links,frames)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    p,r=frames[:2];batch=torch.arange(2)[:,None]
    local=(r[batch,links].transpose(-1,-2)@(points-p[batch,links])[...,None]).squeeze(-1)
    through_loss=collision.distance(local,links,configuration=q,pose_ids=batch,frames=frames)
    torch.testing.assert_close(through_loss,actual,rtol=2.e-5,atol=2.e-8)
    dq,dx=torch.autograd.grad(through_loss.sum(),(q,points),retain_graph=True)
    reference=torch.autograd.grad(actual.sum(),(q,points))
    for a,b in zip((dq,dx),reference):torch.testing.assert_close(a,b,rtol=1.e-3,atol=2.e-7)
    assert dq[:,4:8].abs().sum()>0 and dx.abs().sum()>0
    assert dq[:,:4].abs().max()<1.e-7
    eps=1.e-3
    for index in (4,7):
        a=q.detach().clone();b=a.clone();a[:,index]+=eps;b[:,index]-=eps
        finite=(model.query(hand,a,points.detach(),links)-model.query(hand,b,points.detach(),links))/(2*eps)
        torch.testing.assert_close(dq[:,index],finite[:,0],rtol=.03,atol=1.e-6)
    with frozen(model):
        q2=q.detach().requires_grad_();x2=points.detach().requires_grad_()
        model.query(hand,q2,x2,links).sum().backward()
        assert q2.grad.abs().sum()>0 and x2.grad.abs().sum()>0
        assert not any(p.grad is not None for p in model.parameters())
    path.unlink()  # Deployment must not need the original initializer file.
    restored=restore_geometry(hand,spec);restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.query(hand,q.detach(),points.detach(),links),actual.detach(),rtol=0,atol=0)


def test_pretrained_distance_survives_contact_replay_transfer_and_updates_online(hand,tmp_path):
    path=tmp_path/'sdf.pt';_,_=checkpoint(hand,path)
    model,_=load_pretrained(hand,path);original={k:v.clone() for k,v in model.pretrained.state_dict().items()}
    old=CollisionModel(hand,LinkSDF());source=SDFLearner(old,capacity=64,batch_size=16)
    source.collect();source.update()
    model.load_contact_state(old.model.state_dict())
    for k,v in original.items():torch.testing.assert_close(model.pretrained.state_dict()[k],v,rtol=0,atol=0)
    for k,v in old.model.contact_field.state_dict().items():torch.testing.assert_close(model.contact_field.state_dict()[k],v,rtol=0,atol=0)
    learner=SDFLearner(CollisionModel(hand,model),capacity=64,batch_size=16)
    assert [group['lr'] for group in learner.optimizer.param_groups]==[1.e-5,1.e-3]
    learner.load_state_dict(source.state_dict(),load_optimizer=False)
    assert not learner.optimizer.state and learner.steps==source.steps
    losses=learner.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    assert any(not torch.equal(v,model.pretrained.state_dict()[k]) for k,v in original.items())
    x=torch.randn(17,3)*.01;links=torch.arange(17)%21
    rng=torch.get_rng_state().clone()
    a=model.training_distance(x,links,validation=True)
    torch.testing.assert_close(a,model.training_distance(x,links,validation=True),rtol=0,atol=0)
    assert torch.equal(rng,torch.get_rng_state())
    saved=tmp_path/'resume.pt';torch.save(dict(model=model.state_dict(),learner=learner.state_dict()),saved)
    state=torch.load(saved,weights_only=True);second,_=load_pretrained(hand,path);second.load_state_dict(state['model'])
    resumed=SDFLearner(CollisionModel(hand,second),capacity=64,batch_size=16);resumed.load_state_dict(state['learner'])
    rng=torch.get_rng_state();learner.update();torch.set_rng_state(rng);resumed.update()
    for k,v in model.state_dict().items():torch.testing.assert_close(v,second.state_dict()[k],rtol=0,atol=0)


def test_penetration_queries_use_the_correct_pose_for_ragged_tactile_rows(hand,tmp_path):
    path=tmp_path/'sdf.pt';_,_=checkpoint(hand,path)
    model,_=load_pretrained(hand,path);collision=CollisionModel(hand,model)
    q=((hand.low+hand.high)*.5)[None].repeat(2,1);q[1,4:8]+=.15
    points=torch.tensor([[.01,.02,.06],[.02,.02,.07]])
    rows=[dict(tactile_positions_m=points.tolist(),tactile_state=[0,0],tactile_valid=[True,True]),
          dict(tactile_positions_m=points.tolist(),tactile_state=[1,1],tactile_valid=[True,True])]
    captured=[];forward=model.pretrained.forward
    def capture(q,x,links):captured.append((q.detach().clone(),x.detach().clone(),links.clone()));return forward(q,x,links)
    model.pretrained.forward=capture
    frames=hand.frames(q);a,b=collision.penalties(q,rows,frames)
    assert a.shape==b.shape==(2,) and torch.isfinite(a+b).all()
    cq,cx,cl=captured[0]
    torch.testing.assert_close(cq[-42:],q[1].expand(42,-1))
    expected=points@hand.basis.T+frames[3][1,hand.ids[0]]
    torch.testing.assert_close(cx[-42:].reshape(2,21,3),expected[:,None].expand(-1,21,-1),rtol=1.e-5,atol=3.e-8)
    assert cl[-42:].tolist()==list(range(21))*2


@pytest.mark.parametrize('field,value',[('units','mm'),('frame','canonical_wrist'),('sign','positive_inside_negative_outside'),('asset_sha256','changed')])
def test_pretrained_import_rejects_incompatible_geometry(hand,tmp_path,field,value):
    path=tmp_path/'sdf.pt';_,saved=checkpoint(hand,path)
    saved=copy.deepcopy(saved);saved['geometry'][field]=value;torch.save(saved,path)
    with pytest.raises(ValueError,match='do not match'):load_pretrained(hand,path)
