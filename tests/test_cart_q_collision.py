"""Q-only inference, native teacher derivatives, and checkpoint continuity."""
import json
import pytest
import torch
from moxian_web.cart_objective import CartHand
from moxian_web.cart_collision import CollisionModel,frozen
from moxian_web.cart_q_collision import QCollisionField,QCollisionTeacher,QCollisionLearner,attach_field,field_spec
from moxian_web.cart_card_neural_surface import NeuralCardCostObjective
from test_cart_card_cost import targets,row


@pytest.fixture(scope='module')
def hand():
    torch.set_num_threads(1)
    return CartHand('cpu')


def test_q_only_forward_and_escape_need_no_fk_or_mesh_and_pass_finite_differences(hand,monkeypatch):
    field=QCollisionField(hand,json.loads(json.dumps(field_spec(hand)))).double()
    q=((hand.low+hand.high)/2)[None].double().requires_grad_()
    def forbidden(*a,**k):raise AssertionError('Geometry lookup entered q-only inference')
    monkeypatch.setattr(hand,'frames',forbidden);monkeypatch.setattr(hand.fixed_fk,'frames',forbidden)
    assert torch.autograd.gradcheck(lambda x:field(x),(q,),eps=1.e-5,atol=1.e-4,rtol=1.e-3)
    energy,_,cap=field(q);penalty=cap.add(1.e-8).sqrt().sub(5).clamp_min(0).square()
    g=torch.autograd.grad((1.5**2*energy+40**2*penalty).sum(),q)[0]
    result=field.escape(q)
    torch.testing.assert_close(result['direction_rad'],-torch.nn.functional.normalize(g,dim=-1))
    assert torch.isfinite(result['collision_probability']).all()


def test_mesh_teacher_matches_card_energy_coarse_depth_and_finite_difference_gradient(hand):
    c=CollisionModel(hand);teacher=QCollisionTeacher(c)
    torch.manual_seed(171)
    q=hand.low+torch.rand(3,20)*(hand.high-hand.low)
    labels=teacher.label(q);frames=hand.frames(q)
    local,links,valid=c.oracle.self_queries(frames);d,_=c.oracle.label(local,links)
    expected=torch.where(valid,(6-d.reshape(links.shape)*1000).clamp_min(0).square(),0.).sum(-1)
    torch.testing.assert_close(labels['amplitude'].square().sum(-1),expected,rtol=1.e-5,atol=.001)
    native,valid=hand.collision.query(frames[3],frames[4],dense=False)
    assert valid.all()
    torch.testing.assert_close(labels['depth_mm'],native.amax((1,2)),rtol=1.e-4,atol=.001)
    native_energy,native_depth,native_cap=teacher.values(frames)
    torch.testing.assert_close(native_energy,expected,rtol=1.e-5,atol=.001)
    torch.testing.assert_close(native_cap,labels['cap_amplitude'].square().sum(-1),rtol=1.e-5,atol=.001)
    eps=1.e-4
    for joint in (0,4,13):
        a=q.clone();b=q.clone();a[:,joint]+=eps;b[:,joint]-=eps
        av=teacher.label(a)['amplitude'].square().sum(-1).add(.01).sqrt()
        bv=teacher.label(b)['amplitude'].square().sum(-1).add(.01).sqrt()
        torch.testing.assert_close(labels['gradient'][:,joint],(av-bv)/(2*eps),rtol=.03,atol=.08)
        av=teacher.label(a)['cap_amplitude'].square().sum(-1).add(.0001).sqrt()
        bv=teacher.label(b)['cap_amplitude'].square().sum(-1).add(.0001).sqrt()
        torch.testing.assert_close(labels['cap_gradient'][:,joint],(av-bv)/(2*eps),rtol=.03,atol=.08)


def test_policy_has_no_self_queries_and_preserves_bptt_and_all_loss_gradients(hand,monkeypatch):
    c=CollisionModel(hand);attach_field(c,field_spec(hand));obj=NeuralCardCostObjective(hand,c)
    q=((hand.low+hand.high)/2)[None].requires_grad_();previous=(q.detach()-.01).requires_grad_()
    t=targets(obj,q.detach()+.04);rows=[row(t,0,20.)];old=[row(t,0,0.)]
    def forbidden(*a,**k):raise AssertionError('Native or self-surface query entered policy')
    monkeypatch.setattr(c.oracle,'self_queries',forbidden);monkeypatch.setattr(c.oracle,'label',forbidden)
    monkeypatch.setattr(hand,'nearest',forbidden)
    with frozen(c.model):
        loss,parts,_=obj.batch(q,rows,previous,previous_rows=old,collect_metrics=False)
        for term in ('style_bone','contact_position','contact_direction','collision_self','penetration_cap'):
            g=torch.autograd.grad(parts[term],q,retain_graph=True)[0]
            assert torch.isfinite(g).all() and g.abs().sum()>0,term
        g,old_g=torch.autograd.grad(loss,(q,previous))
    assert g.abs().sum()>0 and old_g.abs().sum()>0


def test_q_learner_updates_jacobian_and_roundtrips_optimizer_replay(hand,tmp_path):
    torch.manual_seed(37);c=CollisionModel(hand);attach_field(c,field_spec(hand))
    learner=QCollisionLearner(c,capacity=64,batch_size=8)
    learner.add(hand.low+torch.rand(12,20)*(hand.high-hand.low))
    before={k:v.clone() for k,v in learner.model.state_dict().items()}
    losses=learner.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    assert any(not torch.equal(v,learner.model.state_dict()[k]) for k,v in before.items())
    second=CollisionModel(hand);attach_field(second,field_spec(hand),learner.model.state_dict())
    torch.save(learner.state_dict(),tmp_path/'learner.pth')
    resumed=QCollisionLearner(second,capacity=64,batch_size=8);resumed.load_state_dict(torch.load(tmp_path/'learner.pth',weights_only=True))
    rng=torch.get_rng_state();learner.update();torch.set_rng_state(rng);resumed.update()
    for k,v in learner.model.state_dict().items():torch.testing.assert_close(v,resumed.model.state_dict()[k],rtol=0,atol=0)
