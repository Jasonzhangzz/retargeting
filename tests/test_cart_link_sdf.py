"""Link routing, real distance derivatives and the distance-only geometry gate."""
import torch
from moxian_web.cart_link_sdf import IndependentLinkDistance,VALIDATION
from moxian_web.cart_collision import geometry_rejection_reasons
from moxian_web.cart_geometry_schedule import selection_key
from moxian_web.cart_link_sdf_objective import LinkSDFObjective


def test_independent_link_routing_and_distance_autograd():
    torch.manual_seed(3)
    model=IndependentLinkDistance(width=8,depth=2).double()
    x=torch.tensor([[.01,.02,.03],[-.02,.01,.01]],dtype=torch.double,requires_grad=True)
    owner=torch.tensor([2,4])
    assert torch.autograd.gradcheck(lambda a:model(a,owner),(x,),fast_mode=True)
    d=model(x,owner)
    assert d.shape==(2,)
    expected=torch.stack([.05*model.decoders[int(l)](a/.05).squeeze(-1) for a,l in zip(x,owner)])
    torch.testing.assert_close(d,expected,rtol=1.e-12,atol=1.e-12)
    actual=torch.autograd.grad(d.sum(),x,retain_graph=True)[0]
    reference=torch.autograd.grad(expected.sum(),x,retain_graph=True)[0]
    torch.testing.assert_close(actual,reference,rtol=1.e-12,atol=1.e-12)
    d.sum().backward()
    for i,decoder in enumerate(model.decoders):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in decoder.parameters())==(i in (2,4))
    assert torch.isfinite(x.grad).all()


def test_contact_labels_keep_exact_link_pairs_not_finger_wide_exemptions():
    obj=object.__new__(LinkSDFObjective)
    rows=[{'contacts':{'self_pairs':[{'regions':['thumb_distal','index_distal'],'points_m':[[0,0,0],[0,0,0]]}]}},
          {'contacts':{'self_pairs':[{'regions':['thumb_proximal','index_intermediate'],'points_m':[[0,0,0],[0,0,0]]}]}}]
    groups=obj.contact_labels(rows)
    assert set(groups)=={(4,8),(3,7)}
    assert [v[0][0] for v in groups.values()]==[0,1]


def report():
    return dict(geometry_validation=VALIDATION,distance_p95_mm=.3,penetration_cases=100,
        penetration_false_negative_rate=0.,spatial_gradient_norm_mean=1.,spatial_gradient_cosine_mean=.95,
        q_gradient_cosine_mean=.9,q_escape_cost_descent_rate=.95,q_cap_gradient_cosine_mean=.9,
        q_combined_escape_cap_descent_rate=.95,q_combined_escape_cost_descent_rate=.95,
        q_collision_cases=70,q_cap_cases=30,derivative_label_valid_fraction=.98,
        articulated_gradient_cosine_mean=.95,q_false_negative_rate=0.)


def test_distance_geometry_gate_requires_first_derivatives_without_projection_metrics():
    r=report();assert geometry_rejection_reasons(r)==[]
    assert selection_key(r,[])[0]==0 and selection_key(r,[])[1]<float('inf')
    r['articulated_gradient_cosine_mean']=.2
    assert 'articulated_gradient_cosine_mean' in geometry_rejection_reasons(r)
    r=report();r['q_cap_gradient_cosine_mean']=float('nan')
    assert 'q_cap_gradient_cosine_mean' in geometry_rejection_reasons(r)
