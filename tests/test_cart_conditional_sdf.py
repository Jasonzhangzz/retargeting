import torch
from moxian_web.cart_conditional_sdf import ConditionalDistances,norm_penetration_guard,smooth_taxel_depth,VALIDATION
from moxian_web.cart_collision import geometry_rejection_reasons
from moxian_web.cart_geometry_schedule import selection_key
from test_cart_link_sdf import report


def test_q_encoder_reused_for_all_points_and_true_input_derivatives():
    torch.manual_seed(4)
    model=ConditionalDistances(torch.full((20,),-1.),torch.ones(20),width=8,latent=4).double()
    calls=[];handle=model.q_encoder.register_forward_hook(lambda *a:calls.append(True))
    q=(.2*torch.randn(2,20,dtype=torch.double)).requires_grad_()
    x=(.03*torch.randn(2,3,3,dtype=torch.double)).requires_grad_()
    d=model(q,x);assert d.shape==(2,3,21) and len(calls)==1
    handle.remove()
    assert torch.autograd.gradcheck(model,(q,x),fast_mode=True)
    selected=d[...,3].sum();gq,gx=torch.autograd.grad(selected,(q,x),create_graph=True)
    assert gq.norm()>0 and gx.norm()>0
    # The geometry learner differentiates first derivatives into parameters.
    model.zero_grad();(gq.square().sum()+gx.square().sum()).backward()
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.q_encoder.parameters())


def test_v4_norm_guard_counts_multiple_shallow_penetrations():
    d=torch.full((1,4),-.003,dtype=torch.double,requires_grad=True)
    norm,guard=norm_penetration_guard(d,torch.ones_like(d))
    torch.testing.assert_close(norm,torch.tensor([6.],dtype=torch.double))
    torch.testing.assert_close(guard,torch.tensor([1.],dtype=torch.double))
    assert torch.autograd.gradcheck(lambda a:norm_penetration_guard(a,torch.ones_like(a))[1],(d,))
    assert torch.autograd.grad(guard.sum(),d)[0].lt(0).all()


def test_taxel_smooth_depth_keeps_every_row_and_gradient():
    values=torch.full((3,21),.01,dtype=torch.double)
    values[0,4]=-.006;values[2,8]=-.007;values.requires_grad_()
    depth=smooth_taxel_depth(values,[2,0,1])
    assert depth.shape==(3,) and depth[1]==0
    torch.testing.assert_close(depth[[0,2]],torch.tensor([6.,7.],dtype=torch.double))
    assert torch.autograd.gradcheck(lambda a:smooth_taxel_depth(a,[2,0,1]),(values,),fast_mode=True)


def test_conditional_gate_requires_actual_guard_gradient():
    r=report();r.update(geometry_validation=VALIDATION,projection_required=False,q_guard_gradient_cosine_mean=.9)
    assert geometry_rejection_reasons(r)==[] and selection_key(r,[])[0]==0
    r['q_guard_gradient_cosine_mean']=.7
    assert 'q_guard_gradient_cosine_mean' in geometry_rejection_reasons(r)
