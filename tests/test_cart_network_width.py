"""Capacity variants preserve the physical interface and differentiable paths."""
import torch
from moxian_web.cart_palm_alignment import PalmAlignedNetwork


def make_policy(**kwargs):
    return PalmAlignedNetwork(-torch.ones(20),torch.ones(20),
        alignment=dict(translation_m=[0.]*3,rotation_vector_rad=[0.]*3,wrist_mode='unbounded'),**kwargs)


def test_default_checkpoint_layout_unchanged():
    old=make_policy();explicit=make_policy(width=128,decoder_width=256)
    assert sum(p.numel() for p in old.parameters())==1065242
    explicit.load_state_dict(old.state_dict(),strict=True)


def test_wide_policy_full_input_backward_and_checkpoint():
    torch.set_num_threads(1);torch.manual_seed(7)
    m=make_policy(width=384,decoder_width=744)
    assert sum(p.numel() for p in m.parameters())==8819690
    q=m.initial_state(torch.zeros(1,20)).requires_grad_()
    # Open the zero-initialized output heads to test the full encoder gradient.
    for head in [*m.decoders,m.wrist_decoder]:torch.nn.init.normal_(head[-2].weight,std=.001)
    emf=torch.randn(1,10,5,9)*.01
    valid=torch.ones(1,10,5,dtype=torch.bool);dt=torch.arange(1,11)[None]/120
    tactile=torch.randn(1,10,460,3)*.01
    state=torch.zeros(1,10,460);tv=torch.ones_like(state,dtype=torch.bool)
    delta=m(q,emf,valid,dt,tactile,state,tv)
    next_q=m.apply_delta(q,delta)
    assert next_q.shape==(1,26) and torch.isfinite(next_q).all()
    next_q.square().sum().backward()
    for module in [m.emf,m.touch,m.current,m.blocks[0],*m.decoders,m.wrist_decoder]:
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads)>0
    restored=make_policy(width=m.width,decoder_width=m.decoder_width)
    restored.load_state_dict(m.state_dict(),strict=True)
    torch.testing.assert_close(restored(q.detach(),emf,valid,dt,tactile,state,tv),delta.detach())
