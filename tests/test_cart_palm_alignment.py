import numpy as np
import torch
from moxian_web.cart_palm_alignment import (palm_registration,PalmAlignedNetwork,wrist_residual,
    so3_exp,rotation_vector,LIMITS)


def network():
    return PalmAlignedNetwork(torch.full((20,),-1.),torch.ones(20),'attention',
        alignment=dict(translation_m=[.01,-.012,.025],rotation_vector_rad=[.2,-.1,.05])).double()


def test_palm_registration_known_transform_and_no_scale():
    a=np.zeros((21,3));a[[5,9,13,17]]=np.array([[.002,.02,.08],[0,0,.09],[.002,-.02,.08],[.007,-.04,.07]])
    r=so3_exp(torch.tensor([.1,.3,-.2],dtype=torch.double)).numpy();t=np.array([.03,-.01,.02])
    actual_r,actual_t=palm_registration(a,a@r.T+t)
    np.testing.assert_allclose(actual_r,r,atol=1.e-12);np.testing.assert_allclose(actual_t,t,atol=1.e-12)
    assert np.linalg.det(actual_r)>0


def test_residual_roundtrip_and_zero_delta_preserves_alignment():
    m=network();q=m.initial_state(torch.zeros(2,20,dtype=torch.double))
    torch.testing.assert_close(m.normalize(q)[:,20:],torch.zeros(2,6,dtype=torch.double),atol=1.e-12,rtol=0)
    torch.testing.assert_close(m.apply_delta(q,torch.zeros_like(q)),q,atol=1.e-12,rtol=0)
    u=torch.randn(3,26,dtype=torch.double)*.1
    torch.testing.assert_close(m.normalize(m.unnormalize(u)),u,atol=1.e-10,rtol=1.e-10)


def test_long_rollout_cannot_accumulate_unbounded_wrist_drift():
    m=network();q=m.initial_state(torch.zeros(1,20,dtype=torch.double))
    for _ in range(1000):q=m.apply_delta(q,torch.ones_like(q))
    residual=wrist_residual(q,m.palm_base)
    assert (residual.abs()<=torch.tensor(LIMITS)+1.e-6).all()


def test_rotation_and_residual_backward_at_zero_and_nonzero():
    m=network();u=torch.zeros(1,26,dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:m.unnormalize(x),(u,),fast_mode=True)
    x=torch.tensor([[.1,.02,-.03]],dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda v:rotation_vector(so3_exp(v)),(x,),fast_mode=True)


def test_unbounded_wrist_has_no_pose_or_action_clipping_and_keeps_finger_bounds():
    from moxian_web.cart_palm_alignment import unbounded_wrist_alignment
    alignment=unbounded_wrist_alignment(dict(translation_m=[0.,0.,0.],rotation_vector_rad=[0.,0.,0.]))
    m=PalmAlignedNetwork(torch.full((20,),-1.),torch.ones(20),'attention',alignment=alignment).double()
    q=m.initial_state(torch.zeros(1,20,dtype=torch.double))
    delta=torch.zeros_like(q);delta[:,:20]=5;delta[:,20]=3;delta[:,23]=3
    delta.requires_grad_();result=m.apply_delta(q,delta)
    torch.testing.assert_close(result[:,20],torch.tensor([.06],dtype=torch.double),rtol=1e-6,atol=1e-9)
    torch.testing.assert_close(result[:,23],torch.tensor([.6],dtype=torch.double),rtol=1e-6,atol=1e-9)
    assert result[:,:20].abs().max()<=1 and m.normalize(result)[:,20]>1
    gradient=torch.autograd.grad(result[:,20:].sum(),delta)[0]
    assert torch.isfinite(gradient).all() and (gradient[:,20:].abs()>0).all()
    assert isinstance(m.wrist_decoder[-1],torch.nn.Identity)
    assert torch.autograd.gradcheck(lambda x:m.apply_delta(q,x),(delta,),fast_mode=True)
