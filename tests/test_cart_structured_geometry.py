"""Analytic feature projections retain exact first/second AD derivatives."""
from types import SimpleNamespace
import numpy as np
import torch
from moxian_web.cart_structured_geometry import SmoothPadProjection


def field(triangles=None):
    triangles=np.asarray(triangles if triangles is not None else
        [[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],[[1.,1.,0.],[0.,1.,0.],[1.,0.,0.]]])
    surface=SimpleNamespace(contact_surface=lambda owner:dict(triangles=triangles))
    return SmoothPadProjection(SimpleNamespace(surface=surface,device='cpu'),chunk_size=2)


def test_face_edge_and_vertex_projections_and_local_jacobians():
    model=field();points=torch.tensor([[.2,.3,.1],[1.2,.3,.1],[1.2,1.3,.1]],dtype=torch.double,requires_grad=True)
    owner=torch.zeros(3,dtype=torch.long);hit,normal=model(points,owner)
    expected=torch.tensor([[.2,.3,0.],[1.,.3,0.],[1.,1.,0.]],dtype=torch.double)
    torch.testing.assert_close(hit,expected,atol=1e-8,rtol=0)
    j=torch.stack([torch.autograd.grad(hit[:,k].sum(),points,retain_graph=True)[0] for k in range(3)],1)
    target=torch.diag_embed(torch.tensor([[1.,1.,0.],[0.,1.,0.],[0.,0.,0.]],dtype=torch.double))
    torch.testing.assert_close(j,target,atol=1e-6,rtol=0)
    torch.testing.assert_close(normal,torch.tensor([[0.,0.,1.]]*3,dtype=torch.double))


def test_projection_gradcheck_and_gradgradcheck():
    model=field();points=torch.tensor([[.2,.3,.1]],dtype=torch.double,requires_grad=True)
    owner=torch.zeros(1,dtype=torch.long)
    assert torch.autograd.gradcheck(lambda x:model(x,owner)[0],(points,),fast_mode=True)
    assert torch.autograd.gradgradcheck(lambda x:model(x,owner)[0],(points,),fast_mode=True)


def test_soft_edge_transition_has_finite_nonzero_curvature():
    model=field();x=torch.tensor([[.99999,.3,.1]],dtype=torch.double,requires_grad=True)
    owner=torch.zeros(1,dtype=torch.long)
    hit,_=model(x,owner)
    first=torch.autograd.grad(hit[0,0],x,create_graph=True)[0]
    second=torch.autograd.grad(first[0,0],x)[0]
    assert torch.isfinite(second).all() and second[0,0].abs()>1.
    assert torch.autograd.gradcheck(lambda a:model(a,owner)[0],(x,),eps=1e-8,atol=1e-4,rtol=1e-3,fast_mode=True)


def test_skinny_triangle_does_not_admit_remote_plane_extension():
    model=field([[[0.,0.,0.],[1.,0.,0.],[1.-1e-8,1e-8,0.]]])
    x=torch.tensor([[.5,2.,.1]],dtype=torch.double,requires_grad=True)
    hit,_=model(x,torch.zeros(1,dtype=torch.long))
    assert hit[0,1].abs()<1e-6 and hit[0,2].abs()<1e-6
    assert torch.isfinite(torch.autograd.grad(hit.sum(),x)[0]).all()
