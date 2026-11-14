import math
import pytest
import torch
from moxian_web.mano_contact_loss import normal_residual,normal_error_deg,vertex_normals,gradient_report


def opposing(theta):
    return torch.stack((theta.sin(),theta*0,-theta.cos()),-1)


def test_normal_cone_is_zero_inside_tolerance_and_has_finite_boundary_gradients():
    theta=torch.deg2rad(torch.tensor([0.,5.,14.,15.,16.,30.,90.,180.],dtype=torch.float64)).requires_grad_()
    a=torch.tensor([0.,0.,1.],dtype=theta.dtype).expand(len(theta),-1)
    loss=normal_residual(a,opposing(theta)).square().sum(-1)
    assert torch.all(loss[:4]<1e-25) and torch.all(loss[4:]>0)
    grad=torch.autograd.grad(loss.sum(),theta)[0]
    assert torch.isfinite(grad).all() and torch.all(grad[:3]==0) and torch.all(grad[4:7]>0)
    torch.testing.assert_close(normal_error_deg(a,opposing(theta.detach())),torch.rad2deg(theta.detach()),atol=1e-6,rtol=0)


def test_angle_scale_and_rotation_invariance_and_finite_difference():
    t=torch.tensor(math.radians(30),dtype=torch.float64,requires_grad=True)
    a=torch.tensor([0.,0.,1.],dtype=t.dtype)
    r=normal_residual(a,opposing(t))
    expected=(2*math.sin(math.pi/12)-2*math.sin(math.pi/24))/(2*math.sin(math.pi/24))
    assert float(r.detach().norm())==pytest.approx(expected)
    assert torch.autograd.gradcheck(lambda x:normal_residual(a,opposing(x)),(t,))
    rotation=torch.tensor([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]],dtype=t.dtype)
    torch.testing.assert_close(normal_residual(a@rotation,opposing(t)@rotation),r@rotation)


def test_smoothed_original_mesh_normals_batching_winding_and_geometry_gradients():
    faces=torch.tensor([[0,1,2],[0,2,3]])
    v=torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,.2],[0.,1.,0.]],dtype=torch.float64,requires_grad=True)
    n=vertex_normals(v,faces)
    tri=v[faces];area=torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    expected=torch.nn.functional.normalize(area.sum(0),dim=0)
    torch.testing.assert_close(n[0],expected);torch.testing.assert_close(n[2],expected)
    torch.testing.assert_close(vertex_normals(torch.stack((v,v*2)),faces),torch.stack((n,n)))
    torch.testing.assert_close(vertex_normals(v,faces.flip(-1)),-n)
    target=opposing(torch.tensor(.6,dtype=v.dtype))
    def residual(vertices):return normal_residual(vertex_normals(vertices,faces)[0],target)
    assert torch.autograd.gradcheck(residual,(v,),atol=1e-5,rtol=1e-3)
    gradient=torch.autograd.grad(residual(v).square().sum(),v)[0]
    assert torch.isfinite(gradient).all() and gradient.norm()>0


def test_action_normalization_is_invariant_to_repeated_contact_frames():
    a=torch.tensor([[0.,0.,1.],[0.,0.,1.]])
    b=opposing(torch.tensor([.4,.8]))
    r=normal_residual(a,b)
    before=(r*torch.tensor([.5,.5])[:,None].sqrt()).square().sum()
    repeated=r[[0,1,1,1]]
    after=(repeated*torch.tensor([.5,1/6,1/6,1/6])[:,None].sqrt()).square().sum()
    torch.testing.assert_close(before,after)


def test_gradient_diagnostics_never_reads_validation():
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='留出'):
        gradient_report(SimpleNamespace(rows=[{'partition':'validation'}]),torch.zeros(89))


def test_cpu_contact_path_uses_same_smoothed_normal_cone():
    import numpy as np
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.mano_runtime import ManoSurface
    from moxian_web.mano_objective import ManoObjective
    if not DEFAULT_MANO.exists():pytest.skip('Licensed MANO required')
    torch.set_num_threads(1)
    m=ManoSurface(DEFAULT_MANO,device='cpu')
    o=ManoObjective(m,[{'stage':'pinch_index','timestamp_ms':0}],np.zeros((1,15,3)),np.eye(3)[None],training=True,shared=True,tactile=False)
    o.contacts=[(0,{'a':{'source_face':0,'barycentric':[.2,.3,.5]},'b':{'source_face':100,'barycentric':[.3,.4,.3]}})]
    x=o.initial.detach().clone().requires_grad_(True)
    loss=o.residual(x,record=True).square().sum()
    v=o.surface(x)[1];faces=torch.as_tensor(m.faces,dtype=torch.long)
    n=vertex_normals(v,faces)[0,faces[[0,100]]]
    bc=torch.tensor([[.2,.3,.5],[.3,.4,.3]])
    n=torch.nn.functional.normalize((n*bc[...,None]).sum(-2),dim=-1)
    expected=normal_residual(n[0],n[1]).square().sum()*40/3
    assert o.last_terms['contact_normal']==pytest.approx(float(expected.detach()),rel=2e-6)
    assert torch.isfinite(torch.autograd.grad(loss,x)[0]).all()
