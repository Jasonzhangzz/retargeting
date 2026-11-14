import math
import numpy as np
import pytest
import torch
from moxian_web.qnbot_link_contact import LinkContactQuery
from moxian_web.mano_contact_loss import normal_residual,normal_error_deg
from moxian_web.qnbot_surface import QnbotSurface
from moxian_web.thumb_limits import THUMB_EXTENSION_DEG


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_exact_distance_gradient_and_moving_witnesses(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA')
    v=torch.tensor([[[0.,0,0],[1,0,0],[0,1,0],[0,0,.01],[0,1,.01],[1,0,.01]]],device=device,requires_grad=True)
    f=torch.tensor([[0,1,2],[3,4,5]],device=device);q=LinkContactQuery(v,f,{'a':[0],'b':[1]},[dict(row=0,regions=['a','b'])])
    points,n,_=q.evaluate(v);gap=(points[:,0]-points[:,1]).norm(dim=-1)
    assert float(gap.detach()[0])==pytest.approx(.01,abs=1e-7)
    assert float(normal_error_deg(n[:,0],n[:,1])[0].detach())<.01
    gap.square().sum().backward();assert torch.isfinite(v.grad).all() and v.grad.norm()>0
    with torch.no_grad():v[:,3:,2]=.02
    moved,_,_=q.evaluate(v);assert float((moved[:,0]-moved[:,1]).norm().detach())==pytest.approx(.02,abs=1e-7)
    if device.startswith('cuda'):assert q.graph is not None,q.graph_error
    # Disabled batch edges are not silently queried/used by a contact loss.
    disabled,_,_=q.evaluate(v,torch.tensor([False],device=device));assert torch.count_nonzero(disabled)==0


def test_edge_edge_interior_and_intersection_are_not_vertex_only_distances():
    # Nearest points are in the middle of two skew edges, not any vertex.
    v=torch.tensor([[[-1.,0,0],[1,0,0],[0,-1,0],[0,-1,.1],[0,1,.1],[0,0,1.]]])
    f=torch.tensor([[0,1,2],[3,4,5]]);q=LinkContactQuery(v,f,{'a':[0],'b':[1]},[dict(row=0,regions=['a','b'])])
    p,_,_=q.evaluate(v);assert float((p[:,0]-p[:,1]).norm())==pytest.approx(.1,abs=1e-6)
    v[:,3:,2]-=.2
    p,n,_=q.evaluate(v);assert float((p[:,0]-p[:,1]).norm())<1e-6
    assert normal_residual(n[:,0],n[:,1],tolerance_deg=0.).square().sum()>0


def test_antiparallel_normal_target_has_no_fifteen_degree_dead_zone():
    a=torch.tensor([[0.,0.,1.]]);angle=math.radians(5)
    b=torch.tensor([[0.,math.sin(angle),-math.cos(angle)]],requires_grad=True)
    loss=normal_residual(a,b,tolerance_deg=0.).square().sum()
    assert loss>0;loss.backward();assert b.grad.norm()>0
    assert normal_residual(a,-a,tolerance_deg=0.).square().sum()<1e-10


def test_equal_distance_selects_opposing_normals_at_same_material_points():
    v=torch.tensor([[[0.,0,0],[3,0,0],[0,3,0],[0,0,.01],[1,0,.01],[0,1,.01],[2,0,.01],[2,1,.01],[3,0,.01]]])
    f=torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
    q=LinkContactQuery(v,f,{'a':[0],'b':[1,2]},[dict(row=0,regions=['a','b'])]);p,n,w=q.evaluate(v)
    assert float((p[:,0]-p[:,1]).norm())==pytest.approx(.01,abs=1e-6)
    assert float(normal_error_deg(n[:,0],n[:,1])[0])<.1
    assert w['triangles'][0,1].min()>=6


@pytest.mark.parametrize('side',['left','right'])
def test_native_thumb_has_signed_extension_relative_to_straight(side):
    m=QnbotSurface(side)
    for name,straight in m.thumb_straight.items():
        i=m.joint_names.index(name);physical_low=float(m.limits[i,0])-straight
        assert math.degrees(physical_low)<=-THUMB_EXTENSION_DEG+1e-4
        q=m.q0[None].repeat(2,1);q[:,i]=torch.tensor([straight-.2,straight+.2])
        with torch.no_grad():r,p=m.fk(q)
        assert torch.isfinite(p).all() and torch.linalg.norm(r[0]-r[1])>.1
