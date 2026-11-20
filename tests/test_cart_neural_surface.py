"""Neural CARD has no facet search in its policy graph; teacher remains exact."""
import numpy as np
import pytest
import torch
from moxian_web.cart_card_neural_surface import NeuralCardCostObjective
from moxian_web.cart_card_cost import CardCostObjective,selected_surface
from moxian_web.cart_collision import CollisionModel,frozen
from moxian_web.cart_contact_field import ContactTeacher
from moxian_web.cart_objective import CartHand
from test_cart_card_cost import targets,row

@pytest.fixture(scope='module')
def setup():
    torch.set_num_threads(1);h=CartHand('cpu');c=CollisionModel(h)
    return h,c,NeuralCardCostObjective(h,c)

def test_nearest_teacher_matches_native_card_pads_including_palm(setup):
    h,c,obj=setup;teacher=ContactTeacher(h)
    for region,owner in enumerate(obj.region_owners):
        tri=obj.pad_triangles[region]
        area=torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]).norm(dim=-1)
        face=area.argmax();x=tri[face].mean(0)[None]+obj.pad_normals[region][face][None]*1.e-6
        hit,normal,valid=teacher.label(x,torch.zeros_like(x),torch.tensor([owner]),torch.tensor([True]))
        exact,ids=selected_surface(x,tri)
        assert valid.all();torch.testing.assert_close(hit,exact,atol=2.e-6,rtol=1.e-4)
        torch.testing.assert_close(normal,obj.pad_normals[region][ids],atol=2.e-6,rtol=1.e-4)

def test_policy_contact_continuity_and_collision_backward_never_searches_mesh(setup,monkeypatch):
    import moxian_web.cart_card_neural_surface as module
    h,c,obj=setup;q=((h.low+h.high)/2)[None].requires_grad_();prev=(q.detach()-.01).requires_grad_()
    t=targets(obj,q.detach()+.03);rows=[row(t,0,20.)];previous=[row(t,0,0.)]
    def forbidden(*a,**k):raise AssertionError('Mesh query entered policy cost')
    monkeypatch.setattr(module,'selected_surface',forbidden);monkeypatch.setattr(h,'nearest',forbidden)
    monkeypatch.setattr(c.oracle,'label',forbidden)
    with frozen(c.model):
        loss,parts,_=obj.batch(q,rows,prev,previous_rows=previous,collect_metrics=False)
        g,oldg=torch.autograd.grad(loss,(q,prev))
    assert g.isfinite().all() and g.abs().sum()>0
    assert oldg.isfinite().all() and oldg.abs().sum()>0
    obj.collect_geometry_queries(q.detach(),rows,h.frames(q.detach()))
    assert c.contact_collected and all(not x.requires_grad for values in c.contact_collected for x in values)
    assert all(not values[1].any() for values in c.contact_collected)

def test_cost_expressions_match_card_with_same_projection_operator(setup,monkeypatch):
    h,c,obj=setup;exact=CardCostObjective(h,c)
    q=((h.low+h.high)/2)[None].requires_grad_();previous_q=q.detach()-.01
    t=targets(obj,q.detach()+.02);rows=[row(t,0,20.)];previous=[row(t,0,0.)]
    def project(x,region):
        local,index=selected_surface(x,obj.pad_triangles[region]);return local,obj.pad_normals[region][index]
    monkeypatch.setattr(obj,'project_surface',project)
    with frozen(c.model):
        a,pa,_=obj.batch(q,rows,previous_q,previous_rows=previous,collect_metrics=False)
        b,pb,_=exact.batch(q,rows,previous_q,previous_rows=previous,collect_metrics=False)
        ga=torch.autograd.grad(a,q,retain_graph=True)[0];gb=torch.autograd.grad(b,q)[0]
    torch.testing.assert_close(a,b,rtol=2.e-5,atol=.2)
    for k in pa:torch.testing.assert_close(pa[k],pb[k],rtol=2.e-5,atol=.2)
    torch.testing.assert_close(ga,gb,rtol=2.e-4,atol=2.)
