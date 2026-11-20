"""CARD energy parity, piecewise projection derivatives, and neural collision."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from moxian_web.cart_card_cost import CardCostObjective, selected_surface
from moxian_web.cart_objective import CartHand
from moxian_web.cart_collision import CollisionModel, frozen
from retarget.baseline.card.retarget import TactileRetarget


@pytest.fixture(autouse=True)
def one_thread():torch.set_num_threads(1)


@pytest.fixture(scope='module')
def setup():
    hand=CartHand('cpu');collision=CollisionModel(hand)
    return hand,collision,CardCostObjective(hand,collision)


@pytest.mark.parametrize('point', [[.2,.3,.4],[.6,.6,.3],[-.2,-.3,.5]])
def test_selected_card_surface_has_correct_face_edge_vertex_derivative(point):
    tri=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]],dtype=torch.float64)
    q=torch.tensor([point],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:selected_surface(x,tri)[0],(q,),eps=1.e-6,atol=1.e-5)


def targets(obj,q):
    p,r,_,*_=obj.hand.frames(q)
    owners=[obj.region_owners[i] for i in (3,6,9,12,15)]
    apex=p[:,owners]+(r[:,owners]@obj.apices[[3,6,9,12,15],:,None]).squeeze(-1)
    chains=torch.cat((p[:,obj.region_owners[1:]].reshape(len(q),5,3,3),apex[:,:,None]),2)
    directions=torch.nn.functional.normalize(chains[:,:,1:]-chains[:,:,:-1],dim=-1)
    c=dict(tips=apex.detach()+.001,normals=(r[:,owners]@obj.hand.tensor(obj.hand.surface.normals[owners])[...,None]).squeeze(-1).detach(),
        directions=directions.detach(),mask=torch.zeros(len(q),16,dtype=torch.bool),
        anchor_position=torch.zeros(len(q),16,3),anchor_normal=torch.zeros(len(q),16,3),
        force=torch.zeros(len(q),16),u=torch.full((len(q),16),.7))
    for region in (6,9):
        owner=obj.region_owners[region];pad=obj.pad_triangles[region][0].mean(0)
        c['mask'][:,region]=True;c['force'][:,region]=1.+region/10
        c['anchor_position'][:,region]=(r[:,owner]@pad[:,None]).squeeze(-1)+p[:,owner]+.002
        c['anchor_normal'][:,region]=(r[:,owner]@obj.pad_normals[region][0,:,None]).squeeze(-1)
    return c


def row(c,i,stamp):
    return dict(timestamp_ms=stamp,stream_epoch=0,_card_targets={k:v[i] for k,v in c.items()},
        tactile_positions_m=np.zeros((460,3)),tactile_valid=[False]*460,tactile_state=[0]*460,
        contacts=dict(tactile_ready=True,self_pairs=[],object_contacts=[]))


def reference_owner(obj):
    owner=TactileRetarget.__new__(TactileRetarget)
    owner.settings=replace(obj.card,wself=0.)
    owner.tensor=lambda a,dtype=torch.float32:torch.as_tensor(np.asarray(a),dtype=dtype)
    owner.frames=lambda t,r,q:obj.hand.frames(q)[:2]
    owner.region_body=torch.tensor(obj.region_owners);owner.region_owners=np.array(obj.region_owners)
    owner.surface=obj.hand.surface;owner.pools=[obj.hand.tensor(owner.surface.candidates(o)['points']) for o in obj.region_owners]
    owner.pool_normals=[obj.hand.tensor(owner.surface.candidates(o)['normals']) for o in obj.region_owners]
    owner.pad_triangles=obj.pad_triangles;owner.pad_normals=obj.pad_normals;owner.apices=obj.apices
    owner.pad_face_ids=[torch.arange(len(tri)) for tri in obj.pad_triangles]
    owner.cap_a=torch.zeros(16,3);owner.cap_b=torch.ones(16,3);owner.radii=torch.zeros(16)
    owner.pair_a=torch.tensor([1,0]);owner.pair_b=torch.tensor([4,2]);owner.cross_pairs=[(1,4)];owner.palm_pairs=[(0,2)]
    owner.mesh_collision=SimpleNamespace(query=lambda p,r,dense:(torch.zeros(len(p),1,1),torch.ones(len(p),dtype=torch.bool)))
    return owner


def test_all_noncollision_parts_match_actual_card_evaluator(setup):
    hand,collision,obj=setup
    q=((hand.low+hand.high)/2)[None].repeat(2,1)
    c=targets(obj,q+.02);rows=[row(c,i,20.) for i in range(2)]
    previous=[row(c,i,0.) for i in range(2)];ref=q-.01
    with frozen(collision.model):loss,parts,_=obj.batch(q,rows,ref,previous_rows=previous,collect_metrics=False)
    owner=reference_owner(obj);old_frames=hand.frames(ref);_,_,old_local=obj.contact_points(old_frames,c)
    terms=('style_bone','style_tip','style_normal','contact_position','contact_direction','contact_fix','smooth_joint')
    actual={k:[] for k in terms}
    for i in range(2):
        owner.previous_contacts={k:v[i:i+1] for k,v in enumerate(old_local) if v is not None}
        context={k:c[k][i] for k in ('tips','normals','directions')}
        context.update(anchors=[dict(region=k,position_m=c['anchor_position'][i,k].tolist(),direction=c['anchor_normal'][i,k].tolist(),
            force=float(c['force'][i,k]),u=float(c['u'][i,k]),segment=(k-1)%3) for k in c['mask'][i].nonzero().flatten().tolist()],
            free=torch.ones(5),previous=(torch.zeros(3),torch.eye(3),ref[i]),sdf=SimpleNamespace(kind='pseudo_object',metadata=lambda:{'ready':False}),
            self_pairs=[],motion_joint=None)
        _,detail=owner.evaluate(torch.zeros(1,3),torch.eye(3)[None],q[i:i+1],context,details=True)
        for k in terms:actual[k].append(detail['parts'][k][0])
    for k in terms:torch.testing.assert_close(parts[k],torch.stack(actual[k]).mean(),rtol=3.e-5,atol=.2)
    primary=terms+('collision_self','collision_object','link_continuity','penetration_cap')
    torch.testing.assert_close(loss,sum(parts[k] for k in primary))


def test_card_bone_gradient_matches_finite_difference(setup):
    hand,collision,obj=setup;q=((hand.low+hand.high)/2)[None].requires_grad_()
    c=targets(obj,q.detach()+.05);rows=[row(c,0,20.)]
    with frozen(collision.model):
        _,parts,_=obj.batch(q,rows,collect_metrics=False)
        grad=torch.autograd.grad(parts['style_bone'],q)[0]
        eps=1.e-3;dq=torch.zeros_like(q);dq[0,6]=eps
        _,plus,_=obj.batch(q.detach()+dq,rows,collect_metrics=False)
        _,minus,_=obj.batch(q.detach()-dq,rows,collect_metrics=False)
    assert grad.isfinite().all() and grad.abs().sum()>0
    torch.testing.assert_close(grad[0,6],(plus['style_bone']-minus['style_bone'])/(2*eps),atol=.03,rtol=.02)


def test_five_mm_cap_and_link_continuity_ignore_discontinuities(setup,monkeypatch):
    hand,collision,obj=setup;q=((hand.low+hand.high)/2)[None].requires_grad_()
    c=targets(obj,q.detach());rows=[row(c,0,20.)];prev=[row(c,0,0.)]
    monkeypatch.setattr(obj,'collision_cost',lambda q,*a:(q[:,0]*0,q[:,0]*0,q[:,0]*0+.006))
    with frozen(collision.model):
        loss,parts,_=obj.batch(q,rows,q-.01,previous_rows=prev,collect_metrics=False)
        torch.testing.assert_close(parts['penetration_cap'],torch.tensor(1600.),rtol=1.e-5,atol=.01)
        assert parts['link_continuity']>0 and obj.settings.penetration_limit_m==.005
        rows[0]['stream_epoch']=1
        _,broken,_=obj.batch(q,rows,q-.01,previous_rows=prev,collect_metrics=False)
        assert broken['link_continuity']==broken['contact_fix']==0
        monkeypatch.setattr(obj,'collision_cost',lambda q,*a:(q[:,0]*0,q[:,0]*0,q[:,0]*0+.004))
        _,inside,_=obj.batch(q,rows,collect_metrics=False)
        assert inside['penetration_cap']==0


def test_card_cost_keeps_gradient_to_previous_predicted_state(setup):
    hand,collision,obj=setup;q=((hand.low+hand.high)/2)[None]
    ref=(q-.01).requires_grad_();c=targets(obj,q+.02)
    with frozen(collision.model):
        _,parts,_=obj.batch(q,[row(c,0,20.)],ref,previous_rows=[row(c,0,0.)],collect_metrics=False)
        grad=torch.autograd.grad(parts['contact_fix']+parts['smooth_joint'],ref)[0]
    assert grad.isfinite().all() and grad.abs().sum()>0
