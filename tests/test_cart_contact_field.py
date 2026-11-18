"""Learned correspondences replace discrete selection in the policy graph."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import trimesh
import warp as wp
from moxian_web.cart_collision import LinkSDF,CollisionModel,SDFLearner,frozen
from moxian_web.cart_contact_field import ContactTeacher
from moxian_web.cart_objective import CartHand,CartObjective,policy_settings
from moxian_web.cart_training import acceptance


@pytest.fixture(autouse=True)
def one_thread():torch.set_num_threads(1)


def test_contact_field_point_direction_and_joint_gradients():
    torch.manual_seed(17);model=LinkSDF().double()
    point=torch.tensor([[.008,.001,.002]],dtype=torch.float64,requires_grad=True)
    direction=torch.tensor([[-1.,.01,.02]],dtype=torch.float64,requires_grad=True)
    q=torch.tensor([.003],dtype=torch.float64,requires_grad=True)
    links=torch.tensor([8]);palmar=torch.tensor([True])
    def contact(x,d,q):
        # Translation-only exact FK fixture checks both coordinate transforms.
        hit,n=model.contact(x-torch.stack((q,q*0,q*0),-1),d,links,palmar)
        return hit+torch.stack((q,q*0,q*0),-1),n
    assert torch.autograd.gradcheck(contact,(point,direction,q),eps=1.e-6,atol=1.e-5,rtol=1.e-4)
    # Cross the former box nearest-face switch at x=0. The learned field and
    # its derivative are continuous instead of changing selected facet.
    derivatives=[]
    for x in (-1.e-7,1.e-7):
        p=point.detach().clone();p[0,0]=x;p.requires_grad_()
        hit,_=contact(p,direction,q)
        derivatives.append(torch.autograd.grad(hit.sum(),p)[0])
    torch.testing.assert_close(derivatives[0],derivatives[1],atol=1.e-5,rtol=1.e-5)


def test_field_can_learn_the_requested_side_through_deep_overlap():
    torch.manual_seed(14);model=LinkSDF();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    def examples():
        point=(torch.rand(64,3)-.5)*.03;point[:,1:]*=.2
        direction=torch.zeros_like(point);direction[:,0]=(torch.randint(2,(64,))*2-1).float()
        hit=point.clone();hit[:,0]=direction[:,0]*.01
        return point,direction,hit
    links=torch.zeros(64,dtype=torch.long);palmar=torch.zeros(64,dtype=torch.bool)
    for _ in range(250):
        x,d,target=examples();hit,normal=model.contact(x,d,links,palmar)
        loss=((hit-target)/.001).square().mean()+.1*(normal-d).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    x,d,target=examples();hit,normal=model.contact(x,d,links,palmar)
    # Includes points beyond the centre and outside the wrong side. This is a
    # simple learnability regression, not a claim of Wuji2 mesh acceptance.
    assert (hit-target).norm(dim=-1).max()<.0005
    assert (normal*d).sum(-1).min()>.99


@pytest.mark.parametrize('device',['cpu','cuda:0'] if torch.cuda.is_available() else ['cpu'])
def test_directional_teacher_retracts_inside_and_outside_back_side(device):
    mesh=trimesh.creation.box(extents=(.02,)*3)
    native=wp.Mesh(points=wp.array(np.asarray(mesh.vertices,np.float32),dtype=wp.vec3,device=device),
        indices=wp.array(np.asarray(mesh.faces,np.int32).ravel(),dtype=int,device=device),support_winding_number=True)
    ids=wp.array([native.id],dtype=wp.uint64,device=device)
    teacher=ContactTeacher.__new__(ContactTeacher);teacher.mesh_ids=ids
    teacher.hand=SimpleNamespace(device=device,collision=SimpleNamespace(mesh_ids=ids))
    point=torch.tensor([[.008,0.,0.],[.015,0.,0.],[-.015,0.,0.]],device=device)
    direction=torch.tensor([[-1.,0.,0.]]*3,device=device)
    hit,normal,valid=teacher.label(point,direction,torch.zeros(3,device=device,dtype=torch.long),torch.zeros(3,device=device,dtype=torch.bool))
    assert valid.all()
    torch.testing.assert_close(hit,torch.tensor([[-.01,0.,0.]]*3,device=device),atol=1.e-7,rtol=0)
    torch.testing.assert_close(normal,direction)


def test_policy_never_queries_native_faces_and_both_heads_update(monkeypatch):
    hand=CartHand('cpu');collision=CollisionModel(hand)
    objective=CartObjective(hand,policy_settings(),collision)
    q=((hand.low+hand.high)/2)[None].requires_grad_();p,r,j,*_=hand.frames(q)
    surface=hand.surface.contact_surface(8);local=hand.tensor(surface['triangles'][0]).mean(0)
    normal=(r[0,8]@hand.tensor(surface['normals'][0])).detach().tolist()
    point=(r[0,8]@local+p[0,8]).detach().tolist()
    row=dict(timestamp_ms=20.,stream_epoch=0,joints_m=j[0].detach().tolist(),tactile_valid=[False]*460,
        tactile_state=[0]*460,tactile_positions_m=np.zeros((460,3)).tolist(),contacts=dict(tactile_ready=True,
        object_contacts=[dict(region='index_distal',position_m=point,normal=normal,force=1.)],
        self_pairs=[dict(regions=['thumb_distal','index_distal'],points_m=[point,point])]))
    def forbidden(*a,**kw):raise AssertionError('Native facet selection entered policy graph')
    monkeypatch.setattr(hand,'nearest',forbidden);monkeypatch.setattr(hand,'pair',forbidden)
    original={k:v.clone() for k,v in collision.model.state_dict().items()}
    with frozen(collision.model):
        loss,parts,_=objective.batch(q,[row],q.detach())
        gradient=torch.autograd.grad(parts['contact']+parts['normal'],q,retain_graph=True)[0]
        assert gradient.isfinite().all() and gradient.abs().sum()>0
        loss.backward()
    assert all(torch.equal(v,original[k]) for k,v in collision.model.state_dict().items())
    learner=SDFLearner(collision,capacity=512,batch_size=64)
    with torch.no_grad():collision.collect_queries(q,[row],hand.frames(q))
    learner.collect();learner.update()
    for prefix in ('network.','contact_field.'):
        assert any(not torch.equal(v,original[k]) for k,v in collision.model.state_dict().items() if k.startswith(prefix))
    assert learner.contact_size>0 and learner.state_dict()['contact_replay']['directions'].shape[-1]==3


def test_bad_contact_field_cannot_pass_native_acceptance():
    quality=dict(frames=100,contact_frames=30,feasible_rate=1.)
    report=dict(distance_p95_mm=.1,penetration_cases=25,penetration_false_negative_rate=0.,
        contact_samples=100,contact_position_p95_mm=2.,contact_normal_p95_deg=80.)
    reasons=acceptance(quality,report)
    assert len(reasons)==2 and all('连续接触场' in r for r in reasons)


def test_fixed_collar_exclusions_cannot_hide_another_penetration():
    collision=CollisionModel.__new__(CollisionModel);collision.auditing=False
    local=torch.zeros(1,2,3);links=torch.tensor([[0,1]])
    collision.oracle=SimpleNamespace(self_queries=lambda frames:(local,links,torch.tensor([[False,True]])),
        tactile_queries=lambda frames,rows:[(torch.empty(0,21,3),torch.empty(0,21,dtype=torch.long))],
        query_sources=torch.tensor([0,1]),query_targets=torch.tensor([0,1]),surface_owners=torch.tensor([1,2]),
        surface=torch.zeros(2,3),collar=torch.tensor([0.,.01,0.]))
    sdf=torch.tensor([-.02,-.001],requires_grad=True)
    collision.distance=lambda points,owners:sdf
    depth,_=collision.penalties(torch.zeros(1,20),[{}],None)
    assert depth.item()==pytest.approx(.001,abs=1.e-7)
    gradient=torch.autograd.grad(depth.sum(),sdf)[0]
    assert gradient[0]==0 and gradient[1]<-.99


@pytest.mark.parametrize('device',['cpu','cuda:0'] if torch.cuda.is_available() else ['cpu'])
def test_batched_finger_fk_matches_native_tree_values_and_gradients(device):
    torch.manual_seed(131);hand=CartHand(device)
    q=(hand.low+torch.rand(7,20,device=device)*(hand.high-hand.low)).requires_grad_()
    expected=hand.fk.frames(torch.cat((q.new_zeros((len(q),6)),q),-1))
    actual=hand.fixed_fk.frames(q)
    weights=[torch.randn_like(v) for v in expected]
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=2.e-7,rtol=2.e-6)
    old=torch.autograd.grad(sum((a*w).sum() for a,w in zip(expected,weights)),q)[0]
    new=torch.autograd.grad(sum((a*w).sum() for a,w in zip(actual,weights)),q)[0]
    torch.testing.assert_close(new,old,atol=3.e-6,rtol=1.e-5)
