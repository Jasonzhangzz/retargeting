"""Loss/gradient parity, active witnesses and source-label temporal semantics."""
import numpy as np
import pytest
import torch
from retarget.baseline.geort.gpu import nearest_chamfer
from retarget.baseline.geort.training import reference
from retarget.baseline.geort.constraints import max_penetration_loss
from retarget.baseline.geort.source_targets import contact_hysteresis


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_nearest_chamfer_preserves_loss_and_both_input_gradients(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA required')
    torch.manual_seed(43)
    x=(torch.randn(2,19,3,device=device)*.03).requires_grad_();y=(torch.randn(2,23,3,device=device)*.03).requires_grad_()
    x2=x.detach().clone().requires_grad_();y2=y.detach().clone().requires_grad_()
    expected=reference('loss').chamfer_distance(x,y);actual=nearest_chamfer(x2,y2)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-9)
    expected.backward();actual.backward()
    torch.testing.assert_close(x.grad,x2.grad,rtol=1e-5,atol=1e-8)
    torch.testing.assert_close(y.grad,y2.grad,rtol=1e-5,atol=1e-8)
    # Equal-distance tie: retain the same first-point convention as torch.min.
    a=torch.tensor([[[0.,0.,0.]]],device=device,requires_grad=True)
    b=torch.tensor([[[-1.,0.,0.],[1.,0.,0.]]],device=device,requires_grad=True)
    aa=a.detach().clone().requires_grad_();bb=b.detach().clone().requires_grad_()
    reference('loss').chamfer_distance(a,b).backward();nearest_chamfer(aa,bb).backward()
    torch.testing.assert_close(a.grad,aa.grad);torch.testing.assert_close(b.grad,bb.grad)


def test_sparse_penetration_keeps_maximum_value_and_tied_witness_gradients():
    torch.manual_seed(10)
    p=torch.tensor([[[0.,0.,0.],[.001,0.,0.]],[[0.,0.,0.],[.002,0.,0.]]],requires_grad=True)
    r=torch.eye(3).expand(2,2,3,3).clone().requires_grad_();points=torch.zeros(3,3);owners=torch.tensor([0,0,0])
    hits=torch.zeros(2,3,2,3);normals=torch.zeros_like(hits);normals[:,:,:,0]=1
    valid=torch.zeros(2,3,2,dtype=torch.bool);valid[0,:,1]=True  # Duplicate deepest witnesses; frame 2 is clear.
    pp=p.detach().clone().requires_grad_();rr=r.detach().clone().requires_grad_()
    world=(r[:,owners]@points[...,None]).squeeze(-1)+p[:,owners]
    target=(r[:,None]@hits[...,None]).squeeze(-1)+p[:,None]
    normal=(r[:,None]@normals[...,None]).squeeze(-1)
    depth=torch.where(valid,(-((world[:,:,None]-target)*normal).sum(-1)).clamp_min(0),0.)
    expected=(depth.amax((1,2))/.0005).square().mean()
    actual=max_penetration_loss(pp,rr,points,owners,hits,normals,valid)
    torch.testing.assert_close(actual,expected);expected.backward();actual.backward()
    torch.testing.assert_close(p.grad,pp.grad);torch.testing.assert_close(r.grad,rr.grad)
    assert pp.grad[1].abs().sum()==0


def test_contact_hysteresis_resets_on_gap_or_timestamp_regression():
    gaps=np.tile(np.array([1.,3.,5.,3.,1.,3.,3.])[:,None],(1,10));depth=np.zeros_like(gaps)
    # Stay active at 3mm, leave at 5mm; a long gap and timestamp regression reset it.
    stamps=[0,10,20,30,40,400,390]
    expected=[True,True,False,False,True,False,False]
    assert contact_hysteresis(gaps,depth,stamps)[:,0].tolist()==expected
    depth[:,0]=.2
    assert contact_hysteresis(gaps,depth,stamps)[:,0].all()


def test_robot_contact_targets_exclude_out_of_domain_deep_and_invalid_poses(tmp_path,monkeypatch):
    from retarget.baseline.geort.contact_sampling import robot_targets
    from retarget.baseline.geort.adapter import RobotAdapter
    import retarget.baseline.card.collision as module
    robot=RobotAdapter();q=np.zeros((4,20),np.float32);q[1,6]=-.5;q[2,6]=.2;q[3,6]=.3
    points=np.broadcast_to(np.arange(4,dtype=np.float32)[:,None,None],(4,5,3)).copy()*.01
    labels=tmp_path/'kinematics.npz';np.savez(labels,qpos=q,keypoints=points)
    calls=[]
    class Collision:
        def __init__(self,*a):pass
        def query(self,p,r,dense):
            calls.append(len(p));assert dense and len(p)==3
            depth=torch.zeros((len(p),21,21));depth[1,0,1]=6.
            return depth,torch.tensor([True,True,False])
    monkeypatch.setattr(module,'WujiMeshCollision',Collision)
    domain=dict(lower=[-.1]*20,upper=[.5]*20)
    target,meta=robot_targets(tmp_path,robot,domain,labels,'cpu',lambda *a,**k:None,lambda:None)
    assert target.shape==(5,1,3) and np.array_equal(target[:,0],points[0])
    assert meta['accepted_samples']==meta['rejected_output_domain_samples']==1
    assert meta['rejected_penetration_samples']==meta['rejected_invalid_samples']==1
    target2,meta2=robot_targets(tmp_path,robot,domain,labels,'cpu',lambda *a,**k:None,lambda:None)
    assert calls==[3] and np.array_equal(target,target2) and meta==meta2


def test_weighted_target_duplicates_preserve_chamfer_and_parameter_gradients():
    import copy
    from retarget.baseline.geort.training import training_losses
    torch.set_num_threads(1);torch.manual_seed(5)
    x=torch.randn(1,8,3,requires_grad=True);y=torch.randn(1,4,3,requires_grad=True)
    xx=x.detach().clone().requires_grad_();yy=y.detach().clone().requires_grad_();counts=torch.tensor([3,1,4,2])
    expected=reference('loss').chamfer_distance(x,y.repeat_interleave(counts,dim=1));actual=nearest_chamfer(xx,yy,counts)
    torch.testing.assert_close(expected,actual);expected.backward();actual.backward()
    torch.testing.assert_close(x.grad,xx.grad);torch.testing.assert_close(y.grad,yy.grad)
    net=reference('model');parts=[list(range(4*i,4*i+4)) for i in range(5)]
    ik=net.IKModel(parts);fk=net.FKModel(parts).eval();other=copy.deepcopy(ik)
    points=torch.randn(8,5,3)*.03;robot=np.random.default_rng(5).normal(size=(5,12,3)).astype(np.float32)*.03
    torch.manual_seed(42);np.random.seed(42);a,ap=training_losses(points,ik,fk,robot)
    torch.manual_seed(42);np.random.seed(42);b,bp=training_losses(points,other,fk,robot,efficient=True)
    for key in ap:torch.testing.assert_close(ap[key],bp[key],rtol=1e-5,atol=1e-8)
    a.backward();b.backward()
    for pa,pb in zip(ik.parameters(),other.parameters()):torch.testing.assert_close(pa.grad,pb.grad,rtol=2e-4,atol=1e-5)


def test_contact_geometry_buffer_reuse_does_not_mutate_retained_gradients():
    from retarget.baseline.geort.adapter import RobotAdapter
    from retarget.baseline.geort.constraints import ContactGeometry
    torch.set_num_threads(1);robot=RobotAdapter();geometry=ContactGeometry(robot,'cpu')
    q=robot.lower+(robot.upper-robot.lower)*.9
    x=torch.tensor(robot.normalize(q)[None],requires_grad=True)
    source=torch.randn(1,21,3)*.03;contact=torch.ones(1,10,dtype=torch.bool)
    before,_=geometry.losses(x,source,contact)
    # A second forward overwrites the same query buffers before first backward.
    geometry.losses(x*.5,source,contact)
    before.backward();gradient=x.grad.clone()
    x.grad=None;after,_=geometry.losses(x,source,contact);after.backward()
    torch.testing.assert_close(before,after);torch.testing.assert_close(gradient,x.grad)
    assert len(geometry._buffers)==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA batched mesh query required')
def test_batched_original_skin_queries_match_legacy_and_keep_frames_isolated():
    import trimesh
    from retarget.baseline.geort.source_targets import BatchSourceContacts
    from retarget.baseline.geort.diagnostic_queries import SurfacePairs
    from retarget.baseline.geort.diagnostics import source_allowed
    from retarget.baseline.geort.constraints import PAIRS
    from retarget.baseline.geort.adapter import FINGERS
    from moxian_web.qnbot_collision import QnbotCollision
    names=['right_wrist']
    for finger in ('thumb','index','middle','ring','little'):
        names += ['right_'+finger+'_'+p for p in (('metacarpal','proximal','distal') if finger=='thumb' else ('metacarpal','proximal','intermediate','distal'))]
    distal=[names.index('right_'+('little' if f=='pinky' else f)+'_distal') for f in FINGERS]
    meshes=[]
    for part in range(20):
        mesh=trimesh.creation.box(extents=[.02]*3)
        center=[-.3,.05*part,0.] if part not in distal else [[0.,.025,.15,.22,.29][distal.index(part)],0.,0.]
        mesh.apply_translation(center);meshes.append(mesh)
    mesh=trimesh.util.concatenate(meshes)
    class Source:
        device='cuda:0';bone_names=names
        vertices0=torch.tensor(np.array(mesh.vertices),device=device,dtype=torch.float32)
        faces=torch.tensor(np.array(mesh.faces),device=device,dtype=torch.long)
        weights=torch.nn.functional.one_hot(torch.arange(20,device=device).repeat_interleave(8),20).float()
        q0=torch.zeros(29,device=device)
        def __call__(self,q):return (self.vertices0[None].expand(len(q),-1,-1),)
    model=Source();top=dict(regions={f+'_distal':list(range(part*12,(part+1)*12)) for f,part in zip(FINGERS,distal)})
    fast=BatchSourceContacts(model,top,batch=4)
    old=QnbotCollision(model,dense=True);allowed=torch.tensor(source_allowed(names),device=model.device)
    query=SurfacePairs(model.vertices0,model.faces,top['regions'],[dict(regions=[FINGERS[a]+'_distal',FINGERS[b]+'_distal']) for a,b in PAIRS])
    vertices=model.vertices0[None].repeat(2,1,1)
    vertices[1,distal[1]*8:(distal[1]+1)*8,0]-=.01
    expected_gap=[];expected_depth=[]
    for v in vertices:
        gap,_=query.evaluate(v);d=old.query(v,part_allowed=allowed)['part_depths_mm']
        expected_gap.append(gap)
        expected_depth.append(torch.stack([torch.maximum(d[distal[a],distal[b]],d[distal[b],distal[a]]) for a,b in PAIRS]))
    gaps,depth=fast.query(vertices)
    torch.testing.assert_close(gaps,torch.tensor(expected_gap,device=model.device),rtol=1e-5,atol=1e-4)
    torch.testing.assert_close(depth,torch.stack(expected_depth),rtol=1e-5,atol=1e-4)
    assert depth[0,0]==0 and depth[1,0]>.1
    before=gaps.clone();later,_=fast.query(vertices.flip(0))
    torch.testing.assert_close(later,before.flip(0))
