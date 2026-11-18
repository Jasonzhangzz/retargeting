"""Behavioral coverage for future inputs, BPTT, online geometry and deployment."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from moxian_web.cart_network import CartNetwork,FutureBuffer
from moxian_web.cart_observations import canonical_emf,pack_future
from moxian_web.cart_collision import LinkSDF,CollisionModel,SDFLearner,frozen
from moxian_web.cart_objective import CartHand,CartObjective,policy_settings
from moxian_web.cart_training import rollout,position_statistics,save_checkpoint


@pytest.fixture(autouse=True)
def one_thread(): torch.set_num_threads(1)


def inputs(batch=2):
    return dict(current_q=torch.rand(batch,20),future_emf=torch.randn(batch,10,5,9)*.01,
        emf_valid=torch.ones(batch,10,5,dtype=torch.bool),future_dt=torch.arange(1,11)[None].expand(batch,-1)*.02,
        tactile_positions=torch.rand(batch,10,460,3)*.1,tactile_state=torch.zeros(batch,10,460),
        tactile_valid=torch.ones(batch,10,460,dtype=torch.bool))


@pytest.mark.parametrize('architecture',['attention','fsq'])
def test_future_policy_gradients_bounds_and_unknowns(architecture):
    torch.manual_seed(1);model=CartNetwork(torch.zeros(20),torch.ones(20),architecture)
    args=inputs();args['current_q'].requires_grad_();args['future_emf'].requires_grad_()
    output=model(**args);output.sum().backward()
    assert output.shape==(2,20) and output.abs().max()<=1
    assert args['current_q'].grad.abs().sum()>0 and args['future_emf'].grad.abs().sum()>0
    assert model.decoder[-2].weight.grad.abs().sum()>0
    if architecture=='fsq':assert model.encoder[-1].weight.grad.abs().sum()>0
    with torch.no_grad():
        original=model(**args);args['tactile_state'][:]=1;changed=model(**args)
        if architecture=='fsq':torch.testing.assert_close(original,changed)
        else:assert not torch.allclose(original,changed)
        args['tactile_valid'][:]=False;unknown=model(**args)
        args['tactile_positions'][:]=float('nan');args['tactile_state'][:]=float('nan')
        torch.testing.assert_close(unknown,model(**args))
        args['emf_valid'][:]=False;args['future_emf'][:]=float('nan')
        assert model(**args).isfinite().all()
    with pytest.raises(ValueError):model(**dict(args,future_dt=torch.zeros(2,10)))


def test_unknown_tactile_is_distinct_from_valid_zero():
    model=CartNetwork(torch.zeros(20),torch.ones(20));args=inputs()
    a=model(**args);args['tactile_valid'][:]=False;b=model(**args)
    assert not torch.allclose(a,b)


def test_emf_rotation_and_left_reflection():
    from scipy.spatial.transform import Rotation
    rotation=Rotation.from_rotvec([.3,-.2,.1]).as_matrix()
    obs=dict(timestamp_ms=100.,coordinate_frame='wrist_local',positions_m=[[1.,2.,3.]]*5,
        rotations=np.broadcast_to(rotation,(5,3,3)).tolist(),valid=[True]*5,
        raw_positions_m=[[1.,2.,3.]]*5,raw_rotations=np.broadcast_to(rotation,(5,3,3)).tolist())
    a=np.diag([-1.,1.,1.]);pose,valid=canonical_emf(obs,SimpleNamespace(transform=a),np.array([.1,.2,.3]),100.)
    np.testing.assert_allclose(pose[0,:3],[-.9,1.8,2.7],atol=1.e-6)
    expected=a.T@rotation@a
    np.testing.assert_allclose(pose[0,3:].reshape(2,3).T,expected[:,:2],atol=1.e-6)
    assert np.linalg.det(expected)==pytest.approx(1.) and valid.all()
    with pytest.raises(ValueError):canonical_emf(obs,SimpleNamespace(transform=a),np.zeros(3),200.)


def sensor_row(stamp):
    return dict(timestamp_ms=float(stamp),emf_pose=np.zeros((5,9)).tolist(),emf_valid=[True]*5,
        tactile_positions_m=np.zeros((460,3)).tolist(),tactile_state=[0]*460,tactile_valid=[True]*460,
        contacts=dict(tactile_ready=True,self_pairs=[],object_contacts=[]))


def test_exact_horizon_pack_and_buffer_boundaries():
    buffer=FutureBuffer();rows=[sensor_row(i*20.) for i in range(12)]
    for i in range(10):assert buffer.append(rows[i]['timestamp_ms'],rows[i],0) is None
    window=buffer.append(200.,rows[10],0)
    assert len(window)==11 and window[0][0]==0 and window[-1][0]==200
    packed=pack_future(0.,[x[1] for x in window[1:]])
    before=packed['future_emf'].clone();rows[11]['emf_pose'][0][0]=1000
    torch.testing.assert_close(before,pack_future(0.,[x[1] for x in window[1:]])['future_emf'])
    assert buffer.append(220.,rows[11],1) is None and len(buffer.entries)==1
    assert buffer.append(600.,sensor_row(600.),1) is None and len(buffer.entries)==1
    assert buffer.append(600.,sensor_row(600.),1) is None and len(buffer.entries)==1
    with pytest.raises(ValueError):pack_future(0.,rows[:9])


def test_rollout_bptt_does_not_detach_feedback_and_uses_real_dt():
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.2));self.calls=[]
        def forward(self,q,**kwargs):
            value=q*self.weight+.1;value.retain_grad() if value.requires_grad else None
            self.calls.append(value);return value
        def unnormalize(self,q):return q
    class Objective:
        collision=None
        hand=SimpleNamespace(device='cpu')
        def batch(self,q,rows,reference,acceleration,**kwargs):
            if acceleration is not None:
                expected=((q-reference)/.04-reference/.02)/.03
                torch.testing.assert_close(acceleration,expected)
            # Only the last output contributes: first output must still get grad.
            return (q.square().mean() if rows[0]['timestamp_ms']>=60 else q.sum()*0),{},[]
    rows=[sensor_row(t) for t in [0.,20.,60.,*range(80,260,20)]]
    model=Model()
    with patch('moxian_web.cart_training.neutral',return_value=torch.zeros(20)):
        loss,_,_,predictions=rollout(model,Objective(),[dict(rows=rows,length=2)])
    loss.backward()
    assert len(predictions)==2 and model.calls[0].grad.abs().sum()>0


@pytest.fixture(scope='module')
def geometry():
    hand=CartHand('cpu');return hand,CollisionModel(hand)


def test_sdf_query_joint_and_point_gradients_match_finite_difference(geometry):
    hand,collision=geometry;model=collision.model.double()
    # A local differentiable FK fixture isolates the query wrapper with high
    # precision; native FK is checked separately below.
    class Hand:
        def frames(self,q):
            p=q[:,:3,None].transpose(1,2).expand(-1,21,-1)
            r=torch.eye(3,dtype=q.dtype).expand(len(q),21,3,3)
            return p,r
    q=torch.zeros(1,20,dtype=torch.float64,requires_grad=True)
    point=torch.tensor([[[.04,.03,.02]]],dtype=torch.float64,requires_grad=True)
    link=torch.tensor([3]);value=model.query(Hand(),q,point,link).sum()
    qgrad,pgrad=torch.autograd.grad(value,(q,point))
    eps=1.e-6
    for tensor,grad,index in ((q,qgrad,(0,0)),(point,pgrad,(0,0,0))):
        plus=tensor.detach().clone();minus=plus.clone();plus[index]+=eps;minus[index]-=eps
        args=(plus,point.detach()) if tensor is q else (q.detach(),plus)
        a=model.query(Hand(),*args,link).sum()
        args=(minus,point.detach()) if tensor is q else (q.detach(),minus)
        b=model.query(Hand(),*args,link).sum()
        assert float(grad[index])==pytest.approx(float(((a-b)/(2*eps)).detach()),rel=2.e-4,abs=1.e-7)
    model.float()
    q=((hand.low+hand.high)/2)[None].requires_grad_();index=4;eps=.001
    def native(q):return hand.frames(q)[0][0,8,0]
    g=torch.autograd.grad(native(q),q)[0][0,index]
    plus=q.detach().clone();minus=plus.clone();plus[0,index]+=eps;minus[0,index]-=eps
    assert float(g)==pytest.approx(float(((native(plus)-native(minus))/(2*eps)).detach()),abs=1.e-5)


def test_online_sdf_updates_freeze_and_replay_restore(geometry,tmp_path):
    hand,collision=geometry;learner=SDFLearner(collision,capacity=256,batch_size=32)
    learner.collect();before=[p.detach().clone() for p in collision.model.parameters()]
    learner.update();assert any(not torch.equal(a,b) for a,b in zip(before,collision.model.parameters()))
    q=((hand.low+hand.high)/2)[None].requires_grad_();rows=[sensor_row(20.)]
    before=[p.detach().clone() for p in collision.model.parameters()]
    with frozen(collision.model):
        a,b=collision.penalties(q,rows,hand.frames(q));(a+b).sum().backward()
    assert q.grad is not None and q.grad.isfinite().all()
    assert all(torch.equal(a,b) for a,b in zip(before,collision.model.parameters()))
    path=tmp_path/'state.pth';save_checkpoint(path,dict(model=collision.model.state_dict(),learner=learner.state_dict(),rng=torch.get_rng_state()))
    state=torch.load(path,weights_only=True);other=SDFLearner(collision,capacity=256,batch_size=32);other.load_state_dict(state['learner'])
    torch.set_rng_state(state['rng']);first=learner.update();weights=[p.detach().clone() for p in collision.model.parameters()]
    collision.model.load_state_dict(state['model']);torch.set_rng_state(state['rng']);second=other.update()
    assert first==second and all(torch.equal(a,b) for a,b in zip(weights,collision.model.parameters()))


def test_contact_keep_slides_and_stops_on_unknown_release_or_gap(geometry):
    hand,collision=geometry;q=((hand.low+hand.high)/2)[None].requires_grad_();frames=hand.frames(q)
    owner=8
    def contact(point,sid):return dict(region='index_distal',position_m=point,normal=[1.,0,0],force=1.,sids=[sid],observed_points_m=[point])
    a=sensor_row(0.);b=sensor_row(20.)
    a['contacts']['object_contacts']=[contact([.03,.02,0.],1)]
    b['contacts']['object_contacts']=[contact([.04,.02,0.],2)]
    keep,count=collision.retention(q,[b],[a],frames)
    assert count==[1] and keep.isfinite().all()
    for kind in ('release','unknown','gap'):
        row=copy.deepcopy(b)
        if kind=='release':row['contacts']['object_contacts']=[]
        if kind=='unknown':row['contacts']['tactile_ready']=False
        if kind=='gap':row['timestamp_ms']=300.
        keep,count=collision.retention(q,[row],[a],frames)
        assert count==[0] and keep.item()==0


def test_tactile_penetration_queries_all_links_and_zero_contact(geometry):
    hand,collision=geometry;q=((hand.low+hand.high)/2)[None]
    row=sensor_row(20.);row['tactile_state'][0]=1;row['tactile_positions_m'][0]=[.03,.02,0.]
    x,links=collision.oracle.tactile_queries(hand.frames(q),[row])[0]
    assert x.shape==(1,21,3) and links.tolist()==[list(range(21))]
    row['tactile_valid'][0]=False
    assert not collision.oracle.tactile_queries(hand.frames(q),[row])[0][0].numel()


def test_training_normalization_excludes_validation():
    training=sensor_row(0);validation=sensor_row(0)
    validation['emf_pose'][0][0]=1000
    mean,scale=position_statistics([({},[training],{})]);assert np.all(mean==0) and np.all(scale==.001)


def test_preprocessing_has_no_tube_dependency_and_old_models_are_rejected():
    import inspect
    import moxian_web.cart_data as data
    from moxian_web.cart_training import validate_model_geometry
    assert 'TubeBuilder' not in inspect.getsource(data._preprocess)
    with pytest.raises(ValueError):validate_model_geometry(dict(protocol='cart_contact_tube_v4',architecture='cart_tcn64_fsq8_v1'))


def test_policy_contact_position_normal_and_keep_have_gradients(geometry):
    hand,collision=geometry
    q=((hand.low+hand.high)/2)[None];p,r,j,*_=hand.frames(q)
    mesh=hand.surface.contact_surface(8);local=hand.tensor(mesh['triangles'][0]).mean(0);normal=hand.tensor(mesh['normals'][0])
    point=(r[0,8]@local+p[0,8]).tolist();normal=(r[0,8]@normal).tolist()
    a=sensor_row(0.);b=sensor_row(20.);a['joints_m']=b['joints_m']=j[0].tolist()
    for row in (a,b):
        row['contacts']['object_contacts']=[dict(region='index_distal',position_m=point,normal=normal,force=1.,observed_points_m=[point])]
        row['tactile_positions_m'][0]=point;row['tactile_state'][0]=1
    moved=q.clone();moved[0,6]+=.05;moved.requires_grad_()
    objective=CartObjective(hand,policy_settings(),collision)
    with frozen(collision.model):
        loss,parts,quality=objective.batch(moved,[b],q,previous_rows=[a])
        assert parts['contact']>0 and parts['contact_keep'].isfinite() and quality[0]['contact_keep_count']==1
        loss.backward()
    assert moved.grad.isfinite().all() and moved.grad.abs().sum()>0


def test_recorder_preserves_emf_independently_of_mutable_source(tmp_path):
    import queue
    import threading
    from moxian_web.geort_service import GeoRTService
    topology=dict(surface_backend='qnbot',coordinate_frame='wrist_local',hand_side='right',geometry_version='g',model_hash='m',
        joint_chains=[[0,*range(1+4*f,5+4*f)] for f in range(5)],palm_view=dict(normal=[1.,0.,0.],forward=[0.,0.,1.]),binding_version='b')
    frame=dict(ready=True,calibrated=True,timestamp_ms=100.,geometry_version='g',hand_side='right',model_hash='m',
        source_joint_positions_rad=[0.]*29,calibration_fingerprint='c',cart_label=dict(protocol='cart_diff_v5'),
        emf_observation=dict(positions_m=[[1.,2.,3.]]*5),source_contact=None)
    service=GeoRTService.__new__(GeoRTService);service.lock=threading.RLock();service.latest=None;service.latest_at=0
    service.recording=dict(invalid_frames=0)
    from retarget.baseline.geort.adapter import SourceAdapter
    source=SourceAdapter(topology).identity('serial',frame)
    service.capture=dict(source=source,last_stamp=-float('inf'),meta=dict(capture_mode='cart_v5'),binding_version='b',queue=queue.Queue())
    service.observe(frame,topology,np.zeros((21,3)),'serial')
    frame['emf_observation']['positions_m'][0][0]=99.
    assert service.capture['queue'].get_nowait()['emf_observation']['positions_m'][0][0]==1.
