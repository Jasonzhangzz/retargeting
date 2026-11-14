"""Native trajectory objective and small-Delta contract, without live devices."""
import copy,math
import pytest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from moxian_web.qnbot_surface import QnbotSurface,FINGERS
from moxian_web.qnbot_calibration import ReplayBuffer,TrajectoryObjective,DeltaPolicy,policy_artifact
from moxian_web.qnbot_calibration import spatial_ema,apply_delta,configure_dimensions,contact_band_residual,nearest_region_anchors
from moxian_web.qnbot_worker import StreamingModel
from moxian_web.contact import default_mapping


def fixture(side='right'):
    torch.set_num_threads(1);m=QnbotSurface(side);mapping=dict(assignments=default_mapping(side),confirmed=[]);m.bind_mapping(mapping)
    q=m.q0[None].repeat(8,1);q[:,7]=torch.arange(8)*.02+.1
    with torch.no_grad():p,r=m.sensor_prediction(q)
    rows=[]
    for i in range(8):
        fingers={f:dict(position=p[i,k].tolist(),quaternion_xyzw=Rotation.from_matrix(r[i,k].numpy()).as_quat().tolist(),valid=True) for k,f in enumerate(FINGERS)}
        rows.append(dict(stage='contact_pair_index',partition='train',capture_protocol='contact_motion_v4',timestamp_ms=i*20.,window_id='train:0',cycle=0,
            contact_confirmed=i==4,contact_event_id='event',mapping=mapping,fingers=fingers,skeleton=dict(joint_names=m.joint_names,joint_positions_rad=q[i].tolist())))
    b=ReplayBuffer(m,rows);b.seed_stats(m)
    templates=dict(actions=[dict(stage='contact_pair_index',edges=[dict(regions=['thumb_distal','index_distal'],required=True,anchors=[m.patch_anchors['thumb_distal'],m.patch_anchors['index_distal']])])])
    return m,b,TrajectoryObjective(m,b,templates)


def test_zero_delta_equals_ik_and_labels_are_not_features():
    m,b,o=fixture();policy=DeltaPolicy();torch.testing.assert_close(b.corrected(m,policy),b.q,atol=0,rtol=0)
    rows=copy.deepcopy(b.rows)
    for r in rows:r.update(stage='contact_pair_middle',cycle=99,partition='validation',expected_contact_regions={'secret':'not_an_input'},contact_confirmed=False)
    other=ReplayBuffer(m,rows)
    torch.testing.assert_close(b.features(m),other.features(m))
    assert b.features(m).shape[1]==146


def test_outside_native_range_seed_has_no_unlearned_delta_jump():
    seed=torch.tensor([[1.2,-.2,.5]]);limits=torch.tensor([[0.,1.],[0.,1.],[0.,1.]])
    torch.testing.assert_close(apply_delta(seed,torch.zeros_like(seed),limits),seed,atol=0,rtol=0)
    torch.testing.assert_close(apply_delta(seed,torch.tensor([[.3,-.3,.8]]),limits),torch.tensor([[1.2,-.2,1.]]))
    torch.testing.assert_close(apply_delta(seed,torch.tensor([[-.1,.1,0.]]),limits),torch.tensor([[1.1,-.1,.5]]))


def test_large_delta_penalizes_amplitude_and_emf_orientation():
    m,b,o=fixture();ids=torch.arange(len(b.rows));baseline=b.q.clone();b.last_raw_delta=torch.zeros_like(baseline)
    _,base,_=o.loss(baseline,ids)
    changed=baseline.clone();changed[:,7]+=.2;b.last_raw_delta=changed-baseline
    _,terms,metrics=o.loss(changed,ids)
    assert terms['delta_pose_prior']>base['delta_pose_prior'] and terms['delta_output_prior']>base['delta_output_prior']
    assert terms['sensor_orientation']>base['sensor_orientation'] and terms['tip_emf_orientation']>base['tip_emf_orientation']
    assert metrics['delta_max_deg']>10 and metrics['emf_orientation_error_deg']>0
    # Priors are soft: larger changes remain differentiable rather than rejected.
    changed=changed.requires_grad_(True);o.loss(changed,ids)[0].backward();assert torch.isfinite(changed.grad).all()


def test_tensor_loss_compilation_preserves_terms_and_joint_gradients():
    m,b,o=fixture();ids=torch.arange(len(b.rows));q=b.q.clone().requires_grad_(True)
    def fn(q,ids):
        loss,terms,_=o.loss(q,ids,measure=False)
        return loss,terms
    expected,terms=fn(q,ids);gradient=torch.autograd.grad(expected,q)[0]
    actual,compiled_terms=torch.compile(fn,backend='eager')(q,ids)
    torch.testing.assert_close(actual,expected)
    assert set(compiled_terms)==set(terms)
    torch.testing.assert_close(torch.autograd.grad(actual,q)[0],gradient)


def test_non_target_observations_have_zero_gradient_and_trajectory_is_not_event_only():
    m,b,o=fixture();b.rawp.requires_grad_(True);b.rawr.requires_grad_(True);q=b.q.clone();q[2,7]+=.04
    loss,terms,_=o.loss(q,torch.tensor([2]));loss.backward()
    assert torch.count_nonzero(b.rawp.grad[:,2:])==0 and torch.count_nonzero(b.rawr.grad[:,2:])==0
    assert terms['contact_position']==0 and terms['trajectory_position']>0 and terms['trajectory_normal']>0
    assert m.geometry['residual'].grad.norm()>0


def test_same_causal_delta_in_buffer_and_streaming_with_gap_reset():
    m,b,o=fixture();policy=DeltaPolicy()
    with torch.no_grad():policy.net[-1].weight.normal_(0,.0005)
    artifact=policy_artifact(policy,b,m);artifact['geometry_hash']=m.version();candidate=dict(m.artifact(),delta_policy=artifact)
    expected=b.corrected(m,policy).detach();stream=StreamingModel(m);stream.load(candidate)
    actual=[]
    for row in b.rows:actual.append(stream.infer(row)[3][0])
    torch.testing.assert_close(torch.stack(actual),expected,atol=2e-6,rtol=2e-5)
    frame=copy.deepcopy(b.rows[-1]);frame['timestamp_ms']+=1000
    _,_,_,q,_,_=stream.infer(frame)
    assert torch.isfinite(q).all() and stream.previous[0]==frame['timestamp_ms']


def test_invalid_policy_cannot_partially_replace_current_geometry():
    m,b,o=fixture();stream=StreamingModel(m);original=m.version();bad=m.artifact()
    bad['geometry']['residual'][0][0]=.01
    bad['delta_policy']=dict(geometry_hash='wrong')
    with pytest.raises(ValueError,match='指纹'):stream.load(bad)
    assert m.version()==original and stream.policy is None


@pytest.mark.parametrize("side",["left","right"])
def test_dimensions_preserve_finger_proportions_and_never_write_vertex_residual(side):
    m,b,o=fixture(side);parameters=configure_dimensions(m);opt=torch.optim.Adam(parameters,lr=.001)
    frozen={k:p.detach().clone() for k,p in m.geometry.items() if not p.requires_grad}
    for _ in range(3):
        before={k:p.detach().clone() for k,p in m.geometry.items()}
        with torch.no_grad():rest=m.rest()[0].clone()
        opt.zero_grad();v=m(b.q)[0];(v[:,::31].square().sum()+o.priors()[0]).backward();opt.step()
        spatial_ema(m,before,rest,torch.tensor([1.,1.,0.,0.,0.]))
    for k in ('length','width','depth'):
        for f in range(5):
            x=m.geometry[k][m.dimension_movable&(m.dimension_owners==f)]
            torch.testing.assert_close(x,x[0].expand_as(x),atol=0,rtol=0)
        assert torch.count_nonzero(m.geometry[k][~m.dimension_movable])==0
    for k,value in frozen.items():torch.testing.assert_close(m.geometry[k],value,atol=0,rtol=0)


def test_contact_band_allows_only_small_normal_compression():
    a=torch.zeros(5,3);b=torch.tensor([[0,0,.001],[0,0,-.001],[0,0,-.002],[0,0,-.003],[.001,0,-.001]])
    na=torch.tensor([0.,0.,1.]).expand_as(a);nb=-na;r=contact_band_residual(a,b,na,nb)
    torch.testing.assert_close(r.norm(dim=-1),torch.tensor([.001,0.,0.,.001,.001]),atol=1e-8,rtol=1e-5)


def test_collision_does_not_repel_nearby_nonpenetrating_surfaces():
    from moxian_web.qnbot_collision import QnbotCollision
    vertices=torch.tensor([[0.,0.,0.],[0,0,.001]])
    witness=dict(source=torch.tensor([[0,0,0]]),source_bc=torch.tensor([[1.,0,0]]),target=torch.tensor([[1,1,1]]),target_bc=torch.tensor([[1.,0,0]]),sign=torch.tensor([1.]))
    assert QnbotCollision.residual(vertices,witness).item()==0
    witness['sign']=torch.tensor([-1.])
    assert QnbotCollision.residual(vertices,witness,tolerance_m=.002).item()==0
    assert QnbotCollision.residual(vertices,witness,tolerance_m=0.).item()>0


def test_fold_template_does_not_pick_a_shared_joint_boundary_vertex():
    m,b,o=fixture()
    with torch.no_grad():vertices=m(m.q0[None])[0][0].numpy()
    gap,anchors=nearest_region_anchors(vertices,m.faces.numpy(),m.regions,'index_distal','index_proximal')
    assert gap>1e-6
    assert not set(m.faces[anchors[0]['face']].tolist())&set(m.faces[anchors[1]['face']].tolist())


def test_spatial_ema_preserves_critical_updates_and_damps_actual_coupled_surface():
    m,b,o=fixture();before={k:p.detach().clone() for k,p in m.geometry.items()}
    with torch.no_grad():
        old=m.rest()[0].clone();m.geometry['root'][1,0]+=.2;m.geometry['residual'][:,0]+=.001
        proposed=m.rest()[0].clone();active=torch.tensor([1.,1.,0.,0.,0.]);critical=spatial_ema(m,before,old,active)
        after=m.rest()[0]
    torch.testing.assert_close(after[critical]-old[critical],proposed[critical]-old[critical],atol=1e-7,rtol=1e-5)
    torch.testing.assert_close(after[~critical]-old[~critical],.05*(proposed[~critical]-old[~critical]),atol=1e-7,rtol=1e-5)


@torch.no_grad()
def test_cuda_graph_render_is_same_as_eager_and_can_update_q():
    if not torch.cuda.is_available():return
    m=QnbotSurface('right','cuda:0');m.fixed=True;stream=StreamingModel(m);stream.warmup()
    assert stream.graph is not None,stream.graph_error
    m.bind_mapping(dict(assignments=default_mapping('right'),confirmed=[]))
    # Mapping invalidates rest cache; allocator churn must not invalidate the
    # graph's retained geometry inputs.
    temporary=torch.empty((2000000,),device=m.device);temporary.fill_(999)
    for bend in (.1,.7):
        q=m.q0[None].clone();q[:,7]=bend;stream.static_q.copy_(q);stream.graph.replay()
        actual=[v.clone() for v in stream.static_outputs]
        for a,b in zip(actual,stream.render_forward(q)):torch.testing.assert_close(a,b,atol=2e-7,rtol=1e-5)


def test_collision_penalty_is_soft_and_not_diluted_by_zero_witnesses():
    from moxian_web.qnbot_calibration import collision_violation_loss
    residual=torch.tensor([0.,.5,2.],requires_grad=True)
    loss=collision_violation_loss(residual)
    torch.testing.assert_close(loss,collision_violation_loss(torch.cat((residual,torch.zeros(100)))))
    assert collision_violation_loss(torch.zeros(10))==0
    assert collision_violation_loss(torch.empty(0))==0
    assert collision_violation_loss(residual*2)>loss
    loss.backward();assert torch.isfinite(residual.grad).all() and residual.grad[0]==0 and residual.grad[2]>0
