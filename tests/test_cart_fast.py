"""True parallel batches and temporal vectorization preserve contact BPTT."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from moxian_web.cart_fast import ObservationStore,PreparedBatch,prepare_windows,group_normalizers,choose_micro_batch,choose_parallel_batch,DEFAULT_BATCH_SIZE
from moxian_web.cart_training import rollout,policy_update,windows,validation_schedule,predict_trajectory
from moxian_web.cart_observations import batch_future
from moxian_web.cart_network import CartNetwork
from moxian_web.cart_objective import CartHand,CartObjective,policy_settings
from moxian_web.cart_collision import CollisionModel,SDFLearner,frozen


@pytest.fixture(autouse=True)
def one_thread():torch.set_num_threads(1)


def row_at(stamp,joints):
    return dict(timestamp_ms=float(stamp),joints_m=joints,stream_epoch=0,
        ik_valid=True,ik_q_rad=[.4]*20,
        emf_pose=np.zeros((5,9)).tolist(),emf_valid=[True]*5,
        tactile_positions_m=np.zeros((460,3)).tolist(),tactile_state=[0]*460,tactile_valid=[True]*460,
        contacts=dict(tactile_ready=True,tactile_status='valid',self_pairs=[],object_contacts=[]))


def test_dense_windows_are_distinct_and_preserve_future_and_stream_boundaries():
    from moxian_web.cart_data import sequence_indices
    rows=[dict(timestamp_ms=i*20,stream_epoch=i//60,ik_valid=True,ik_q_rad=[i]*20) for i in range(120)]
    sessions=[(dict(id='one'),rows,{})]
    default=windows(sessions,16)
    assert [w['indices'] for w in default]==sequence_indices(rows,16)
    dense=windows(sessions,16,stride=2)
    starts=[w['initial_q'][0] for w in dense]
    assert len(starts)==len(set(starts)) and len(dense)>len(default)
    assert starts==list(range(0,35,2))+list(range(60,95,2))
    for w in dense:
        assert len({rows[i]['stream_epoch'] for i in w['indices']})==1
        offset=len(w['indices'])-16-10
        current=w['indices'][offset]
        assert w['indices'][offset+1:offset+11]==list(range(current+1,current+11))
        assert w['indices'][-1]==current+16+9
    # A dropout and an invalid IK seed are still hard window boundaries.
    rows[30]['timestamp_ms']+=10000
    rows[60]['ik_valid']=False
    for w in windows(sessions,16,stride=1):
        stamps=[rows[i]['timestamp_ms'] for i in w['indices']]
        assert all(0<b-a<=250 for a,b in zip(stamps,stamps[1:]))
        assert w['initial_q'][0]!=60
    for invalid in (-1,True,1.5):
        with pytest.raises(ValueError):windows(sessions,16,stride=invalid)


@pytest.fixture(scope='module')
def geometry():
    hand=CartHand('cpu');return hand,CollisionModel(hand)


def make_rows(hand):
    q=((hand.low+hand.high)/2)[None];p,r,j,*_=hand.frames(q)
    rows=[row_at(i*20,j[0].tolist()) for i in range(32)]
    tri=torch.tensor(hand.surface.contact_surface(8)['triangles'][0],dtype=torch.float32)
    ln=torch.tensor(hand.surface.contact_surface(8)['normals'][0],dtype=torch.float32)
    point=(r[0,8]@tri.mean(0)+p[0,8]).tolist();normal=(r[0,8]@ln).tolist()
    for i,row in enumerate(rows):
        row['ik_q_rad']=q[0].tolist()
        if i%3:
            row['contacts']['object_contacts']=[dict(region='index_distal',position_m=point,normal=normal,force=1.,observed_points_m=[point])]
        if i%4==0:
            row['contacts']['self_pairs']=[dict(regions=['thumb_distal','index_distal'],points_m=[point,point])]
        row['tactile_positions_m'][0]=point;row['tactile_state'][0]=int(i%2==0)
    return rows


def test_prepared_future_matches_public_packer_and_preserves_boundaries(geometry):
    hand,_=geometry;rows=make_rows(hand);sessions=[(dict(id='one'),rows,{})]
    selected=windows(sessions,length=2);prepare_windows(sessions,selected,'cpu')
    chosen=[selected[0],selected[3],selected[-1]];offsets=[0,2,1]
    expected=batch_future(chosen,offsets,'cpu');actual=PreparedBatch(chosen,'cpu').future_inputs(offsets)
    for key in expected:torch.testing.assert_close(actual[key],expected[key])
    # Validation happens once, before the network's synchronized hot path.
    rows[10]['timestamp_ms']=rows[9]['timestamp_ms']
    with pytest.raises(ValueError):prepare_windows(sessions,windows(sessions,2)+[dict(rows=rows[:14],indices=list(range(14)),length=2,dataset='one')],'cpu')


def test_training_reads_recorded_next_ten_ik_and_tactile_frames(geometry):
    hand,_=geometry;rows=make_rows(hand)
    for i,row in enumerate(rows):
        row['ik_q_rad'][0]=i/100.+.25
        row['tactile_positions_m'][0]=[i/100.,i/200.,i/300.]
        row['tactile_state'][0]=i%2;row['tactile_valid'][0]=i%3!=0
    sessions=[(dict(id='measured'),rows,{})];selected=windows(sessions,length=2)
    prepare_windows(sessions,selected,'cpu');window=selected[-1]
    offset=len(window['rows'])-window['length']-10
    current=window['indices'][offset]
    future=PreparedBatch([window],'cpu').future_inputs([offset])
    ids=list(range(current+1,current+11))
    torch.testing.assert_close(future['future_ik'][0,:,0],torch.tensor([i/100.+.25 for i in ids]))
    torch.testing.assert_close(future['tactile_positions'][0,:,0],torch.tensor([rows[i]['tactile_positions_m'][0] for i in ids]))
    assert future['tactile_state'][0,:,0].tolist()==[i%2 for i in ids]
    assert future['tactile_valid'][0,:,0].tolist()==[i%3!=0 for i in ids]
    assert window['rows'][offset+1]['timestamp_ms']==rows[current+1]['timestamp_ms']
    # The following rollout step shifts all three modalities by exactly one.
    next_inputs=PreparedBatch([window],'cpu').future_inputs([offset+1])
    torch.testing.assert_close(next_inputs['future_ik'][0,:,0],torch.tensor([i/100.+.25 for i in range(current+2,current+12)]))
    assert all(w['indices'][len(w['rows'])-w['length']-10]+w['length']-1+10<len(rows) for w in selected)


def test_batched_cost_matches_reference_loss_and_joint_gradient(geometry):
    hand,collision=geometry;rows=make_rows(hand);chosen=[rows[i] for i in (4,5,7,8)]
    # Also exercise multiple observations in one semantic group of a row.
    chosen[1]['contacts']['object_contacts']*=2
    reference=((hand.low+hand.high)/2)[None].repeat(4,1)
    q=(reference+.03).requires_grad_();obj=CartObjective(hand,replace(policy_settings(),loss_kind='quadratic_legacy'),collision)
    with frozen(collision.model):
        obj.vectorized=False;expected,parts,old_metrics=obj.batch(q,chosen,reference,previous_rows=[rows[i-1] for i in (4,5,7,8)],audit=True)
        old_gradient=torch.autograd.grad(expected,q)[0]
        obj.vectorized=True;actual,new_parts,metrics=obj.batch(q,chosen,reference,previous_rows=[rows[i-1] for i in (4,5,7,8)],audit=True)
        gradient=torch.autograd.grad(actual,q)[0]
    torch.testing.assert_close(actual,expected,rtol=3.e-6,atol=.002)
    torch.testing.assert_close(gradient,old_gradient,rtol=2.e-4,atol=.02)
    for key in parts:torch.testing.assert_close(parts[key],new_parts[key],rtol=3.e-6,atol=.002)
    for a,b in zip(metrics,old_metrics):
        for key in ('max_gap_m','max_normal_deg','feasible','unresolved_contact_count'):
            torch.testing.assert_close(a[key],b[key])


def test_training_can_skip_quality_objects_without_changing_loss_or_gradient(geometry):
    hand,collision=geometry;rows=make_rows(hand);chosen=[rows[i] for i in (4,5,7,8)]
    previous=[rows[i-1] for i in (4,5,7,8)]
    q=(((hand.low+hand.high)/2)[None].repeat(4,1)+.03).requires_grad_()
    obj=CartObjective(hand,policy_settings(),collision)
    with frozen(collision.model):
        expected,parts,metrics=obj.batch(q,chosen,previous_rows=previous)
        a=torch.autograd.grad(expected,q)[0]
        actual,other,empty=obj.batch(q,chosen,previous_rows=previous,collect_metrics=False)
        b=torch.autograd.grad(actual,q)[0]
    assert metrics and empty==[]
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for key in parts:torch.testing.assert_close(parts[key],other[key],rtol=0,atol=0)


def test_retained_links_average_points_directions_and_terms_with_matching_gradient(geometry):
    hand,collision=geometry;q=(((hand.low+hand.high)/2)[None].repeat(4,1)+.02).requires_grad_()
    rows=make_rows(hand)[:4];previous=make_rows(hand)[:4]
    for i,(row,old) in enumerate(zip(rows,previous)):
        old['timestamp_ms']=i*20;row['timestamp_ms']=i*20+10
        for item in (row,old):item['contacts'].update(object_contacts=[],self_pairs=[],tactile_ready=True)
    def contact(region,points):return dict(region=region,position_m=points[0],observed_points_m=points)
    clouds=[[.03,.02,.04],[.04,.01,.05]]
    for item in (rows[1],previous[1]):
        item['contacts']['object_contacts']=[contact('index_distal',clouds)]
        item['contacts']['self_pairs']=[dict(regions=['thumb_distal','index_distal'])]
    for item in (rows[2],previous[2]):item['contacts']['object_contacts']=[contact('index_distal',clouds),contact('middle_distal',clouds[:1])]
    previous[0]=None
    frames=hand.frames(q);p,r=frames[:2];original=collision.distance
    def distance(local,links,**kwargs):return .2*local[...,0]+.1*local[...,1]+.03
    collision.distance=distance
    try:actual,counts=collision.retention(q,rows,previous,frames,scale=.1)
    finally:collision.distance=original
    def penalty(row,owner,world):
        local=(r[row,owner].T@(world-p[row,owner])[...,None]).squeeze(-1)
        cost=(distance(local,None)/.1).square()
        return 1-(torch.softmax(-cost,0)*torch.exp(-.5*cost)).sum()
    points=q.new_tensor(clouds)
    a=penalty(1,8,points)
    directions=[]
    for source,target in ((4,8),(8,4)):
        material=collision.oracle.surface_by_link[source]
        directions.append(penalty(1,target,(r[1,source]@material[...,None]).squeeze(-1)+p[1,source]))
    expected=torch.stack((q.sum()*0,(a+torch.stack(directions).mean())/2,
        (penalty(2,8,points)+penalty(2,12,points[:1]))/2,q.sum()*0))
    assert counts==[0,2,2,0]
    torch.testing.assert_close(actual,expected,rtol=2.e-6,atol=1.e-7)
    a=torch.autograd.grad(actual.sum(),q,retain_graph=True)[0];b=torch.autograd.grad(expected.sum(),q)[0]
    torch.testing.assert_close(a,b,rtol=2.e-5,atol=2.e-7)


def test_label_collection_is_bounded_and_never_expands_the_dense_query_set(geometry,monkeypatch):
    hand,collision=geometry;q=((hand.low+hand.high)/2)[None].repeat(128,1).requires_grad_()
    rows=make_rows(hand)
    for row in rows:
        row['contacts'].update(object_contacts=[],self_pairs=[])
        row['_cart_active_points']=torch.zeros(460,3)
    rows=(rows*4)[:128]
    def forbidden(*args,**kwargs):raise AssertionError('Dense query generation is for losses, not label sampling')
    monkeypatch.setattr(collision.oracle,'self_queries',forbidden);monkeypatch.setattr(collision.oracle,'tactile_queries',forbidden)
    monkeypatch.setattr(torch,'randperm',forbidden)
    collision.collected=[];collision.collect_queries(q,rows,hand.frames(q),sample_budget=32)
    assert len(collision.collected)==2
    for points,links in collision.collected:
        assert 0<len(points)<=32 and not points.requires_grad
        assert torch.isfinite(points).all() and links.min()>=0 and links.max()<21
    collision.collected=[]


def test_one_parallel_batch_has_one_policy_update_and_no_accumulation(geometry):
    hand,collision=geometry;rows=make_rows(hand);sessions=[(dict(id='one'),rows,{})]
    all_windows=windows(sessions,length=2);prepare_windows(sessions,all_windows,'cpu')
    batch=[all_windows[i] for i in (0,1,4,5)]
    torch.manual_seed(2);model=CartNetwork(hand.low,hand.high,'fsq')
    objective=CartObjective(hand,policy_settings(),collision)
    # Isolate policy accumulation: labels/model stay fixed between comparisons.
    updates=[]
    learner=SimpleNamespace(collect=lambda:collision.collected.clear(),update=lambda:updates.append('geometry') or {})
    optimizer=torch.optim.SGD(model.parameters(),lr=0.)
    step=optimizer.step
    optimizer.step=lambda:updates.append('policy') or step()
    value,_,_,_=policy_update(model,objective,learner,optimizer,batch,4)
    assert updates==['geometry','geometry','policy'] and torch.isfinite(value)
    assert all(p.grad is not None for p in model.parameters())
    for capacity in (1,3):
        with pytest.raises(ValueError,match='禁止跨显存分块累积梯度'):
            policy_update(model,objective,learner,optimizer,batch,capacity)
    assert updates==['geometry','geometry','policy']


def test_parallel_batch_default_and_memory_limit(monkeypatch):
    from moxian_web import cart_fast
    assert DEFAULT_BATCH_SIZE==0 and choose_micro_batch('attention','cpu',23)==23
    with pytest.raises(ValueError):choose_micro_batch('attention','cpu',-1)
    monkeypatch.setattr(cart_fast,'choose_micro_batch',lambda *args:16)
    assert choose_parallel_batch('attention','cuda:0',4096)==16
    assert choose_parallel_batch('attention','cuda:0',8)==8
    assert choose_parallel_batch('attention','cuda:0',memory_limit=4)==4
    assert choose_parallel_batch('attention','cuda:0',length=32)==8
    with pytest.raises(ValueError):choose_parallel_batch('attention','cuda:0',True)


def test_large_gpu_parallel_capacity_has_no_256_window_cap(monkeypatch):
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(120*1024**3,140*1024**3))
    capacity=choose_parallel_batch('attention','cuda:0',attention_precision='bfloat16',checkpoint_observations=True)
    assert capacity>1326
    assert choose_parallel_batch('attention','cuda:0',requested=1326,attention_precision='bfloat16',checkpoint_observations=True)==1326


def test_native_validation_is_periodic_and_uses_fixed_independent_windows():
    windows=list(range(259))
    first=validation_schedule(windows,1,200)
    assert len(first)==32 and len(set(first))==32 and first[0]==0 and first[-1]==258
    assert validation_schedule(windows,2,200)==[]
    assert validation_schedule(windows,10,200)==first
    assert validation_schedule(windows,199,199)==first
    assert validation_schedule(windows,10,200,budget=0)==windows


def test_ragged_tactile_transforms_preserve_points_and_joint_gradients(geometry):
    hand,collision=geometry;rows=make_rows(hand)[4:8]
    q=((hand.low+hand.high)/2)[None].repeat(4,1).requires_grad_()
    frames=hand.frames(q);p,r=frames[:2]
    actual=collision.oracle.tactile_queries(frames,rows);expected=[]
    for i,row in enumerate(rows):
        mask=np.asarray(row['tactile_valid'])&np.asarray(row['tactile_state'],bool)
        points=hand.tensor(np.asarray(row['tactile_positions_m'])[mask])
        expected.append((r[i,None].transpose(-1,-2)@(points[:,None]-p[i,None])[...,None]).squeeze(-1))
    assert any(len(x)==0 for x in expected) and any(len(x)>0 for x in expected)
    for (value,links),target in zip(actual,expected):
        torch.testing.assert_close(value,target)
        assert links.shape==value.shape[:-1]
    a=torch.cat([v.reshape(-1,3) for v,_ in actual]);b=torch.cat([v.reshape(-1,3) for v in expected])
    weights=torch.randn_like(a)
    ga=torch.autograd.grad((a*weights).sum(),q,retain_graph=True)[0]
    gb=torch.autograd.grad((b*weights).sum(),q)[0]
    torch.testing.assert_close(ga,gb,rtol=1.e-5,atol=1.e-6)


def test_retained_policy_graph_survives_real_geometry_updates(geometry):
    hand,collision=geometry;rows=make_rows(hand)
    selected=windows([(dict(id='one'),rows,{})],length=2)
    prepare_windows([(dict(id='one'),rows,{})],selected,'cpu')
    batch=selected[:2]
    for w in batch:w['initial_q']=((hand.low+hand.high)/2).clone()
    model=CartNetwork(hand.low,hand.high,'fsq');objective=CartObjective(hand,policy_settings(),collision)
    learner=SDFLearner(collision,capacity=512,batch_size=32)
    old_model=[p.detach().clone() for p in model.parameters()]
    calls=[]
    handle=model.register_forward_hook(lambda *args:calls.append('forward'))
    update=learner.update
    def update_geometry():
        assert all(p.grad is None for p in model.parameters())
        assert all(not x.requires_grad for x,_ in collision.collected)
        return update()
    learner.update=update_geometry
    try:
        value,_,_,_=policy_update(model,objective,learner,torch.optim.AdamW(model.parameters(),lr=1.e-4),batch,2)
    finally:handle.remove()
    assert len(calls)==2 and learner.steps==2 and torch.isfinite(value)
    assert any(not torch.equal(a,b) for a,b in zip(old_model,model.parameters()))


def test_time_batched_sensor_encoder_preserves_causal_policy_and_gradients(geometry):
    hand,collision=geometry;rows=make_rows(hand)
    selected=windows([(dict(id='one'),rows,{})],length=3)
    prepare_windows([(dict(id='one'),rows,{})],selected,'cpu')
    batch=PreparedBatch(selected[:2],'cpu')
    for w in batch:w['initial_q']=((hand.low+hand.high)/2).clone()
    torch.manual_seed(46);model=CartNetwork(hand.low,hand.high,'attention')
    model.checkpoint_observations=True;objective=CartObjective(hand,policy_settings(),collision)
    results=[]
    for batched in (False,True):
        model.zero_grad(set_to_none=True)
        trajectory=predict_trajectory(model,objective,batch,batch_observations=batched)
        predictions=torch.stack(trajectory.predictions)
        predictions.square().mean().backward()
        results.append((predictions.detach(),[p.grad.clone() for p in model.parameters()]))
    torch.testing.assert_close(results[0][0],results[1][0],rtol=1.e-5,atol=1.e-6)
    for a,b in zip(results[0][1],results[1][1]):torch.testing.assert_close(a,b,rtol=.001,atol=1.e-6)


@pytest.mark.parametrize('architecture',['attention','fsq'])
def test_time_batched_losses_preserve_values_and_bptt_gradients(geometry,architecture):
    hand,collision=geometry;rows=make_rows(hand)
    for i,row in enumerate(rows):row['timestamp_ms']=i*20.+(i%3)*2.
    sessions=[(dict(id='one'),rows,{})];selected=windows(sessions,length=4)
    prepare_windows(sessions,selected,'cpu');chosen=[selected[i] for i in (0,1,2)]
    for w in chosen:w['initial_q']=((hand.low+hand.high)/2).clone()
    normalizers=group_normalizers(chosen)
    batch=PreparedBatch(chosen,'cpu',normalizers)
    torch.manual_seed(7);model=CartNetwork(hand.low,hand.high,architecture)
    objective=CartObjective(hand,policy_settings(),collision)
    # Nonzero acceleration also checks its first-frame denominator, independently
    # of the default monitoring-only setting. Contacts vary by time and window.
    objective.settings=replace(objective.settings,acceleration=.00001)
    outputs=[]
    with frozen(collision.model):
        for batched in (False,True):
            model.zero_grad(set_to_none=True)
            loss,parts,metrics,predictions=rollout(model,objective,batch,batch_losses=batched)
            loss.backward()
            outputs.append((loss.detach(),{k:v.detach() for k,v in parts.items()},metrics,
                [p.detach() for p in predictions],[p.grad.clone() for p in model.parameters()]))
    expected,actual=outputs
    torch.testing.assert_close(actual[0],expected[0],rtol=3.e-6,atol=.002)
    for key in expected[1]:torch.testing.assert_close(actual[1][key],expected[1][key],rtol=3.e-6,atol=.002)
    for a,b in zip(actual[3],expected[3]):torch.testing.assert_close(a,b)
    for a,b in zip(actual[4],expected[4]):torch.testing.assert_close(a,b,rtol=.002,atol=.002)
    for a,b in zip(actual[2],expected[2]):
        for key in ('max_gap_m','max_normal_deg','max_penetration_m','contact_keep_loss'):
            torch.testing.assert_close(a[key],b[key],rtol=3.e-6,atol=.001 if key=='max_normal_deg' else 2.e-5)
