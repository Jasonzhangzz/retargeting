"""Vectorization and accumulation must preserve the contact objective and BPTT."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from moxian_web.cart_fast import ObservationStore,PreparedBatch,prepare_windows,group_normalizers,choose_micro_batch,DEFAULT_BATCH_SIZE
from moxian_web.cart_training import rollout,policy_update,windows
from moxian_web.cart_observations import batch_future
from moxian_web.cart_network import CartNetwork
from moxian_web.cart_objective import CartHand,CartObjective,policy_settings
from moxian_web.cart_collision import CollisionModel,frozen


@pytest.fixture(autouse=True)
def one_thread():torch.set_num_threads(1)


def row_at(stamp,joints):
    return dict(timestamp_ms=float(stamp),joints_m=joints,stream_epoch=0,
        emf_pose=np.zeros((5,9)).tolist(),emf_valid=[True]*5,
        tactile_positions_m=np.zeros((460,3)).tolist(),tactile_state=[0]*460,tactile_valid=[True]*460,
        contacts=dict(tactile_ready=True,tactile_status='valid',self_pairs=[],object_contacts=[]))


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


def test_training_reads_recorded_next_ten_emf_and_tactile_frames(geometry):
    hand,_=geometry;rows=make_rows(hand)
    for i,row in enumerate(rows):
        row['emf_pose'][0][0]=i+.25
        row['tactile_positions_m'][0]=[i/100.,i/200.,i/300.]
        row['tactile_state'][0]=i%2;row['tactile_valid'][0]=i%3!=0
    sessions=[(dict(id='measured'),rows,{})];selected=windows(sessions,length=2)
    prepare_windows(sessions,selected,'cpu');window=selected[-1]
    offset=len(window['rows'])-window['length']-10
    current=window['indices'][offset]
    future=PreparedBatch([window],'cpu').future_inputs([offset])
    ids=list(range(current+1,current+11))
    torch.testing.assert_close(future['future_emf'][0,:,0,0],torch.tensor([i+.25 for i in ids]))
    torch.testing.assert_close(future['tactile_positions'][0,:,0],torch.tensor([rows[i]['tactile_positions_m'][0] for i in ids]))
    assert future['tactile_state'][0,:,0].tolist()==[i%2 for i in ids]
    assert future['tactile_valid'][0,:,0].tolist()==[i%3!=0 for i in ids]
    assert window['rows'][offset+1]['timestamp_ms']==rows[current+1]['timestamp_ms']
    # The following rollout step shifts all three modalities by exactly one.
    next_inputs=PreparedBatch([window],'cpu').future_inputs([offset+1])
    torch.testing.assert_close(next_inputs['future_emf'][0,:,0,0],torch.tensor([i+.25 for i in range(current+2,current+12)]))
    assert all(w['indices'][len(w['rows'])-w['length']-10]+w['length']-1+10<len(rows) for w in selected)


def test_batched_cost_matches_reference_loss_and_joint_gradient(geometry):
    hand,collision=geometry;rows=make_rows(hand);chosen=[rows[i] for i in (4,5,7,8)]
    # Also exercise multiple observations in one semantic group of a row.
    chosen[1]['contacts']['object_contacts']*=2
    reference=((hand.low+hand.high)/2)[None].repeat(4,1)
    q=(reference+.03).requires_grad_();obj=CartObjective(hand,policy_settings(),collision)
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


def test_logical_batch_gradient_is_invariant_to_gpu_chunk(geometry):
    hand,collision=geometry;rows=make_rows(hand);sessions=[(dict(id='one'),rows,{})]
    all_windows=windows(sessions,length=2);prepare_windows(sessions,all_windows,'cpu')
    batch=[all_windows[i] for i in (0,1,4,5)]
    torch.manual_seed(2);model=CartNetwork(hand.low,hand.high,'fsq')
    objective=CartObjective(hand,policy_settings(),collision)
    # Isolate policy accumulation: labels/model stay fixed between comparisons.
    learner=SimpleNamespace(collect=lambda:collision.collected.clear(),update=lambda:{})
    initial=copy.deepcopy(model.state_dict());gradients=[]
    for micro in (4,1,3):
        model.load_state_dict(initial);optimizer=torch.optim.SGD(model.parameters(),lr=0.)
        value,_,_,_=policy_update(model,objective,learner,optimizer,batch,micro)
        gradients.append((value.clone(),[p.grad.clone() for p in model.parameters()]))
    for value,gradient in gradients[1:]:
        torch.testing.assert_close(value,gradients[0][0],rtol=3.e-6,atol=.001)
        for a,b in zip(gradient,gradients[0][1]):torch.testing.assert_close(a,b,rtol=.001,atol=2.e-5)


def test_reference_batch_default_and_explicit_memory_chunk():
    assert DEFAULT_BATCH_SIZE==4096 and choose_micro_batch('attention','cpu',23)==23
    with pytest.raises(ValueError):choose_micro_batch('attention','cpu',-1)
