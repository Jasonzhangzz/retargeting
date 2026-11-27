"""Optimized tactile pooling/cache preserve outputs and parameter gradients."""
import copy
import torch
from types import SimpleNamespace
from moxian_web.cart_pointnet import AnatomicalPointEncoder
from moxian_web.cart_fast import PreparedBatch
from moxian_web.cart_palm_alignment import PalmAlignedNetwork
from moxian_web.cart_single_step import predict_single_step


def reference_pool(model,points,state,valid):
    ids=model.sid.weight.expand(*state.shape[:-1],-1,-1)
    features=model.point(torch.cat((points,state[...,None],ids),-1))
    weights=valid.to(features.dtype);active=weights*state
    count=torch.einsum('bts,sg->btg',weights,model.assignment)
    contacts=torch.einsum('bts,sg->btg',active,model.assignment)
    mean=torch.einsum('btsc,bts,sg->btgc',features,weights,model.assignment)/count.clamp_min(1)[...,None]
    contact=torch.einsum('btsc,bts,sg->btgc',features,active,model.assignment)/contacts.clamp_min(1)[...,None]
    token=model.project(torch.cat((mean,contact,(contacts/count.clamp_min(1))[...,None],
        (count/model.assignment.sum(0).clamp_min(1))[...,None]),-1))+model.group.weight
    return token,count>0


def test_pool_matches_original_values_and_gradients():
    torch.set_num_threads(1);torch.manual_seed(12)
    old=AnatomicalPointEncoder((torch.arange(460)%6).tolist(),32).double();new=copy.deepcopy(old)
    a=torch.randn(2,3,460,3,dtype=torch.double,requires_grad=True);b=a.detach().clone().requires_grad_()
    state=(torch.rand(2,3,460)>.8).double();valid=torch.rand(2,3,460)>.2;valid[0,1]=False
    expected,mask=reference_pool(old,a,state,valid);actual,actual_mask=new(b,state,valid)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10);assert torch.equal(mask,actual_mask)
    probe=torch.randn_like(actual);(expected*probe).sum().backward();(actual*probe).sum().backward()
    torch.testing.assert_close(a.grad,b.grad,rtol=1e-9,atol=1e-10)
    for x,y in zip(old.parameters(),new.parameters()):torch.testing.assert_close(x.grad,y.grad,rtol=1e-9,atol=1e-10)


def example():
    torch.manual_seed(14)
    class Store:pass
    stores=[]
    for _ in range(2):
        store=Store();valid=torch.rand(32,460)>.2;valid[3]=False
        positions=torch.randn(32,460,3)*.01;positions[~valid]=float('nan')
        store.values=dict(tactile_positions=positions,tactile_state=(torch.rand(32,460)>.9).float(),
            tactile_valid=valid,future_emf=torch.randn(32,5,9)*.01,emf_valid=torch.rand(32,5)>.1)
        store.stamps=torch.arange(32,dtype=torch.double)*10;stores.append(store)
    batch=[]
    for session,start in [(0,0),(1,2),(0,1),(0,0),(1,0)]:
        indices=list(range(start,start+26));rows=[dict(timestamp_ms=i*10) for i in indices]
        batch.append(dict(store=stores[session],indices=indices,rows=rows,length=16,initial_q=torch.zeros(20)))
    return PreparedBatch(batch,'cpu')


def model():
    m=PalmAlignedNetwork(-torch.ones(20),torch.ones(20),width=32,decoder_width=32,
        tactile_encoder='pointnet',tactile_groups=(torch.arange(460)%6).tolist(),
        alignment=dict(translation_m=[0.]*3,rotation_vector_rad=[0.]*3,wrist_mode='unbounded'))
    for head in [*m.decoders,m.wrist_decoder]:torch.nn.init.normal_(head[-2].weight,std=.01)
    m.observation_chunk_size=1
    return m


def test_repeated_frames_multisession_and_detached_warmup_match():
    torch.set_num_threads(1);batch=example();old=model();new=copy.deepcopy(old)
    old.deduplicate_observations=False;new.deduplicate_observations=True
    objective=SimpleNamespace(hand=SimpleNamespace(device='cpu'))
    torch.manual_seed(90);a=predict_single_step(old,objective,batch,step_noise_rad=.1)
    torch.manual_seed(90);b=predict_single_step(new,objective,batch,step_noise_rad=.1)
    torch.testing.assert_close(a.predictions[0],b.predictions[0],rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(a.previous_command,b.previous_command,rtol=2e-5,atol=2e-6)
    assert not b.previous_command.requires_grad and b.input_state.is_leaf
    a.predictions[0].square().sum().backward();b.predictions[0].square().sum().backward()
    for (name,x),(_,y) in zip(old.named_parameters(),new.named_parameters()):
        assert (x.grad is None)==(y.grad is None),name
        if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,rtol=2e-4,atol=2e-6,msg=name)
    with torch.no_grad():
        c=predict_single_step(new,objective,batch,step_noise_rad=0.)
        assert not c.predictions[0].requires_grad


def test_cache_is_local_and_keeps_window_timestamps():
    torch.set_num_threads(1);batch=example();m=model();offsets=[0]*len(batch)
    first=batch.single_step_inputs(m,offsets)
    for shift in range(2):
        original=batch.future_inputs([shift]*len(batch))
        for key in ('future_emf','emf_valid','future_dt'):
            torch.testing.assert_close(first[shift][key],original[key])
    with torch.no_grad():m.touch.project.bias.add_(1)
    second=batch.single_step_inputs(m,offsets)
    torch.testing.assert_close(second[1]['tactile_tokens']-first[1]['tactile_tokens'],torch.ones_like(first[1]['tactile_tokens']),atol=1e-6,rtol=1e-6)
