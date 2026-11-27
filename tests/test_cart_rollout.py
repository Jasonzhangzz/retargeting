from types import SimpleNamespace
import numpy as np
import torch

def test_observation_dedup_preserves_rollout_and_parameter_gradients():
    import copy
    from moxian_web.cart_training import predict_trajectory
    from moxian_web.cart_fast import PreparedBatch
    class Store:
        values={'future_emf':torch.arange(30,dtype=torch.double)[:,None]*.001}
        stamps=torch.arange(30,dtype=torch.double)*10
    class Policy(torch.nn.Module):
        architecture='attention'
        def __init__(self):
            super().__init__();self.encoder=torch.nn.Linear(1,20).double();self.state=torch.nn.Linear(20,20).double();self.count=0
        def initial_state(self,q):return q
        def encode_prepared_observations(self,future_emf,future_dt):
            self.count+=len(future_emf)
            return self.encoder(future_emf.mean(1)),torch.zeros(len(future_emf),1,dtype=torch.bool)
        def decode_observation_context(self,q,encoded):return .01*(encoded[0]+self.state(q))
        def apply_delta(self,q,d):return q+d
    torch.manual_seed(2);a=Policy();b=copy.deepcopy(a);b.deduplicate_observations=True;b.observation_chunk_size=2
    store=Store();rows=[{'timestamp_ms':10*i} for i in range(30)]
    batch=PreparedBatch([dict(store=store,indices=list(range(i,i+18)),rows=rows[i:i+18],length=4,initial_q=torch.zeros(20,dtype=torch.double)) for i in (0,1)],'cpu')
    objective=SimpleNamespace(hand=SimpleNamespace(device='cpu'))
    ta=predict_trajectory(a,objective,batch);tb=predict_trajectory(b,objective,batch)
    xa=torch.stack(ta.predictions);xb=torch.stack(tb.predictions)
    torch.testing.assert_close(xa,xb,atol=1.e-12,rtol=1.e-12)
    xa.square().sum().backward();xb.square().sum().backward()
    for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa.grad,pb.grad,atol=1.e-12,rtol=1.e-12)
    assert a.count==8 and b.count==5


def test_16_step_noise_is_added_after_every_delta_and_future_emf_advances():
    from moxian_web.cart_training import predict_trajectory
    from moxian_web.cart_fast import PreparedBatch
    class Store:
        values={'future_emf':torch.arange(30,dtype=torch.double)[:,None]}
        stamps=torch.arange(30,dtype=torch.double)*10
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.delta=torch.nn.Parameter(torch.full((20,),.01,dtype=torch.double))
            self.lower=torch.full((20,),-10.);self.upper=-self.lower;self.inputs=[]
        def initial_state(self,q):return q
        def forward(self,q,future_emf,**kwargs):
            self.inputs.append((q,future_emf.clone()))
            return torch.cat((self.delta.expand(len(q),-1)+.02*q[:,:20],q[:,20:]*0),-1)
        def apply_delta(self,q,d):return q+d
    model=Policy();rows=[{'timestamp_ms':10*i} for i in range(30)]
    window=dict(store=Store(),indices=list(range(26)),rows=rows[:26],length=16,initial_q=torch.zeros(26,dtype=torch.double))
    batch=PreparedBatch([window],'cpu');objective=SimpleNamespace(hand=SimpleNamespace(device='cpu'),command_smoothing=True)
    torch.manual_seed(37)
    expected_noise=[torch.empty((1,20),dtype=torch.double).uniform_(-.1,.1) for _ in range(16)]
    torch.manual_seed(37)
    trajectory=predict_trajectory(model,objective,batch,step_noise_rad=.1)
    for t,(state,emf) in enumerate(model.inputs):
        torch.testing.assert_close(emf[0,:,0],torch.arange(t+1,t+11,dtype=torch.double))
        if t:assert state is trajectory.predictions[t-1]
        torch.testing.assert_close(trajectory.commands[t][:,:20],state[:,:20]*1.02+model.delta)
        expected=state[:,:20]*1.02+model.delta+expected_noise[t]
        torch.testing.assert_close(trajectory.predictions[t][:,:20],expected)
        torch.testing.assert_close(trajectory.predictions[t][:,20:],state[:,20:])
        velocity,acceleration,valid=trajectory.controls[t]
        torch.testing.assert_close(velocity[:,:20],(model.delta+.02*state[:,:20])/float(torch.tensor(.01)),rtol=1.e-6,atol=1.e-6)
        assert bool(valid.all())==(t>0)
        if t==0:assert acceleration.eq(0).all()
    # Noise does not detach the recurrence: all 15 subsequent transitions remain.
    grad=torch.autograd.grad(trajectory.predictions[-1].sum(),trajectory.predictions[0],retain_graph=True)[0]
    torch.testing.assert_close(grad[:,:20],torch.full((1,20),1.02**15,dtype=torch.double))
    command_grad=torch.autograd.grad(trajectory.commands[-1].sum(),trajectory.commands[0],retain_graph=True)[0]
    torch.testing.assert_close(command_grad[:,:20],torch.full((1,20),1.02**15,dtype=torch.double))
    trajectory.predictions[-1].sum().backward()
    assert torch.isfinite(model.delta.grad).all() and (model.delta.grad>16).all()
    # Default (validation) path consumes no noise RNG and is deterministic.
    before=torch.get_rng_state();clean=predict_trajectory(model,objective,batch)
    assert torch.equal(before,torch.get_rng_state())
    torch.testing.assert_close(clean.predictions[-1][:,:20],model.delta[None]*((1.02**16-1)/.02))


def test_process_noise_respects_joint_limits_and_never_perturbs_wrist():
    from moxian_web.cart_training import perturb_rollout_state
    model=SimpleNamespace(lower=torch.full((20,),-1.),upper=torch.ones(20))
    q=torch.cat((torch.ones(256,20),torch.randn(256,6)),1).requires_grad_()
    noisy=perturb_rollout_state(model,q,.1)
    assert (noisy[:,:20]<=1).all() and ((noisy[:,:20]-q[:,:20]).abs()<=.100001).all()
    torch.testing.assert_close(noisy[:,20:],q[:,20:])
    assert perturb_rollout_state(model,q,0) is q
