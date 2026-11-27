from types import SimpleNamespace
import torch
from moxian_web.cart_pointnet import AnatomicalPointEncoder
from moxian_web.cart_single_step import predict_single_step,SingleStepReachObjective
from moxian_web.cart_fast import PreparedBatch


def test_point_encoder_masks_and_all_groups_have_gradients():
    torch.manual_seed(4)
    groups=(torch.arange(460)%6).tolist();m=AnatomicalPointEncoder(groups,32)
    points=torch.randn(2,10,460,3,requires_grad=True)
    valid=torch.ones(2,10,460,dtype=torch.bool);valid[...,::7]=False
    state=(torch.arange(460)%3==0).float().expand(2,10,-1)
    out,mask=m(points,state,valid)
    changed=points.detach().clone();changed[~valid]=10000
    torch.testing.assert_close(out,m(changed,state,valid)[0])
    out.square().mean().backward()
    assert points.grad[~valid].eq(0).all()
    for g in range(6):assert points.grad[:,:,torch.tensor(groups)==g].abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    empty,mask=m(points.detach(),state,torch.zeros_like(valid))
    assert torch.isfinite(empty).all() and not mask.any()


def test_single_step_uses_previous_policy_state_but_no_warmup_gradient():
    class Store:
        values={'future_emf':torch.arange(30.)[:,None]}
        stamps=torch.arange(30,dtype=torch.double)*10
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.delta=torch.nn.Parameter(torch.full((26,),.01));self.seen=[]
        def initial_state(self,q):return q
        def forward(self,q,future_emf,**kw):
            self.seen.append((torch.is_grad_enabled(),q,future_emf.clone()))
            return self.delta[None]+.1*q
        def apply_delta(self,q,d):return q+d
    m=Policy();rows=[dict(timestamp_ms=i*10) for i in range(30)]
    seed=torch.zeros(26,requires_grad=True)
    b=PreparedBatch([dict(store=Store(),indices=list(range(26)),rows=rows[:26],length=16,initial_q=seed)],'cpu')
    tr=predict_single_step(m,SimpleNamespace(hand=SimpleNamespace(device='cpu')),b)
    assert len(tr.predictions)==1 and not tr.previous_command.requires_grad
    assert [v[0] for v in m.seen]==[False,True]
    torch.testing.assert_close(m.seen[1][1],tr.previous_command)
    torch.testing.assert_close(m.seen[1][2][0,:,0],torch.arange(2.,12.))
    tr.predictions[0].sum().backward()
    torch.testing.assert_close(m.delta.grad,torch.ones(26))
    assert seed.grad is None
    assert tr.pending[0][1][0] is rows[2]


def test_single_step_preserves_native_smooth_edge_including_wrist():
    from moxian_web.cart_training import RolloutTrajectory
    from moxian_web.cart_reach_loss import command_native_smooth_parts
    class Objective(SingleStepReachObjective):
        def distances(self,q,rows):
            return q[:,0:1].square().expand(-1,16),torch.ones(len(q),16,dtype=torch.bool),torch.ones(len(q),16)
    command=torch.zeros(1,26,requires_grad=True)
    previous=torch.zeros(1,26);previous[:,20]=.001
    tr=RolloutTrajectory([(command,[{}],previous,None,[{}])],[{}],commands=[command]);tr.previous_command=previous
    loss,parts=Objective(None).trajectory_loss(tr)
    expected=sum(v.mean()*2 for v in command_native_smooth_parts(torch.stack((previous,command))).values())
    torch.testing.assert_close(loss,expected)
    assert parts['smooth_translation']>0
    loss.backward();assert command.grad[:,20].abs().sum()>0


def test_pointnet_policy_full_backward_and_reload():
    from moxian_web.cart_palm_alignment import PalmAlignedNetwork
    torch.set_num_threads(1);torch.manual_seed(7)
    args=dict(alignment=dict(translation_m=[0.]*3,rotation_vector_rad=[0.]*3,wrist_mode='unbounded'),
              width=128,decoder_width=128,tactile_encoder='pointnet',tactile_groups=(torch.arange(460)%6).tolist())
    m=PalmAlignedNetwork(-torch.ones(20),torch.ones(20),**args)
    for head in [*m.decoders,m.wrist_decoder]:torch.nn.init.normal_(head[-2].weight,std=.001)
    q=m.initial_state(torch.zeros(2,20));emf=torch.randn(2,10,5,9)*.01
    valid=torch.ones(2,10,5,dtype=torch.bool);dt=torch.arange(1,11)[None].expand(2,-1)/120
    points=torch.randn(2,10,460,3)*.01;state=torch.zeros(2,10,460);state[...,::10]=1
    tv=torch.ones_like(state,dtype=torch.bool)
    # Completely missing tactile frames must still use the finite null token.
    tv[1]=False;points[1]=float('nan')
    delta=m(q,emf,valid,dt,points,state,tv);m.apply_delta(q,delta).square().sum().backward()
    for module in [m.touch,m.emf,m.current,m.blocks[0],*m.decoders,m.wrist_decoder]:
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads)>0
    restored=PalmAlignedNetwork(-torch.ones(20),torch.ones(20),**args)
    restored.load_state_dict(m.state_dict(),strict=True)
    torch.testing.assert_close(restored(q,emf,valid,dt,points,state,tv),delta)


def test_detached_late_states_use_correct_observation_and_only_one_step_gradient():
    class Store:
        values={'future_emf':torch.arange(30.)[:,None]}
        stamps=torch.arange(30,dtype=torch.double)*10
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.delta=torch.nn.Parameter(torch.full((26,),.01));self.seen=[]
        def initial_state(self,q):return q
        def forward(self,q,future_emf,**kw):
            self.seen.append((torch.is_grad_enabled(),future_emf.clone()))
            return self.delta[None]
        def apply_delta(self,q,d):return q+d
    for depth in [0,7,15]:
        m=Policy();rows=[dict(timestamp_ms=i*10) for i in range(26)]
        b=PreparedBatch([dict(store=Store(),indices=list(range(26)),rows=rows,length=16,initial_q=torch.zeros(26))],'cpu')
        tr=predict_single_step(m,SimpleNamespace(hand=SimpleNamespace(device='cpu')),b,warmup_steps=depth)
        assert [v[0] for v in m.seen]==[False]*depth+[True]
        assert tr.pending[0][1][0] is rows[depth+1]
        assert tr.observation_offsets==[depth]
        torch.testing.assert_close(m.seen[-1][1][0,:,0],torch.arange(depth+1.,depth+11.))
        torch.testing.assert_close(tr.predictions[0],torch.full((1,26),.01*(depth+1)))
        tr.predictions[0].sum().backward()
        torch.testing.assert_close(m.delta.grad,torch.ones(26))
