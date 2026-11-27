from types import SimpleNamespace
import torch
from moxian_web.cart_reach_loss import ReachObjective, command_motion_jitter


class PlaneHand:
    def frames(self,q):
        p=torch.zeros(len(q),21,3,dtype=q.dtype);p[:,:,0]=q[:,0,None]
        r=torch.eye(3,dtype=q.dtype).expand(len(q),21,3,3)
        return p,r
    def nearest(self,x,owner,palmar=False):
        hit=x.clone();hit[:,0]=0
        return hit,torch.zeros_like(x),torch.ones(len(x),dtype=torch.bool)


def fixture(parameter,offsets,active=True):
    b=len(offsets);t=3;pred=[];pending=[];controls=[]
    for k in range(t):
        q=torch.zeros(b,26,dtype=torch.float64)+parameter*0
        q[:,0]=parameter+offsets+.0001*k
        rows=[]
        for i in range(b):
            mask=torch.zeros(16,dtype=torch.bool);mask[0]=active
            rows.append(dict(timestamp_ms=k*1000/120,_card_targets=dict(anchor_position=torch.zeros(16,3,dtype=q.dtype),mask=mask,force=torch.zeros(16,dtype=q.dtype))))
        pred.append(q);pending.append((q,rows,q,None,rows));controls.append((q*0,q*0,torch.zeros(b,dtype=torch.bool)))
    return SimpleNamespace(commands=pred.copy(),predictions=pred,pending=pending,controls=controls,batch=[None]*b)


def test_exact_linear_card_reach_and_gradient():
    p=torch.tensor(.002,dtype=torch.float64,requires_grad=True);tr=fixture(p,torch.zeros(4,dtype=p.dtype))
    loss,parts=ReachObjective(PlaneHand()).trajectory_loss(tr)
    expected=1600*torch.stack([(p+.0001*k)**2 for k in range(3)]).mean()
    torch.testing.assert_close(parts['contact'],expected)
    a=torch.autograd.grad(loss,p)[0]
    torch.testing.assert_close(a,3200*(p+.0001))


def test_equal_sequence_accumulation_matches_full_batch():
    p=torch.tensor(.002,dtype=torch.float64,requires_grad=True);offsets=torch.linspace(0,.004,8,dtype=p.dtype);obj=ReachObjective(PlaneHand())
    full=obj.trajectory_loss(fixture(p,offsets))[0];full.backward();reference=p.grad.clone();p.grad=None
    for block in offsets.split(2):(obj.trajectory_loss(fixture(p,block))[0]/4).backward()
    torch.testing.assert_close(p.grad,reference)


def test_released_contacts_have_no_reach_or_jitter_gradient():
    p=torch.tensor(.002,dtype=torch.float64,requires_grad=True)
    loss,parts=ReachObjective(PlaneHand()).trajectory_loss(fixture(p,torch.zeros(2,dtype=p.dtype),False))
    loss.backward();assert parts['contact']==0 and p.grad==0 and parts['contact_jitter']==0


def test_contact_jumps_have_no_extra_loss_or_gradient():
    p=torch.tensor(.002,dtype=torch.float64,requires_grad=True)
    tr=fixture(p,torch.zeros(2,dtype=p.dtype))
    # Abrupt contact-state change with unchanged clean pose commands isolates
    # contact distance from the removed temporal gap penalty.
    tr.predictions[2]=tr.predictions[2]+torch.nn.functional.pad((p*3).expand(2,1),(0,25))
    obj=ReachObjective(PlaneHand());loss,parts=obj.trajectory_loss(tr)
    expected=parts['contact']+parts['smooth_joint']+parts['smooth_translation']+parts['smooth_rotation']
    torch.testing.assert_close(loss,expected)
    assert parts['contact_jitter']==0 and not parts['contact_jitter'].requires_grad
    a=torch.autograd.grad(loss,p,retain_graph=True)[0]
    b=torch.autograd.grad(expected,p)[0]
    torch.testing.assert_close(a,b)


def test_native_smooth_matches_raw_card_and_ignores_dt():
    from retarget.baseline.card.math import CardSettings, axis_angle_near, exp_so3
    from moxian_web.cart_reach_loss import command_native_smooth_parts
    torch.manual_seed(7)
    q=torch.randn(5,2,26,dtype=torch.double)*.02
    stamps=torch.arange(5,dtype=torch.double)[:,None].expand(-1,2)*8
    cfg=CardSettings();raw=torch.zeros(2,dtype=q.dtype)
    for t in range(1,5):
        pr=axis_angle_near(exp_so3(q[t-1,:,23:]))
        raw+=cfg.wtheta**2*(q[t,:,:20]-q[t-1,:,:20]).square().sum(-1)
        raw+=cfg.wt**2*((q[t,:,20:23]-q[t-1,:,20:23])*1000).square().sum(-1)
        raw+=cfg.womega**2*(axis_angle_near(exp_so3(q[t,:,23:]),pr)-pr).square().sum(-1)
    torch.testing.assert_close(command_motion_jitter(q,stamps),raw/5/1e6)
    torch.testing.assert_close(command_motion_jitter(q,stamps*3+1e12),raw/5/1e6)
    stationary=q[:1].expand(5,-1,-1)
    assert command_motion_jitter(stationary,stamps).eq(0).all()
    assert sum(command_native_smooth_parts(q[:1]).values()).eq(0).all()


def test_native_translation_units_and_frame_average():
    q=torch.zeros(16,1,26,dtype=torch.double)
    q[:,0,20]=torch.arange(16,dtype=torch.double)*.001
    stamps=torch.arange(16,dtype=torch.double)[:,None]*8
    torch.testing.assert_close(command_motion_jitter(q,stamps),torch.tensor([15/16*125/1e6],dtype=q.dtype))


def test_motion_gradient_and_zero_rotation_are_finite():
    q=torch.zeros(4,1,26,dtype=torch.double);q[2,0,0]=.08;q[2,0,20]=.003
    q.requires_grad_();stamps=torch.arange(4,dtype=torch.double)[:,None]*(1000/120)
    assert torch.autograd.gradcheck(lambda x:command_motion_jitter(x,stamps),(q,),fast_mode=True)
    gradient=torch.autograd.grad(command_motion_jitter(q,stamps).sum(),q)[0]
    assert torch.isfinite(gradient).all() and gradient[2,0,20]>0


def test_rollout_keeps_clean_commands_separate_from_noisy_state(monkeypatch):
    import moxian_web.cart_training as training
    class Policy:
        lower=torch.full((20,),-1.);upper=torch.ones(20)
        def initial_state(self,q):return torch.cat((q,torch.zeros(len(q),6)),dim=-1)
        def __call__(self,q,**_):return self.target-q
        def apply_delta(self,q,delta):return q+delta
    model=Policy();model.target=torch.zeros(1,26,requires_grad=True)
    dt=torch.full((1,10),1/120)
    monkeypatch.setattr(training,'batch_future',lambda *args,**kwargs:dict(future_dt=dt))
    mask=torch.zeros(16,dtype=torch.bool)
    rows=[dict(timestamp_ms=k*1000/120,_card_targets=dict(anchor_position=torch.zeros(16,3),mask=mask,force=torch.zeros(16))) for k in range(14)]
    obj=ReachObjective(PlaneHand());obj.hand.device='cpu'
    batch=[dict(length=4,rows=rows,initial_q=torch.zeros(20))]
    tr=training.predict_trajectory(model,obj,batch,step_noise_rad=.1)
    commands=torch.stack(tr.commands);states=torch.stack(tr.predictions)
    torch.testing.assert_close(commands,torch.zeros_like(commands),atol=1e-7,rtol=0)
    assert states[...,:20].abs().max()>.01
    assert (states[...,:20]-commands[...,:20]).abs().max()<=.100001
    loss,parts=obj.trajectory_loss(tr)
    assert parts['motion_jitter']==0
    loss.backward();assert torch.isfinite(model.target.grad).all()
