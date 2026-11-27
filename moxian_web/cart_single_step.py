"""One differentiable policy step following a detached policy warmup step."""
import torch
from .cart_training import RolloutTrajectory,perturb_rollout_state
from .cart_reach_loss import ReachObjective,command_native_smooth_parts


def predict_single_step(model,objective,batch,*,step_noise_rad=0.,warmup_steps=1):
    if not isinstance(warmup_steps,int) or warmup_steps<0 or any(w['length']<=warmup_steps for w in batch):
        raise ValueError('Warmup depth must lie within the recorded rollout window')
    device=objective.hand.device
    offsets=[len(w['rows'])-w['length']-10 for w in batch]
    cached=None
    if (warmup_steps==1 and getattr(model,'tactile_encoder',None)=='pointnet'
            and getattr(model,'deduplicate_observations',True) and hasattr(batch,'single_step_inputs')):
        cached=batch.single_step_inputs(model,offsets)
    def inputs(shift):
        return dict(cached[shift] if cached is not None else batch.future_inputs([i+shift for i in offsets]),validate=False)
    # Cached IK seeds only the window boundary. The trained step receives the
    # preceding policy's full q20+wrist6 state, never future robot actions.
    with torch.no_grad():
        q=model.initial_state(torch.stack([torch.as_tensor(w['initial_q'],device=device) for w in batch]))
        previous_command=q
        previous_state=q
        for step in range(warmup_steps):
            previous_command=model.apply_delta(previous_state,model(previous_state,**inputs(step)))
            previous_state=perturb_rollout_state(model,previous_command,step_noise_rad)
    current=previous_state.detach().requires_grad_()
    command=model.apply_delta(current,model(current,**inputs(warmup_steps)))
    predicted=perturb_rollout_state(model,command,step_noise_rad)
    rows=[w['rows'][i+warmup_steps+1] for w,i in zip(batch,offsets)]
    previous_rows=[w['rows'][i+warmup_steps] for w,i in zip(batch,offsets)]
    tr=RolloutTrajectory([(predicted,rows,current,None,previous_rows)],batch,commands=[command])
    tr.previous_command=previous_command.detach()
    tr.input_state=current
    tr.warmup_steps=warmup_steps
    tr.observation_offsets=[i+warmup_steps for i in offsets]
    return tr


class SingleStepReachObjective(ReachObjective):
    def trajectory_loss(self,tr):
        loss,parts=super().trajectory_loss(tr)
        if not hasattr(tr,'previous_command') or getattr(tr,'warmup_steps',1)==0:return loss,parts
        # Native smoothness for one actual transition. The helper averages over
        # two poses including a zero first edge; undo that factor of two.
        smooth={k:2*v.mean() for k,v in command_native_smooth_parts(
            torch.stack((tr.previous_command,tr.commands[0]))).items()}
        motion=sum(smooth.values())
        return loss+motion,dict(parts,total=loss+motion,motion_jitter=motion,**smooth)
