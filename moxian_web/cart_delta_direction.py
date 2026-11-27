"""GeoRT-style local input/output motion direction, using exact robot FK.

The recurrent q26 input is identical in both branches. Independent per-finger
translations are constant across the ten measured frames; associated taxels
move with their finger, while palm, orientations and contact flags stay fixed.
This is a local observation augmentation, not a physically reconstructed glove
trajectory or a new contact target. Native contact cost uses original data only.
"""
import math
import torch
from torch.nn import functional as F


def perturb_finger_observations(inputs,groups,delta):
    """Return a fresh observation pair; never alter cached measured tensors."""
    emf=inputs['future_emf'];valid=inputs['emf_valid'].bool()
    if delta.shape!=(len(emf),5,3):raise ValueError('Expected finger displacement [B,5,3]')
    groups=torch.as_tensor(groups,device=emf.device,dtype=torch.long)
    if groups.shape!=(460,) or bool(((groups<0)|(groups>5)).any()):
        raise ValueError('Expected 460 finger/palm group IDs')
    result=dict(inputs);pose=emf.clone()
    pose[...,:3]=torch.where(valid[...,None],pose[...,:3]+delta[:,None],pose[...,:3])
    result['future_emf']=pose
    shifts=torch.cat((delta,torch.zeros_like(delta[:,:1])),1)[:,groups]
    tv=inputs['tactile_valid'].bool()
    # Do not shift taxels belonging to a finger whose EMF observation is missing.
    finger_valid=torch.cat((valid,torch.zeros_like(valid[...,:1])),dim=-1)[...,groups]
    mask=tv&finger_valid
    result['tactile_positions']=torch.where(mask[...,None],inputs['tactile_positions']+shifts[:,None],inputs['tactile_positions'])
    return result


def motion_direction_loss(source_delta,robot_delta,valid,epsilon=1.e-5):
    """1-cos in [0,2], equivalent to GeoRT's -cos up to a constant.

    The epsilon avoids division by zero and supplies a nonzero learning signal
    when a policy has zero local response. Both prediction branches retain grads.
    """
    a=F.normalize(source_delta,dim=-1,eps=epsilon)
    b=F.normalize(robot_delta,dim=-1,eps=epsilon)
    cosine=(a*b).sum(-1);weight=valid.to(cosine.dtype)
    count=weight.sum().clamp_min(1)
    loss=((1-cosine)*weight).sum()/count
    return loss,dict(delta_direction_cosine=(cosine*weight).sum()/count,
                     delta_response_mm=(robot_delta.norm(dim=-1)*weight).sum()/count*1000)


def delta_direction_regularizer(model,hand,batch,tr,*,minimum_m=.001,maximum_m=.003,max_samples=128):
    if not (math.isfinite(minimum_m) and math.isfinite(maximum_m) and 0<minimum_m<=maximum_m):
        raise ValueError('Invalid perturbation magnitude')
    if not hasattr(tr,'input_state') or len(tr.commands)!=1:
        raise ValueError('Direction regularizer requires a detached single-step trajectory')
    if getattr(model,'tactile_encoder',None)!='pointnet':
        raise ValueError('Direction regularizer requires anatomical PointNet groups')
    if max_samples<1:raise ValueError('Need positive direction sample count')
    selected=torch.randperm(len(batch),device=tr.input_state.device)[:max_samples]
    from .cart_fast import PreparedBatch
    selected_cpu=selected.cpu().tolist()
    offsets_all=getattr(tr,'observation_offsets',[len(w['rows'])-w['length']-10+1 for w in batch])
    offsets=[offsets_all[i] for i in selected_cpu]
    batch=PreparedBatch([batch[i] for i in selected_cpu],batch.device)
    inputs=batch.future_inputs(offsets)
    direction=F.normalize(torch.randn_like(inputs['future_emf'][:,0,:,:3]),dim=-1)
    scale=torch.empty_like(direction[...,:1]).uniform_(minimum_m,maximum_m)
    delta=direction*scale
    groups=model.touch.assignment.argmax(-1)  # immutable topology, not geometric search
    shifted=perturb_finger_observations(inputs,groups,delta)
    # Small finite differences must not be dominated by BF16 rounding. Recompute
    # both paired forwards in FP32, reusing the same detached recurrent state.
    from .cart_network import CrossBlock
    precision=[(block,block.attention_precision) for block in model.modules() if isinstance(block,CrossBlock)]
    try:
        model.set_attention_precision('float32')
        state=tr.input_state[selected].detach()
        base=model.apply_delta(state,model(state,**inputs,validate=False))
        moved=model.apply_delta(state,model(state,**shifted,validate=False))
    finally:
        for block,value in precision:block.attention_precision=value
    tips=hand.frames(torch.cat((base,moved),dim=0))[2][:,[4,8,12,16,20]]
    robot_delta=tips[len(base):]-tips[:len(base)]
    valid=inputs['emf_valid'][:,0].bool()
    loss,parts=motion_direction_loss(delta,robot_delta,valid)
    return loss,dict(delta_direction=loss,delta_direction_samples=loss.new_tensor(len(batch)),**parts)
