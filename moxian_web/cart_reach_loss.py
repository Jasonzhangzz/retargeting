"""Exact nearest semantic-palmar reach energy plus weak temporal regularizers.

The nearest witness is detached. For squared Euclidean distance to a fixed
surface, 2*(x-proj(x)) is the exact first derivative almost everywhere (envelope
rule). This is piecewise differentiable, not a globally smooth SDF or Hessian.
"""
import torch

VERSION='nearest_palmar_reach_native_smooth_no_jitter_v23'
SPEC=dict(version=VERSION,reach='mean_frames(sum_regions(rho^2*40^2*distance_mm^2))/1e6',
    smooth_joint_weight=.1,smooth_translation_weight=125**.5,smooth_rotation_weight=10.,
    smooth_normalization=1e6,contact_jitter_weight=0.,
    motion_jitter='native CARD first difference of clean pose commands; squared weights; rad joints, mm translation, nearest axis-angle chart rotation; no dt division; zero first-frame boundary contribution',
    contact_jitter='disabled; excluded from objective',
    surface='native_CARD_semantic_palmar_triangles',correspondence_alpha=0.,collision_loss=False,
    geometry_gradient='exact_first_order_envelope_through_FK_piecewise_differentiable',bptt_length=16,step_noise_rad=.1)


def command_native_smooth_parts(commands):
    """Native CARD energy /1e6, averaged over T emitted poses (first cost zero).

    Commands precede injected recurrence noise. The native nearest-equivalent
    axis-angle chart is retained exactly, rather than replacing it with a
    geodesic or a timestamp-scaled angular velocity penalty.
    """
    from retarget.baseline.card.math import axis_angle_near, exp_so3
    if len(commands)<2:
        zero=commands.sum((0,2))*0
        return dict(smooth_joint=zero,smooth_translation=zero,smooth_rotation=zero)
    previous=commands[:-1];current=commands[1:]
    reference=axis_angle_near(exp_so3(previous[...,23:]))
    rotation=axis_angle_near(exp_so3(current[...,23:]),reference)-reference
    scale=len(commands)*SPEC['smooth_normalization']
    return dict(
        smooth_joint=SPEC['smooth_joint_weight']**2*(current[...,:20]-previous[...,:20]).square().sum((0,2))/scale,
        smooth_translation=SPEC['smooth_translation_weight']**2*((current[...,20:23]-previous[...,20:23])*1000).square().sum((0,2))/scale,
        smooth_rotation=SPEC['smooth_rotation_weight']**2*rotation.square().sum((0,2))/scale)


def command_motion_jitter(commands, timestamps_ms):
    """Compatibility name for audit callers; current term is native smooth."""
    return sum(command_native_smooth_parts(commands).values())


class ReachObjective:
    command_smoothing=False
    collision=None
    def __init__(self,hand):
        self.hand=hand
        self.owners=[0]+[2+4*f+k for f in range(5) for k in range(3)]

    def distances(self,states,rows):
        p,r=self.hand.frames(states)[:2]
        targets=torch.stack([x['_card_targets']['anchor_position'] for x in rows])
        mask=torch.stack([x['_card_targets']['mask'] for x in rows]).bool()
        force=torch.stack([x['_card_targets']['force'] for x in rows]).clamp_min(0)
        maximum=force.masked_fill(~mask,0).amax(-1,keepdim=True)
        rho=(1+.5*force/maximum.clamp_min(1e-12)).clamp(1,1.5)
        distances=[];valid_regions=[]
        for region,owner in enumerate(self.owners):
            local=(r[:,owner].transpose(-1,-2)@(targets[:,region]-p[:,owner])[...,None]).squeeze(-1)
            # Inactive anchors still use finite coordinates but have zero weight.
            with torch.no_grad():hit,_,valid=self.hand.nearest(local,owner,palmar=True)
            valid_regions.append(valid|~mask[:,region])
            distances.append((local-hit.detach()).square().sum(-1))
        if not bool(torch.stack(valid_regions,-1).all()):raise ValueError('Missing active palmar projection')
        squared=torch.stack(distances,-1)
        return squared,mask,rho

    def trajectory_loss(self,trajectory):
        length=len(trajectory.pending);batch=len(trajectory.batch)
        states=torch.cat(trajectory.predictions)
        rows=[row for item in trajectory.pending for row in item[1]]
        sq,mask,rho=self.distances(states,rows)
        reach=(1600*rho.square()*sq*mask).sum(-1).mean()
        gap=(sq+1e-12).sqrt().reshape(length,batch,16)
        mask=mask.reshape(length,batch,16)
        # Require explicit commands: falling back to predictions would silently
        # reintroduce the injected process noise into the motion regularizer.
        smooth={k:v.mean() for k,v in command_native_smooth_parts(torch.stack(trajectory.commands)).items()}
        motion=sum(smooth.values())
        # Compatibility metric only: contact gap changes have no loss or gradient.
        jitter=reach.detach().new_zeros(())
        total=reach+motion
        active=mask.sum().clamp_min(1)
        parts=dict(total=total,contact=reach,motion_jitter=motion,contact_jitter=jitter,
            contact_mean_mm=(gap*mask).sum()/active*1000,contact_within_1mm=((gap<.001)&mask).sum()/active,
            active_anchors_per_frame=mask.float().sum(-1).mean(),**smooth)
        return total,parts
