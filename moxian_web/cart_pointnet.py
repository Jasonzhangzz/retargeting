"""Smooth anatomical pooling of measured taxels, before temporal attention."""
import torch
from torch import nn
from torch.nn import functional as F


def tactile_groups_from_topology(topology):
    from .contact import PATCH_IDS, FINGERS
    patches=topology['sid_patch_indices']
    if len(patches)!=460:raise ValueError('Expected 460 taxel assignments')
    names=[PATCH_IDS[int(i)].split('_')[0] for i in patches]
    return [5 if name=='palm' else FINGERS.index(name) for name in names]


class AnatomicalPointEncoder(nn.Module):
    """Shared point MLP, valid/active means, six tokens per measured frame.

    Separate active pooling prevents sparse contacts being diluted by inactive
    sensors. Masked coordinates cannot affect either pooled features or counts.
    No hard max or learned nearest-neighbour routing is used.
    """
    def __init__(self,groups,width):
        super().__init__()
        groups=torch.as_tensor(groups,dtype=torch.long)
        if groups.shape!=(460,) or (groups<0).any() or (groups>5).any():
            raise ValueError('Expected 460 anatomical group IDs in [0,5]')
        self.register_buffer('assignment',F.one_hot(groups,6).float())
        self.sid=nn.Embedding(460,16)
        self.point=nn.Sequential(nn.Linear(20,64),nn.GELU(),nn.Linear(64,64),nn.GELU())
        self.project=nn.Linear(130,width)
        self.group=nn.Embedding(6,width)

    def forward(self,normalized_points,state,valid):
        ids=self.sid.weight.expand(*state.shape[:-1],-1,-1)
        features=self.point(torch.cat((normalized_points,state[...,None],ids),-1))
        weights=valid.to(features.dtype)
        active=weights*state
        # Contract the small group weights against contiguous point features.
        # Multiplying/transposing features first creates GB-scale intermediates.
        group_weights=weights[...,None,:]*self.assignment.T
        contact_weights=active[...,None,:]*self.assignment.T
        count=group_weights.sum(-1);contact_count=contact_weights.sum(-1)
        pooled=torch.matmul(torch.cat((group_weights,contact_weights),-2),features)
        mean=pooled[...,:6,:]/count.clamp_min(1)[...,None]
        contact=pooled[...,6:,:]/contact_count.clamp_min(1)[...,None]
        fraction=contact_count/count.clamp_min(1)
        coverage=count/self.assignment.sum(0).clamp_min(1)
        token=self.project(torch.cat((mean,contact,fraction[...,None],coverage[...,None]),-1))+self.group.weight
        return token,count>0
