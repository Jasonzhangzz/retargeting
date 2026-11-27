"""Floating robot wrist in the fixed demonstrator-wrist coordinate system.

The 20 finger angles remain the inputs of the rigid-invariant collision field.
The six additional state coordinates are translation (m) and a rotation vector
(rad). Observed targets never move with the predicted robot wrist.
"""
from dataclasses import replace
import math
import torch
from torch import nn
from torch.nn import functional as F
from .cart_network import CartNetwork
from .cart_objective import CartHand

VERSION='cart_floating_wrist_kinematics_v1'
OUTPUT='normalized_delta_fingers_wrist_v1'
TRANSITION='smooth_finger_and_relative_wrist_delta_v2'
ARCHITECTURE='cart_emf_cross_attention128_five_finger_wrist_delta_v1'
WRIST_LIMITS=(.1,.1,.1,math.pi,math.pi,math.pi)
WRIST_STEP=(.005,.005,.005,.05,.05,.05)
REFERENCE_DT=1/120


def smooth_bound_unit(x):
    """Smooth bounded projection; no hard clipping in the learned rollout."""
    return x+F.softplus(-1-x,beta=50)-F.softplus(x-1,beta=50)


def so3_exp(vector):
    """Rodrigues with a finite derivative at the identity."""
    x,y,z=vector.unbind(-1);zero=torch.zeros_like(x)
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*vector.shape[:-1],3,3)
    angle=vector.norm(dim=-1)
    a=torch.sinc(angle/math.pi)[...,None,None]
    b=(.5*torch.sinc(angle/(2*math.pi)).square())[...,None,None]
    return torch.eye(3,device=vector.device,dtype=vector.dtype)+a*skew+b*(skew@skew)


def relative_rotation(current,previous):
    """Spatial SO(3) log; adjacent bounded steps stay away from the pi cut."""
    rotation=so3_exp(current)@so3_exp(previous).transpose(-1,-2)
    sine=torch.stack((rotation[...,2,1]-rotation[...,1,2],
        rotation[...,0,2]-rotation[...,2,0],rotation[...,1,0]-rotation[...,0,1]),-1)*.5
    cosine=(rotation.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5
    angle=torch.atan2(sine.norm(dim=-1),cosine.clamp(-1,1))
    return sine/torch.sinc(angle/math.pi).clamp_min(1.e-4)[...,None]


def state_velocity(current,previous,dt):
    fingers=(current[...,:20]-previous[...,:20])/dt[...,None]
    translation=(current[...,20:23]-previous[...,20:23])/dt[...,None]
    angular=relative_rotation(current[...,23:26],previous[...,23:26])/dt[...,None]
    return torch.cat((fingers,translation,angular),-1)


class FloatingCartHand(CartHand):
    def frames(self,q):
        if q.shape[-1] not in (20,26):raise ValueError('Expected 20 finger joints or 26 finger/wrist state coordinates')
        p,r,landmarks,native_p,native_r=super().frames(q[...,:20])
        if q.shape[-1]==20:return p,r,landmarks,native_p,native_r
        rotation=so3_exp(q[...,23:26]);translation=q[...,20:23]
        p=(rotation[:,None]@p[...,None]).squeeze(-1)+translation[:,None]
        r=rotation[:,None]@r
        landmarks=(rotation[:,None]@landmarks[...,None]).squeeze(-1)+translation[:,None]
        # SDF is trained in the native, fixed robot wrist frame. Inverse-link
        # queries computed with p/r retain the floating-wrist gradient.
        return p,r,landmarks,native_p,native_r


class CartWristNetwork(CartNetwork):
    state_size=26

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.architecture!='attention':raise ValueError('Floating wrist currently requires the attention policy')
        self.current=nn.Linear(26,self.width)
        self.wrist_decoder=nn.Sequential(nn.Linear(self.width,self.decoder_width),nn.LayerNorm(self.decoder_width),nn.GELU(),
            nn.Linear(self.decoder_width,self.decoder_width),nn.GELU(),nn.Linear(self.decoder_width,6),nn.Tanh())
        nn.init.zeros_(self.wrist_decoder[-2].weight);nn.init.zeros_(self.wrist_decoder[-2].bias)
        self.register_buffer('wrist_limits',torch.tensor(WRIST_LIMITS))
        self.register_buffer('wrist_step',torch.tensor(WRIST_STEP))

    def initial_state(self,q):
        if q.shape[-1]==26:return q
        if q.shape[-1]!=20:raise ValueError('Initial state requires 20 finger joints')
        return torch.cat((q,q.new_zeros((*q.shape[:-1],6))),-1)

    def normalize(self,q):
        return torch.cat((super().normalize(q[...,:20]),q[...,20:]/self.wrist_limits),-1)

    def unnormalize(self,q):
        return torch.cat((super().unnormalize(q[...,:20]),q[...,20:]*self.wrist_limits),-1)

    def decode_fingers(self,context):
        return torch.cat((super().decode_fingers(context),self.wrist_decoder(context)),-1)

    def delta_to_radians(self,delta):
        return torch.cat((super().delta_to_radians(delta[...,:20]),delta[...,20:]*self.wrist_step),-1)

    def apply_delta(self,current_q,delta,*,normalized=False):
        finger_u=smooth_bound_unit(super().normalize(current_q[...,:20])+delta[...,:20])
        fingers=super().unnormalize(finger_u)
        wrist=smooth_bound_unit((current_q[...,20:]+delta[...,20:]*self.wrist_step)/self.wrist_limits)*self.wrist_limits
        state=torch.cat((fingers,wrist),-1)
        return self.normalize(state) if normalized else state
