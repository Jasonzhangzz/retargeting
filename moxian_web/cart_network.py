"""Ten-frame measured EMF policies with five independent finger decoders."""
import math
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

LEVELS = (8, 8, 8, 8, 5, 5, 5, 5)


class FiniteScalarQuantizer(nn.Module):
    """FSQ bounded scalar grids, including the offset of even-sized grids."""
    def __init__(self, levels=LEVELS):
        super().__init__()
        if not levels or any(type(n) is not int or n < 2 for n in levels):
            raise ValueError('FSQ levels must be integers >= 2')
        self.register_buffer('levels', torch.tensor(levels, dtype=torch.long))
        self.register_buffer('radix', torch.tensor([math.prod(levels[:i]) for i in range(len(levels))]))

    def forward(self, z):
        half = (self.levels.to(z) - 1) * 1.001 / 2
        offset = (self.levels % 2 == 0).to(z) * .5
        bounded = (z + torch.atanh(offset / half)).tanh() * half - offset
        rounded = bounded.round()
        codes = (bounded + (rounded - bounded).detach()) / (self.levels // 2).to(z)
        indices = ((rounded.long() + self.levels // 2) * self.radix).sum(-1)
        return codes, indices

    def decode_indices(self, indices):
        digits = (indices[..., None] // self.radix) % self.levels
        return (digits - self.levels // 2).float() / (self.levels // 2)


class CrossBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True, bias=False)
        self.norm = nn.LayerNorm(128)
        self.feed = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))
        self.out_norm = nn.LayerNorm(128)
        self.attention_precision = 'float32'

    def forward(self, query, values, padding):
        # BF16 enables fused CUDA attention. Geometry, normalization, the MLP
        # decoder and its joint-angle output retain FP32 precision.
        with torch.autocast('cuda',dtype=torch.bfloat16,
                enabled=query.is_cuda and self.attention_precision=='bfloat16'):
            context = self.attention(query, values, values, key_padding_mask=padding, need_weights=False)[0]
        context=context.to(query.dtype)
        query = self.norm(query + context)
        return self.out_norm(query + self.feed(query))


class JointDeltaPolicy(nn.Module):
    """Physical-unit boundaries for the five-finger delta policy."""
    def normalize(self, q):
        return 2*(q-self.lower)/(self.upper-self.lower)-1

    def unnormalize(self, q):
        return self.lower+(q+1)*.5*(self.upper-self.lower)

    def delta_to_radians(self,delta):
        return delta*.5*(self.upper-self.lower)

    def apply_delta(self,current_q,delta,*,normalized=False):
        """Residual state transition; normalization of an increment has no bias.

        Joint-limit projection is piecewise differentiable. In the interior,
        the identity state path and the complete delta graph are preserved.
        """
        next_q=(self.normalize(current_q)+delta).clamp(-1.,1.)
        return next_q if normalized else self.unnormalize(next_q)

    def set_attention_precision(self,precision):
        if precision not in ('float32','bfloat16'):raise ValueError('Invalid attention precision')
        for block in self.modules():
            if isinstance(block,CrossBlock):block.attention_precision=precision
        return self



class CartNetwork(JointDeltaPolicy):
    """Current q + ten measured future EMF poses/taxels -> normalized delta q.

    No future IK, skeleton or IK reference correction enters the policy graph.
    Each EMF receiver supplies position in metres and two rotation columns.
    """
    def __init__(self, lower, upper, architecture='attention', levels=LEVELS,
                 position_mean=None, position_scale=None):
        super().__init__()
        if architecture not in ('attention', 'fsq'):
            raise ValueError('architecture must be attention or fsq')
        low, high = torch.as_tensor(lower).float(), torch.as_tensor(upper).float()
        if low.shape != (20,) or high.shape != (20,) or not torch.isfinite(torch.stack((low, high))).all() or (low >= high).any():
            raise ValueError('CART requires finite 20-joint output bounds')
        self.register_buffer('lower', low); self.register_buffer('upper', high)
        self.register_buffer('position_mean', torch.zeros(3) if position_mean is None else torch.as_tensor(position_mean).float())
        self.register_buffer('position_scale', torch.ones(3)*.1 if position_scale is None else torch.as_tensor(position_scale).float())
        if self.position_mean.shape != (3,) or self.position_scale.shape != (3,) or not torch.isfinite(self.position_mean).all() or not torch.isfinite(self.position_scale).all() or (self.position_scale <= 0).any():
            raise ValueError('Invalid position normalization')
        self.architecture=architecture;self.checkpoint_observations=False
        if architecture=='fsq':
            # Current q:20; ten poses:10*5*9; masks:10*5; timestamps:10.
            self.encoder=nn.Sequential(nn.Linear(530,256),nn.LayerNorm(256),nn.GELU(),
                nn.Linear(256,256),nn.GELU(),nn.Linear(256,len(levels)))
            self.fsq=FiniteScalarQuantizer(levels);incoming=len(levels)
        else:
            self.emf=nn.Linear(9,128);self.touch=nn.Linear(4,128)
            self.time=nn.Linear(1,128);self.finger=nn.Embedding(5,128)
            self.sid=nn.Embedding(460,128);self.frame=nn.Embedding(10,128)
            self.current=nn.Linear(20,128)
            self.null_touch=nn.Parameter(torch.zeros(1,1,128));self.null_emf=nn.Parameter(torch.zeros(1,1,128))
            self.blocks=nn.ModuleList([CrossBlock(),CrossBlock()]);self.pool=CrossBlock();incoming=128
        self.decoders=nn.ModuleList([nn.Sequential(nn.Linear(incoming,256),nn.LayerNorm(256),nn.GELU(),
            nn.Linear(256,256),nn.GELU(),nn.Linear(256,4),nn.Tanh()) for _ in range(5)])
        # A fresh policy holds its supplied current state, with an identity
        # state path through BPTT. Delta is not limited around a future IK pose.
        for decoder in self.decoders:
            nn.init.zeros_(decoder[-2].weight);nn.init.zeros_(decoder[-2].bias)

    def decode_fingers(self,context):
        return torch.cat([decoder(context) for decoder in self.decoders],-1)

    def normalized_emf(self,future_emf,valid):
        safe=torch.where(valid[...,None],future_emf,0.)
        pose=torch.cat(((safe[...,:3]-self.position_mean)/self.position_scale,safe[...,3:]),-1)
        return pose*valid[...,None]

    def sensor_context(self,pose,valid,future_dt,tactile_positions,tactile_state,tv):
        values=(pose,valid,future_dt,tactile_positions,tactile_state,tv)
        if self.checkpoint_observations and torch.is_grad_enabled():
            return checkpoint(self.encode_attention,*values,use_reentrant=False,preserve_rng_state=False)
        return self.encode_attention(*values)

    def encode_prepared_observations(self,future_emf,emf_valid,future_dt,tactile_positions,tactile_state,tactile_valid):
        valid=emf_valid.bool()
        return self.sensor_context(self.normalized_emf(future_emf,valid),valid,future_dt,
            tactile_positions,tactile_state,tactile_valid.bool())

    def encode_attention(self,pose,valid,future_dt,tactile_positions,tactile_state,tv):
        b=len(pose);device=pose.device
        tp=torch.where(tv[...,None],tactile_positions,0.);ts=torch.where(tv,tactile_state,0.)
        tx=torch.cat(((tp-self.position_mean)/self.position_scale,ts[...,None]),-1)*tv[...,None]
        frame=self.frame(torch.arange(10,device=device))[None,:,None]
        timing=self.time(future_dt[...,None])[:,:,None]
        e=self.emf(pose)+frame+timing+self.finger(torch.arange(5,device=device))[None,None]
        t=self.touch(tx)+frame+timing+self.sid(torch.arange(460,device=device))[None,None]
        t=torch.cat((t.flatten(1,2),self.null_touch.expand(b,-1,-1)),1)
        padding=torch.cat((~tv.flatten(1),torch.zeros(b,1,device=device,dtype=torch.bool)),1)
        e=e.flatten(1,2)
        for block in self.blocks:e=block(e,t,padding)
        e=torch.cat((e,self.null_emf.expand(b,-1,-1)),1)
        epad=torch.cat((~valid.flatten(1),torch.zeros(b,1,device=device,dtype=torch.bool)),1)
        return e,epad

    def decode_observation_context(self,current_q,encoded):
        e,epad=encoded
        context=self.pool(self.current(self.normalize(current_q))[:,None],e,epad)[:,0]
        return self.decode_fingers(context)

    def forward(self,current_q,future_emf,emf_valid,future_dt,tactile_positions=None,
                tactile_state=None,tactile_valid=None,*,return_codes=False,validate=True):
        b=len(current_q)
        if current_q.shape!=(b,20) or future_emf.shape!=(b,10,5,9) or emf_valid.shape!=(b,10,5) or future_dt.shape!=(b,10):
            raise ValueError('Expected q[B,20], EMF[B,10,5,9], valid[B,10,5], dt[B,10]')
        valid=emf_valid.bool()
        if validate and (not torch.isfinite(current_q).all() or not torch.isfinite(future_emf[valid]).all() or not torch.isfinite(future_dt).all()):
            raise ValueError('Nonfinite CART input')
        if validate and ((future_dt<=0).any() or (future_dt[:,1:]<=future_dt[:,:-1]).any() or (torch.diff(future_dt,dim=1)>.250001).any() or (future_dt[:,0]>.250001).any()):
            raise ValueError('Future timestamps must increase without a stream gap')
        pose=self.normalized_emf(future_emf,valid);q=self.normalize(current_q);extra={}
        if self.architecture=='fsq':
            z=self.encoder(torch.cat((q,pose.flatten(1),valid.flatten(1).to(q),future_dt),-1))
            context,indices=self.fsq(z);extra=dict(codes=context,indices=indices)
            delta=self.decode_fingers(context)
        else:
            if tactile_positions is None or tactile_state is None or tactile_valid is None or tactile_positions.shape!=(b,10,460,3) or tactile_state.shape!=(b,10,460) or tactile_valid.shape!=(b,10,460):
                raise ValueError('Attention requires tactile positions[B,10,460,3], state/valid[B,10,460]')
            tv=tactile_valid.bool()
            if validate and (not torch.isfinite(tactile_positions[tv]).all() or not torch.isfinite(tactile_state[tv]).all() or ((tactile_state[tv]!=0)&(tactile_state[tv]!=1)).any()):
                raise ValueError('Tactile input must contain finite positions and binary states')
            delta=self.decode_observation_context(current_q,self.sensor_context(pose,valid,future_dt,tactile_positions,tactile_state,tv))
        return dict(delta_q=self.delta_to_radians(delta),normalized_delta_q=delta,
            q=self.apply_delta(current_q,delta),normalized_q=self.apply_delta(current_q,delta,normalized=True),
            **extra) if return_codes else delta


class FutureBuffer:
    """Caller-owned true lookahead; never extrapolates or flushes a short tail."""
    def __init__(self): self.reset()

    def reset(self):
        self.entries = []; self.epoch = None; self.last_stamp = None

    def append(self, stamp, value, epoch=None):
        if not np.isfinite(stamp): raise ValueError('Nonfinite observation timestamp')
        if self.last_stamp is not None and (not 0 < stamp-self.last_stamp <= 250 or epoch != self.epoch):
            self.reset()
        self.epoch = epoch; self.last_stamp = stamp
        self.entries.append((float(stamp),value))
        if len(self.entries) < 11: return None
        result = self.entries[:11]; self.entries.pop(0)
        return result
