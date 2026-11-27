"""Ten-frame measured EMF policies with five independent finger decoders."""
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class CrossBlock(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True, bias=False)
        self.norm = nn.LayerNorm(width)
        self.feed = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))
        self.out_norm = nn.LayerNorm(width)
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
    def __init__(self, lower, upper, architecture='attention',
                 position_mean=None, position_scale=None, width=128, decoder_width=None,
                 tactile_encoder="dense", tactile_groups=None):
        super().__init__()
        if architecture!='attention':
            raise ValueError('CART requires the measured-EMF attention architecture')
        low, high = torch.as_tensor(lower).float(), torch.as_tensor(upper).float()
        if low.shape != (20,) or high.shape != (20,) or not torch.isfinite(torch.stack((low, high))).all() or (low >= high).any():
            raise ValueError('CART requires finite 20-joint output bounds')
        self.register_buffer('lower', low); self.register_buffer('upper', high)
        self.register_buffer('position_mean', torch.zeros(3) if position_mean is None else torch.as_tensor(position_mean).float())
        self.register_buffer('position_scale', torch.ones(3)*.1 if position_scale is None else torch.as_tensor(position_scale).float())
        if self.position_mean.shape != (3,) or self.position_scale.shape != (3,) or not torch.isfinite(self.position_mean).all() or not torch.isfinite(self.position_scale).all() or (self.position_scale <= 0).any():
            raise ValueError('Invalid position normalization')
        if width <= 0 or width % 4:raise ValueError('Policy width must be a positive multiple of four')
        decoder_width=2*width if decoder_width is None else decoder_width
        if decoder_width <= 0:raise ValueError('Decoder width must be positive')
        self.width=width;self.decoder_width=decoder_width
        self.architecture=architecture;self.checkpoint_observations=False
        self.emf=nn.Linear(9,width)
        self.tactile_encoder=tactile_encoder
        if tactile_encoder=="dense":
            self.touch=nn.Linear(4,width);self.sid=nn.Embedding(460,width)
        elif tactile_encoder=="pointnet":
            from .cart_pointnet import AnatomicalPointEncoder
            self.touch=AnatomicalPointEncoder(tactile_groups,width)
        else:raise ValueError("Unknown tactile encoder")
        self.time=nn.Linear(1,width);self.finger=nn.Embedding(5,width)
        self.frame=nn.Embedding(10,width)
        self.current=nn.Linear(20,width)
        self.null_touch=nn.Parameter(torch.zeros(1,1,width));self.null_emf=nn.Parameter(torch.zeros(1,1,width))
        self.blocks=nn.ModuleList([CrossBlock(width),CrossBlock(width)]);self.pool=CrossBlock(width);incoming=width
        self.decoders=nn.ModuleList([nn.Sequential(nn.Linear(incoming,decoder_width),nn.LayerNorm(decoder_width),nn.GELU(),
            nn.Linear(decoder_width,decoder_width),nn.GELU(),nn.Linear(decoder_width,4),nn.Tanh()) for _ in range(5)])
        # A fresh policy holds its supplied current state, with an identity
        # state path through BPTT. Delta is not limited around a future IK pose.
        for decoder in self.decoders:
            nn.init.zeros_(decoder[-2].weight);nn.init.zeros_(decoder[-2].bias)

    state_size=20

    def initial_state(self,q):
        return q

    def decode_fingers(self,context):
        return torch.cat([decoder(context) for decoder in self.decoders],-1)

    def normalized_emf(self,future_emf,valid):
        safe=torch.where(valid[...,None],future_emf,0.)
        pose=torch.cat(((safe[...,:3]-self.position_mean)/self.position_scale,safe[...,3:]),-1)
        return pose*valid[...,None]

    def sensor_context(self,pose,valid,future_dt,tactile_positions,tactile_state,tv,tactile_tokens=None,tactile_token_valid=None):
        values=(pose,valid,future_dt,tactile_positions,tactile_state,tv,tactile_tokens,tactile_token_valid)
        if self.checkpoint_observations and torch.is_grad_enabled():
            return checkpoint(self.encode_attention,*values,use_reentrant=False,preserve_rng_state=False)
        return self.encode_attention(*values)

    def encode_prepared_observations(self,future_emf,emf_valid,future_dt,tactile_positions,tactile_state,tactile_valid):
        valid=emf_valid.bool()
        return self.sensor_context(self.normalized_emf(future_emf,valid),valid,future_dt,
            tactile_positions,tactile_state,tactile_valid.bool())

    def encode_tactile_frames(self,positions,state,valid):
        """Encode measured frames before window-specific time/position embeddings."""
        points=torch.where(valid[...,None],positions,0.)
        state=torch.where(valid,state,0.)
        normalized=(points-self.position_mean)/self.position_scale*valid[...,None]
        return self.touch(normalized,state,valid)

    def encode_attention(self,pose,valid,future_dt,tactile_positions,tactile_state,tv,
                         tactile_tokens=None,tactile_token_valid=None):
        b=len(pose);device=pose.device
        frame=self.frame(torch.arange(10,device=device))[None,:,None]
        timing=self.time(future_dt[...,None])[:,:,None]
        e=self.emf(pose)+frame+timing+self.finger(torch.arange(5,device=device))[None,None]
        if tactile_tokens is not None:
            t=tactile_tokens;tv=tactile_token_valid
        elif self.tactile_encoder=='pointnet':
            t,tv=self.encode_tactile_frames(tactile_positions,tactile_state,tv)
        else:
            tp=torch.where(tv[...,None],tactile_positions,0.);ts=torch.where(tv,tactile_state,0.)
            tx=torch.cat(((tp-self.position_mean)/self.position_scale,ts[...,None]),-1)*tv[...,None]
            t=self.touch(tx)+self.sid(torch.arange(460,device=device))[None,None]
        t=t+frame+timing
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
                tactile_state=None,tactile_valid=None,*,validate=True,tactile_tokens=None,tactile_token_valid=None):
        b=len(current_q)
        if current_q.shape!=(b,self.state_size) or future_emf.shape!=(b,10,5,9) or emf_valid.shape!=(b,10,5) or future_dt.shape!=(b,10):
            raise ValueError(f'Expected q[B,{self.state_size}], EMF[B,10,5,9], valid[B,10,5], dt[B,10]')
        valid=emf_valid.bool()
        if validate and (not torch.isfinite(current_q).all() or not torch.isfinite(future_emf[valid]).all() or not torch.isfinite(future_dt).all()):
            raise ValueError('Nonfinite CART input')
        if validate and ((future_dt<=0).any() or (future_dt[:,1:]<=future_dt[:,:-1]).any() or (torch.diff(future_dt,dim=1)>.250001).any() or (future_dt[:,0]>.250001).any()):
            raise ValueError('Future timestamps must increase without a stream gap')
        pose=self.normalized_emf(future_emf,valid)
        if tactile_tokens is not None:
            if (self.tactile_encoder!='pointnet' or tactile_tokens.shape!=(b,10,6,self.width)
                    or tactile_token_valid is None or tactile_token_valid.shape!=(b,10,6)):
                raise ValueError('Expected prepared anatomical tactile tokens[B,10,6,width]')
            return self.decode_observation_context(current_q,self.sensor_context(pose,valid,future_dt,
                None,None,None,tactile_tokens,tactile_token_valid))
        if tactile_positions is None or tactile_state is None or tactile_valid is None or tactile_positions.shape!=(b,10,460,3) or tactile_state.shape!=(b,10,460) or tactile_valid.shape!=(b,10,460):
            raise ValueError('Attention requires tactile positions[B,10,460,3], state/valid[B,10,460]')
        tv=tactile_valid.bool()
        if validate and (not torch.isfinite(tactile_positions[tv]).all() or not torch.isfinite(tactile_state[tv]).all() or ((tactile_state[tv]!=0)&(tactile_state[tv]!=1)).any()):
            raise ValueError('Tactile input must contain finite positions and binary states')
        return self.decode_observation_context(current_q,self.sensor_context(pose,valid,future_dt,tactile_positions,tactile_state,tv))




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
