"""Future EMF/tactile policies; retained v4 history helpers are read-only legacy."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .cart_protocol import HISTORY_LENGTH

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


class CausalBlock(nn.Module):
    def __init__(self, incoming, outgoing, dilation):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(incoming, outgoing, 3, dilation=dilation)
        self.skip = nn.Conv1d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()
        self.norm = nn.LayerNorm(outgoing)

    def forward(self, x):
        y = self.conv(F.pad(x, (self.pad, 0))) + self.skip(x)
        return F.gelu(self.norm(y.transpose(1, 2))).transpose(1, 2)


class LegacyHistoryNetwork(nn.Module):
    def __init__(self, lower, upper, levels=LEVELS):
        super().__init__()
        lower, upper = torch.as_tensor(lower).float(), torch.as_tensor(upper).float()
        if lower.shape != (20,) or upper.shape != (20,) or not torch.isfinite(torch.stack((lower, upper))).all() or (lower >= upper).any():
            raise ValueError('CART requires finite 20-joint output bounds')
        self.register_buffer('lower', lower)
        self.register_buffer('upper', upper)
        self.tcn = nn.Sequential(CausalBlock(22, 64, 1), CausalBlock(64, 64, 2), CausalBlock(64, 64, 4))
        self.encoder = nn.Sequential(nn.Linear(79, 256), nn.LayerNorm(256), nn.GELU(),
                                     nn.Linear(256, 256), nn.GELU(), nn.Linear(256, len(levels)))
        self.fsq = FiniteScalarQuantizer(levels)
        self.decoder = nn.Sequential(nn.Linear(len(levels), 256), nn.LayerNorm(256), nn.GELU(),
                                     nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 20), nn.Tanh())

    def normalize(self, q):
        return 2 * (q - self.lower) / (self.upper - self.lower) - 1

    def forward(self, points, joint_history, history_mask=None, history_dt=None, *, return_codes=False):
        b = len(points)
        if points.shape != (b, 5, 3) or joint_history.shape != (b, HISTORY_LENGTH, 20):
            raise ValueError('Expected points[B,5,3], joint_history[B,10,20]')
        if history_mask is None:
            history_mask = torch.ones((b, HISTORY_LENGTH), device=points.device, dtype=torch.bool)
        if history_dt is None:
            history_dt = points.new_zeros((b, HISTORY_LENGTH))
        if history_mask.shape != (b, HISTORY_LENGTH) or history_dt.shape != (b, HISTORY_LENGTH):
            raise ValueError('Expected history mask/dt[B,10]')
        if not torch.isfinite(points).all() or not torch.isfinite(joint_history).all() or not torch.isfinite(history_dt).all() or (history_dt < 0).any():
            raise ValueError('Nonfinite CART input or negative history interval')
        mask = history_mask.to(points.dtype)
        normalized = self.normalize(joint_history) * mask[..., None]
        x = torch.cat((normalized, mask[..., None], history_dt.clamp(max=.25)[..., None] * mask[..., None]), -1)
        context = self.tcn(x.transpose(1, 2))[:, :, -1]
        codes, indices = self.fsq(self.encoder(torch.cat((points.flatten(1), context), -1)))
        q = self.lower + (self.decoder(codes) + 1) * .5 * (self.upper - self.lower)
        return dict(q=q, codes=codes, indices=indices) if return_codes else q


class JointHistory:
    """Caller-owned outputs only: no current-frame angle can enter its input."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.entries = []
        self.epoch = None

    def input(self, stamp, epoch=None):
        if not np.isfinite(stamp):
            raise ValueError('Nonfinite history timestamp')
        if self.entries and (not 0 < stamp - self.entries[-1][0] <= 250 or
                             epoch is not None and self.epoch is not None and epoch != self.epoch):
            self.reset()
        self.epoch = epoch
        q = np.zeros((10, 20), np.float32)
        mask = np.zeros(10, bool)
        dt = np.zeros(10, np.float32)
        n = len(self.entries)
        if n:
            stamps = [s for s, _ in self.entries] + [stamp]
            q[-n:] = [v for _, v in self.entries]
            mask[-n:] = True
            dt[-n:] = np.diff(stamps) / 1000
        return q, mask, dt

    def append(self, stamp, q):
        q = np.asarray(q, np.float32)
        if q.shape != (20,) or not np.isfinite(q).all() or not np.isfinite(stamp):
            raise ValueError('Invalid joint history sample')
        if self.entries and stamp <= self.entries[-1][0]:
            raise ValueError('History must be strictly increasing')
        self.entries = (self.entries + [(float(stamp), q.copy())])[-HISTORY_LENGTH:]


class CrossBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True, bias=False)
        self.norm = nn.LayerNorm(128)
        self.feed = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))
        self.out_norm = nn.LayerNorm(128)

    def forward(self, query, values, padding):
        context = self.attention(query, values, values, key_padding_mask=padding, need_weights=False)[0]
        query = self.norm(query + context)
        return self.out_norm(query + self.feed(query))


class CartNetwork(nn.Module):
    """q_t + exactly ten future sensor frames -> normalized absolute q_(t+1).

    Physical units are converted only at the input/output boundary. FSQ uses
    a straight-through estimator; the attention policy has no quantization.
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
        self.architecture = architecture
        if architecture == 'fsq':
            # 450 pose values, 50 validity bits, ten time offsets, current q.
            self.encoder = nn.Sequential(nn.Linear(530, 256), nn.LayerNorm(256), nn.GELU(),
                nn.Linear(256, 256), nn.GELU(), nn.Linear(256, len(levels)))
            self.fsq = FiniteScalarQuantizer(levels)
            incoming = len(levels)
        else:
            self.emf = nn.Linear(9, 128); self.touch = nn.Linear(4, 128)
            self.time = nn.Linear(1, 128); self.finger = nn.Embedding(5, 128)
            self.sid = nn.Embedding(460, 128); self.frame = nn.Embedding(10, 128)
            self.current = nn.Linear(20, 128)
            self.null_touch = nn.Parameter(torch.zeros(1, 1, 128))
            self.null_emf = nn.Parameter(torch.zeros(1, 1, 128))
            self.blocks = nn.ModuleList([CrossBlock(), CrossBlock()])
            self.pool = CrossBlock(); incoming = 128
        self.decoder = nn.Sequential(nn.Linear(incoming, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 20), nn.Tanh())

    def normalize(self, q):
        return 2*(q-self.lower)/(self.upper-self.lower)-1

    def unnormalize(self, q):
        return self.lower+(q+1)*.5*(self.upper-self.lower)

    def forward(self, current_q, future_emf, emf_valid, future_dt, tactile_positions=None,
                tactile_state=None, tactile_valid=None, *, return_codes=False, validate=True):
        b = len(current_q)
        if current_q.shape != (b, 20) or future_emf.shape != (b, 10, 5, 9) or emf_valid.shape != (b, 10, 5) or future_dt.shape != (b, 10):
            raise ValueError('Expected q[B,20], emf[B,10,5,9], valid[B,10,5], dt[B,10]')
        valid = emf_valid.bool()
        if validate and (not torch.isfinite(current_q).all() or not torch.isfinite(future_emf[valid]).all() or not torch.isfinite(future_dt).all()):
            raise ValueError('Nonfinite CART input')
        if validate and ((future_dt <= 0).any() or (future_dt[:, 1:] <= future_dt[:, :-1]).any() or (torch.diff(future_dt, dim=1) > .250001).any() or (future_dt[:, 0] > .250001).any()):
            raise ValueError('Future timestamps must increase without a stream gap')
        pose = torch.where(valid[..., None], future_emf, 0.)
        xyz = (pose[..., :3]-self.position_mean)/self.position_scale
        pose = torch.cat((xyz, pose[..., 3:]), -1)*valid[..., None]
        q = self.normalize(current_q)
        extra = {}
        if self.architecture == 'fsq':
            z = self.encoder(torch.cat((q, pose.flatten(1), valid.flatten(1).to(q), future_dt), -1))
            context, indices = self.fsq(z); extra = dict(codes=context, indices=indices)
        else:
            if tactile_positions is None or tactile_state is None or tactile_valid is None or tactile_positions.shape != (b,10,460,3) or tactile_state.shape != (b,10,460) or tactile_valid.shape != (b,10,460):
                raise ValueError('Attention requires tactile positions[B,10,460,3], state/valid[B,10,460]')
            tv = tactile_valid.bool()
            if validate and (not torch.isfinite(tactile_positions[tv]).all() or not torch.isfinite(tactile_state[tv]).all() or ((tactile_state[tv] != 0) & (tactile_state[tv] != 1)).any()):
                raise ValueError('Tactile input must contain finite positions and binary states')
            tp = torch.where(tv[...,None], tactile_positions, 0.)
            ts = torch.where(tv, tactile_state, 0.)
            tx = torch.cat(((tp-self.position_mean)/self.position_scale, ts[...,None]), -1)*tv[...,None]
            frame = self.frame(torch.arange(10,device=q.device))[None,:,None]
            timing = self.time(future_dt[...,None])[:,:,None]
            e = self.emf(pose)+frame+timing+self.finger(torch.arange(5,device=q.device))[None,None]
            t = self.touch(tx)+frame+timing+self.sid(torch.arange(460,device=q.device))[None,None]
            # An always-valid null token avoids NaNs for entirely unknown sensors.
            t = torch.cat((t.flatten(1,2), self.null_touch.expand(b,-1,-1)),1)
            padding = torch.cat((~tv.flatten(1), torch.zeros(b,1,device=q.device,dtype=torch.bool)),1)
            e = e.flatten(1,2)
            for block in self.blocks: e = block(e,t,padding)
            e = torch.cat((e,self.null_emf.expand(b,-1,-1)),1)
            epad = torch.cat((~valid.flatten(1),torch.zeros(b,1,device=q.device,dtype=torch.bool)),1)
            context = self.pool(self.current(q)[:,None],e,epad)[:,0]
        normalized = self.decoder(context)
        return dict(q=self.unnormalize(normalized), normalized_q=normalized, **extra) if return_codes else normalized


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
