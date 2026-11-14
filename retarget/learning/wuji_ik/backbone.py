"""Shared four-block DiT backbone; temporal causality is at frame granularity."""
import math
import warnings
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

_FLEX = None


def timestep_embedding(t, width=256):
    frequency = torch.exp(-math.log(10000) * torch.arange(width//2, device=t.device).float() / (width//2))
    phase = t.float()[:, None] * frequency[None] * 1000
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


def apply_rope(q, k, positions):
    d = q.shape[-1]
    frequency = torch.exp(-math.log(10000) * torch.arange(0, d, 2, device=q.device).float() / d)
    phase = positions.float()[:, None] * frequency[None]
    cos, sin = phase.cos()[None, None], phase.sin()[None, None]
    def rotate(x):
        a, b = x[..., 0::2].float(), x[..., 1::2].float()
        return torch.stack((a*cos-b*sin, a*sin+b*cos), dim=-1).flatten(-2).to(x.dtype)
    return rotate(q), rotate(k)


class AttentionPlan:
    def __init__(self, causal, lengths, frames, device, backend):
        self.causal = causal; self.frames = frames; self.backend = backend
        self.lengths = lengths; self.positions = torch.arange(frames, device=device).repeat_interleave(6)
        self.block_mask = None; self.dense_mask = None
        self.unpadded = bool(torch.all(lengths == frames))
        self.flex_error = None

    def dense(self):
        if self.dense_mask is None:
            p = self.positions
            allowed = p[None, None, None, :] < self.lengths[:, None, None, None]
            if self.causal:
                allowed = allowed & (p[:, None] >= p[None, :])[None, None]
            self.dense_mask = allowed
        return self.dense_mask

    def flex_mask(self):
        if self.block_mask is None:
            from torch.nn.attention.flex_attention import create_block_mask
            lengths = self.lengths
            causal = self.causal
            def allowed(batch, head, query, key):
                # Block-mask construction probes rounded-up token indices.
                return (key // 6 < lengths[batch]) & ((key // 6 <= query // 6) | (not causal))
            self.block_mask = create_block_mask(allowed, B=len(lengths), H=None,
                Q_LEN=len(self.positions), KV_LEN=len(self.positions), device=str(lengths.device))
        return self.block_mask


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3*config.width)
        self.proj = nn.Linear(config.width, config.width)
        self.used_backends = set()

    def forward(self, x, plan):
        global _FLEX
        b, s, c = x.shape
        q, k, v = self.qkv(x).reshape(b, s, 3, self.heads, c//self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_rope(q, k, plan.positions)
        choice = plan.backend
        use_flex = choice == 'flex' or (choice == 'auto' and x.is_cuda and (plan.causal or not plan.unpadded))
        if use_flex:
            try:
                from torch.nn.attention.flex_attention import flex_attention
                if _FLEX is None:
                    _FLEX = torch.compile(flex_attention, dynamic=True)
                result = _FLEX(q, k, v, block_mask=plan.flex_mask())
                self.used_backends.add('flex_attention')
                return self.proj(result.transpose(1, 2).reshape(b, s, c))
            except (RuntimeError, ImportError, NotImplementedError) as exc:
                if choice == 'flex' or s > 768:
                    raise RuntimeError('FlexAttention failed; no local-window substitution was made. '
                                       'Use a supported GPU/runtime or explicitly select --backend sdpa.') from exc
                if plan.flex_error is None:
                    warnings.warn('Small-sequence FlexAttention failed; using SDPA math: ' + str(exc)[:180])
                    plan.flex_error = str(exc)
        eligible_flash = x.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and plan.unpadded and not plan.causal
        if choice == 'flash' and not eligible_flash:
            raise ValueError('Flash SDPA requires CUDA half/BF16 and unpadded noncausal tokens; use flex for frame causality')
        if eligible_flash and choice in ('auto', 'flash'):
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    result = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
                self.used_backends.add('flash_sdpa')
                return self.proj(result.transpose(1, 2).reshape(b, s, c))
            except RuntimeError:
                if choice == 'flash':
                    raise
        # Explicit math reference, never labelled FlashAttention.
        from torch.nn.attention import SDPBackend, sdpa_kernel
        mask = plan.dense() if plan.causal or not plan.unpadded else None
        with sdpa_kernel(SDPBackend.MATH):
            result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        self.used_backends.add('sdpa_math')
        return self.proj(result.transpose(1, 2).reshape(b, s, c))


class DiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(config.width, elementwise_affine=False)
        self.attention = SelfAttention(config)
        self.mlp = nn.Sequential(nn.Linear(config.width, config.ffn_width), nn.GELU(), nn.Linear(config.ffn_width, config.width))
        self.modulation = nn.Linear(config.width, 6*config.width)
        nn.init.zeros_(self.modulation.weight); nn.init.zeros_(self.modulation.bias)

    def forward(self, x, condition, plan):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(F.silu(condition)).chunk(6, dim=-1)
        h = self.norm1(x) * (1+scale1[:, None]) + shift1[:, None]
        x = x + gate1[:, None] * self.attention(h, plan)
        h = self.norm2(x) * (1+scale2[:, None]) + shift2[:, None]
        return x + gate2[:, None] * self.mlp(h)


class IKBackbone(nn.Module):
    def __init__(self, config, flow=False):
        super().__init__()
        self.config = config; self.flow = flow
        self.input_projection = nn.Linear(9, config.width)
        self.input_norm = nn.LayerNorm(config.width)
        self.token_embedding = nn.Embedding(6, config.width)
        self.side_embedding = nn.Embedding(2, config.width)
        self.readout_token = nn.Parameter(torch.randn(config.width) * .02)
        self.noise_projection = nn.Linear(20, config.width) if flow else None
        self.time_embedding = nn.Sequential(nn.Linear(256, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.layers)])
        self.output_norm = nn.LayerNorm(config.width)
        self.output_head = nn.Linear(config.width, 20)

    def forward(self, features, side, lengths, causal=False, noisy_q=None, flow_time=None):
        b, t = features.shape[:2]
        fingers = self.input_projection(features)
        readout = self.readout_token.view(1, 1, 1, -1).expand(b, t, 1, -1)
        if self.flow:
            if noisy_q is None or flow_time is None or noisy_q.shape != (b, t, 20) or flow_time.shape != (b,):
                raise ValueError('Flow requires noisy_q [B,T,20] and flow_time [B]')
            readout = readout + self.noise_projection(noisy_q).unsqueeze(-2)
        elif noisy_q is not None or flow_time is not None:
            raise ValueError('Regression models never accept ground-truth or feedback joint angles')
        x = torch.cat((fingers, readout), dim=-2)
        # Use Embedding's own backward, rather than an expanded raw parameter view.
        # This keeps leaf gradient storage independent of the broadcast readout gradient under DDP.
        token_ids = torch.arange(6, device=features.device)
        x = self.input_norm(x + self.token_embedding(token_ids)[None, None] + self.side_embedding(side)[:, None, None])
        x = x.reshape(b, t*6, -1)
        times = features.new_zeros(b) if flow_time is None else flow_time
        condition = self.time_embedding(timestep_embedding(times)) + self.side_embedding(side)
        plan = AttentionPlan(causal, lengths, t, features.device, self.config.backend)
        try:
            for block in self.blocks:
                if self.config.checkpoint_blocks and self.training and torch.is_grad_enabled():
                    x = checkpoint(block, x, condition, plan, use_reentrant=False)
                else:
                    x = block(x, condition, plan)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError('Whole-segment attention ran out of GPU memory. Reduce batch size, enable block '
                               'checkpointing, or provision more memory; the history was not truncated.') from exc
        query = x.reshape(b, t, 6, -1)[:, :, 5]
        out = self.output_head(self.output_norm(query))
        return out if self.flow else out.tanh()

    def backend_report(self):
        return sorted({name for block in self.blocks for name in block.attention.used_backends})
