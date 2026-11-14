import torch
from torch import nn
from .backbone import IKBackbone
from .config import ModelConfig, VARIANTS
from .temporal_causal import TemporalCausal
from .temporal_bidirectional import TemporalBidirectional


class FrameModel(nn.Module):
    def __init__(self, config, flow=False):
        super().__init__(); self.core = IKBackbone(config, flow=flow)

    def forward(self, features, side, lengths, noisy_q=None, flow_time=None):
        b, t = features.shape[:2]
        # Independent six-token attention per frame, including inside trajectories.
        independent = features.reshape(b*t, 1, 5, 9)
        sides = side[:, None].expand(b, t).reshape(-1)
        ones = torch.ones(b*t, device=features.device, dtype=torch.long)
        noise = noisy_q.reshape(b*t, 1, 20) if noisy_q is not None else None
        times = flow_time.reshape(b*t) if flow_time is not None else None
        return self.core(independent, sides, ones, noisy_q=noise, flow_time=times).reshape(b, t, 20)


def build_model(variant, config=None):
    config = config if config is not None else ModelConfig()
    if variant == 'temporal_causal': return TemporalCausal(config)
    if variant == 'temporal_bidirectional': return TemporalBidirectional(config)
    if variant in ('frame_regression', 'frame_flow'): return FrameModel(config, flow=variant == 'frame_flow')
    raise ValueError(f'variant must be one of {VARIANTS}')


def flow_training_pair(target, generator=None):
    z = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
    t = torch.rand(target.shape[:2], device=target.device, dtype=torch.float32, generator=generator)
    return (1-t[..., None])*z + t[..., None]*target, t, target-z


@torch.no_grad()
def sample_flow(model, features, side, lengths, steps=8, seed=42, shared_noise=True, initial_noise=None):
    if steps < 1:
        raise ValueError('Flow integration needs at least one step')
    b, t = features.shape[:2]
    if initial_noise is None:
        generator = torch.Generator(device=features.device).manual_seed(seed)
        shape = (b, 1 if shared_noise else t, 20)
        x = torch.randn(shape, device=features.device, dtype=features.dtype, generator=generator).expand(b, t, 20).clone()
    else:
        x = initial_noise.clone()
        if x.shape != (b, t, 20): raise ValueError('initial_noise shape mismatch')
    for i in range(steps):
        times = torch.full((b, t), i/steps, device=features.device, dtype=torch.float32)
        x = x + model(features, side, lengths, noisy_q=x, flow_time=times).float()/steps
    return x.clamp(-1, 1)
