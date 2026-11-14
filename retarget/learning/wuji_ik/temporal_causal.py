"""Full-segment frame-causal model: all output frames are computed in parallel."""
from torch import nn
from .backbone import IKBackbone


class TemporalCausal(nn.Module):
    def __init__(self, config):
        super().__init__(); self.core = IKBackbone(config)

    def forward(self, features, side, lengths):
        return self.core(features, side, lengths, causal=True)
