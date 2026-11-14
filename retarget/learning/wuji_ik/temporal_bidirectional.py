"""Offline full-segment attention; permits future EMF context."""
from torch import nn
from .backbone import IKBackbone


class TemporalBidirectional(nn.Module):
    def __init__(self, config):
        super().__init__(); self.core = IKBackbone(config)

    def forward(self, features, side, lengths):
        return self.core(features, side, lengths, causal=False)
