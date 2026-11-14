from dataclasses import asdict, dataclass

VARIANTS = ('temporal_causal', 'temporal_bidirectional', 'frame_regression', 'frame_flow')


@dataclass(frozen=True)
class ModelConfig:
    width: int = 1024
    layers: int = 4
    heads: int = 16
    ffn_width: int = 4096
    backend: str = 'auto'
    checkpoint_blocks: bool = False

    def __post_init__(self):
        if min(self.width, self.layers, self.heads, self.ffn_width) < 1:
            raise ValueError('Model dimensions must be positive')
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError('RoPE requires an even integer head dimension')
        if self.backend not in ('auto', 'sdpa', 'flash', 'flex'):
            raise ValueError('Unknown attention backend')

    def to_dict(self):
        return asdict(self)
