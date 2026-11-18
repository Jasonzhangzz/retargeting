"""Causal pose encoder, temporary reconstruction head, and HACO contact head."""
from pathlib import Path
import torch
from torch import nn
from .rotations import pose_features


class PoseEncoder(nn.Module):
    def __init__(self, width=256, layers=4, heads=8, latent_dim=64):
        super().__init__()
        self.config = dict(width=width, layers=layers, heads=heads, latent_dim=latent_dim)
        self.embed = nn.Sequential(nn.Linear(99, width), nn.GELU(), nn.Linear(width, width))
        self.position = nn.Parameter(torch.randn(1, 8, width) * .02)
        layer = nn.TransformerEncoderLayer(width, heads, 1024, dropout=.0, batch_first=True,
                                           norm_first=True, activation='gelu')
        self.temporal = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.bottleneck = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, latent_dim))

    def forward_features(self, features):
        t = features.shape[1]
        mask = torch.ones(t, t, device=features.device, dtype=torch.bool).triu(1)
        return self.bottleneck(self.temporal(self.embed(features) + self.position[:, :t], mask=mask))

    def forward(self, pose51):
        return self.forward_features(pose_features(pose51))


class PoseAutoencoder(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else PoseEncoder()
        d = self.encoder.config['latent_dim']
        self.reconstruction = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 99))
        self.projection = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 128))
        self.frame_head = nn.Linear(d, 1)

    def forward(self, pose51):
        z = self.encoder(pose51)
        return dict(reconstruction=self.reconstruction(z), latent=z,
                    projection=self.projection(z[:, -1]), frame_contact_logit=self.frame_head(z[:, -1]))


class ContactModel(nn.Module):
    def __init__(self, assets, encoder=None):
        super().__init__()
        from .haco import ContactTransformerDecoderHead
        self.encoder = encoder if encoder is not None else PoseEncoder()
        d = self.encoder.config['latent_dim']
        self.context = nn.Linear(d, 1280)
        self.decoder = ContactTransformerDecoderHead(Path(assets))
        for name,rows in [('V_regressor_336',336),('V_regressor_84',84),('J_regressor',21)]:
            value=getattr(self.decoder,name)
            if value.shape!=(rows,778) or not torch.isfinite(value).all():
                raise ValueError('Invalid HACO asset '+name)
        self.frame_head = nn.Linear(d, 1)

    @classmethod
    def from_autoencoder(cls, autoencoder, assets):
        import copy
        model = cls(assets, copy.deepcopy(autoencoder.encoder))
        model.frame_head.load_state_dict(autoencoder.frame_head.state_dict())
        return model

    def forward(self, pose51):
        latent = self.encoder(pose51)
        # HACO's original spatial-token interface accepts arbitrary H*W.
        context = self.context(latent).transpose(1, 2).unsqueeze(-1)
        values = self.decoder(context)
        names = ('contact_out', 'contact_336_out', 'contact_84_out', 'contact_joint_out')
        return dict(zip(names, values), frame_contact_logit=self.frame_head(latent[:, -1]))


class ModelEMA:
    def __init__(self, model, decay=.999):
        import copy
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0,1)')
        self.module = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        for target, value in zip(self.module.parameters(), model.parameters(), strict=True):
            target.lerp_(value.detach(), 1-self.decay)
        for target, value in zip(self.module.buffers(), model.buffers(), strict=True):
            target.copy_(value)
        self.updates += 1
