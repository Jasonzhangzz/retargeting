"""HACO decoder, adapted only for explicit assets and registered buffers.
Upstream: dqj5182/HACO_RELEASE c774211b968bdf836f9511dbd5ff23bb19fe1c44.
License: CC BY-NC 4.0; see third_party/HACO_CONTACT_LICENSE.txt.
"""
import torch
from torch import nn
import numpy as np
from einops import rearrange
from inspect import isfunction
from typing import Callable, Optional

def exists(val):
    return val is not None


# This function is from HaMeR (https://github.com/geopavlakos/hamer).
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# This class is from HaMeR (https://github.com/geopavlakos/hamer).
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# This class is from HaMeR (https://github.com/geopavlakos/hamer).
class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        context_dim = default(context_dim, dim)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, context=None):
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# This class is from HaMeR (https://github.com/geopavlakos/hamer).
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# This class is from HaMeR (https://github.com/geopavlakos/hamer).
class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, *args):
        for attn, ff in self.layers:
            x = attn(x, *args) + x
            x = ff(x, *args) + x
        return x


class AdaptiveLayerNorm1D(torch.nn.Module):
    def __init__(self, data_dim: int, norm_cond_dim: int):
        super().__init__()
        if data_dim <= 0:
            raise ValueError(f"data_dim must be positive, but got {data_dim}")
        if norm_cond_dim <= 0:
            raise ValueError(f"norm_cond_dim must be positive, but got {norm_cond_dim}")
        self.norm = torch.nn.LayerNorm(
            data_dim
        )  # TODO: Check if elementwise_affine=True is correct
        self.linear = torch.nn.Linear(norm_cond_dim, 2 * data_dim)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (batch, ..., data_dim)
        # t: (batch, norm_cond_dim)
        # return: (batch, data_dim)
        x = self.norm(x)
        alpha, beta = self.linear(t).chunk(2, dim=-1)

        # Add singleton dimensions to alpha and beta
        if x.dim() > 2:
            alpha = alpha.view(alpha.shape[0], *([1] * (x.dim() - 2)), alpha.shape[1])
            beta = beta.view(beta.shape[0], *([1] * (x.dim() - 2)), beta.shape[1])

        return x * (1 + alpha) + beta


def normalization_layer(norm: Optional[str], dim: int, norm_cond_dim: int = -1):
    if norm == "batch":
        return torch.nn.BatchNorm1d(dim)
    elif norm == "layer":
        return torch.nn.LayerNorm(dim)
    elif norm == "ada":
        assert norm_cond_dim > 0, f"norm_cond_dim must be positive, got {norm_cond_dim}"
        return AdaptiveLayerNorm1D(dim, norm_cond_dim)
    elif norm is None:
        return torch.nn.Identity()
    else:
        raise ValueError(f"Unknown norm: {norm}")


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable, norm: str = "layer", norm_cond_dim: int = -1):
        super().__init__()
        self.norm = normalization_layer(norm, dim, norm_cond_dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if isinstance(self.norm, AdaptiveLayerNorm1D):
            return self.fn(self.norm(x, *args), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


# This class is from HaMeR (https://github.com/geopavlakos/hamer).
class TransformerCrossAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca = CrossAttention(
                dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout
            )
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, *args, context=None, context_list=None):
        if context_list is None:
            context_list = [context] * len(self.layers)
        if len(context_list) != len(self.layers):
            raise ValueError(f"len(context_list) != len(self.layers) ({len(context_list)} != {len(self.layers)})")

        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            x = self_attn(x, *args) + x
            x = cross_attn(x, *args, context=context_list[i]) + x
            x = ff(x, *args) + x
        return x


# This class is from HaMeR (https://github.com/geopavlakos/hamer).
class DropTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, dim)
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[0, :, 0], self.p).bernoulli().bool()
            # TODO: permutation idx for each batch using torch.argsort
            if zero_mask.any():
                x = x[:, ~zero_mask, :]
        return x


# This class is from HaMeR (https://github.com/geopavlakos/hamer).
class ZeroTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, dim)
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[:, :, 0], self.p).bernoulli().bool()
            # Zero-out the masked tokens
            x[zero_mask, :] = 0
        return x


# This class is from HaMeR (https://github.com/geopavlakos/hamer).
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = 'drop',
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
        skip_token_embedding: bool = False,
    ):
        super().__init__()
        if not skip_token_embedding:
            self.to_token_embedding = nn.Linear(token_dim, dim)
        else:
            self.to_token_embedding = nn.Identity()
            if token_dim != dim:
                raise ValueError(
                    f"token_dim ({token_dim}) != dim ({dim}) when skip_token_embedding is True"
                )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)
        elif emb_dropout_type == "normal":
            self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerCrossAttn(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            norm=norm,
            norm_cond_dim=norm_cond_dim,
            context_dim=context_dim,
        )

    def forward(self, inp: torch.Tensor, *args, context=None, context_list=None):
        x = self.to_token_embedding(inp)
        b, n, _ = x.shape

        x = self.dropout(x)
        x += self.pos_embedding[:, :n]

        x = self.transformer(x, *args, context=context, context_list=context_list)
        return x


class ContactTransformerDecoderHead(nn.Module):
    """ Cross-attention based MANO Transformer decoder
    """
    def __init__(self, assets):
        super().__init__()
        transformer_args = dict(
            num_tokens=1,
            token_dim=1,
            dim=1024,
        )
        context_dim = 1280
        MANO_HEAD_TRANSFORMER_DECODER_CONFIG = {'depth': 6, 'heads': 8, 'mlp_dim': 1024, 'dim_head': 64, 'dropout': 0.0, 'emb_dropout': 0.0, 'norm': 'layer', 'context_dim': context_dim}
        transformer_args.update(dict(MANO_HEAD_TRANSFORMER_DECODER_CONFIG))
        self.transformer = TransformerDecoder(
            **transformer_args
        )
        self.deccontact = nn.Linear(1024, 778)
        self.init_contact = nn.Parameter(torch.randn(1, 778, requires_grad=True))
        for name in ('V_regressor_336', 'V_regressor_84', 'J_regressor'):
            self.register_buffer(name, torch.from_numpy(np.load(assets / (name + '.npy'))).float())

    def forward(self, x, **kwargs): # x: [b, 1280, 16, 12] (if resnet-50, x: [b, 2048, 8, 8], resnet-34: [b, 512, 8, 8], hrnet-w32: [b, 2048, 8, 8])
        batch_size = x.shape[0]
        device = x.device

        # vit pretrained backbone is channel-first. Change to token-first
        x = rearrange(x, 'b c h w -> b (h w) c')

        init_contact = self.init_contact.expand(batch_size, -1)
        pred_contact = init_contact

        token = torch.zeros(batch_size, 1, 1).to(x.device)

        # Pass through transformer
        token_out = self.transformer(token, context=x) # x: [b, 192, 1280]
        token_out = token_out[:, 0] # (B, C)

        # Readout from token_out
        pred_contact = self.deccontact(token_out) + pred_contact

        # Joint contact
        pred_joint_contact = (self.J_regressor @ pred_contact.T).T
        pred_mesh_contact_336 = (self.V_regressor_336 @ pred_contact.T).T
        pred_mesh_contact_84 = (self.V_regressor_84 @ pred_contact.T).T

        return pred_contact, pred_mesh_contact_336, pred_mesh_contact_84, pred_joint_contact