"""Shared building blocks for ViT / MoE4ViT as well as SwT / MoE4SwT"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.init as init

from einops import rearrange
from einops.layers.torch import Rearrange


class ViTPatchEmbedding(nn.Module):
    """Conv-patch tokenizer with CLS token, optional register tokens, and
    learned absolute positions.

    Register tokens (Darcet et al. 2024, DINOv2-reg) are learnable sink
    slots that absorb attention artifacts without polluting the CLS/patch
    representations. Token layout is ``[CLS, registers..., patches...]``;
    the classifier still reads ``x[:, 0]`` so downstream code is unchanged.
    """

    def __init__(self, image_size, patch_size, emb_size, in_channels, drop_p=0.1,
                 num_registers=0):
        super().__init__()
        self.num_registers = int(num_registers)
        self.num_prefix = 1 + self.num_registers  # CLS + registers

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        init.trunc_normal_(self.cls_token, std=0.02)

        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, self.num_registers, emb_size)
            )
            init.trunc_normal_(self.register_tokens, std=0.02)

        num_patches = (image_size // patch_size) ** 2
        self.positions = nn.Parameter(
            torch.zeros(1, self.num_prefix + num_patches, emb_size)
        )
        init.trunc_normal_(self.positions, std=0.02)

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e'),
            nn.LayerNorm(emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        x = self.projection(x)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        if self.num_registers > 0:
            regs = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([cls, regs, x], dim=1)
        else:
            x = torch.cat([cls, x], dim=1)
        return x + self.positions


class FFN(nn.Module):
    """Transformer feed-forward block: ``Linear → GELU → Dropout → Linear → Dropout``."""

    def __init__(self, emb_size, expansion=4, drop_p=0.1):
        super().__init__()
        inner_dim = emb_size * expansion
        self.net = nn.Sequential(
            nn.Linear(emb_size, inner_dim),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(inner_dim, emb_size),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        return self.net(x)


# Alias used by MoE4ViT's SparseMoE / ShareExpertMoE expert lists.
MLP = FFN


class SwTPatchEmbedding(nn.Module):
    """Conv-patch tokenizer for Swin — no CLS token, learned positions."""

    def __init__(self, channels, emb_size, image_size, patch_size, drop_p=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(drop_p)

        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, emb_size))
        init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        x = x + self.pos_embed
        return self.dropout(x)


class PatchMerging(nn.Module):
    """Swin patch merging — (B, H*W, C) -> (B, H/2 * W/2, 2*C).

    Standard Swin ordering: reshape + concat to ``4*C``, LayerNorm on ``4*C``,
    then Linear reducing to ``2*C``. (The original MoE4SwT code flipped norm
    and linear, which produced a shape mismatch.)
    """

    def __init__(self, emb_size):
        super().__init__()
        self.norm = nn.LayerNorm(4 * emb_size)
        self.linear = nn.Linear(4 * emb_size, 2 * emb_size, bias=False)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        assert H * W == L, 'PatchMerging expects a square token grid'

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b (h s1) (w s2) c -> b (h w) (s1 s2 c)', s1=2, s2=2)
        x = self.norm(x)
        return self.linear(x)


class LEH(nn.Module):
    """Local Feature Enhancement — 1x1 -> dw-3x3 -> 1x1 conv stack."""

    def __init__(self, emb_size, expansion=4, drop_p=0.1):
        super().__init__()
        inner_dim = emb_size * expansion
        self.net = nn.Sequential(
            nn.Conv2d(emb_size, inner_dim, 1),
            nn.GELU(),
            nn.Conv2d(inner_dim, inner_dim, 3, padding=1, groups=inner_dim),
            nn.GELU(),
            nn.Conv2d(inner_dim, emb_size, 1),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        assert H * W == L, 'LEH expects a square token grid'
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.net(x)
        return rearrange(x, 'b c h w -> b (h w) c')


class DropPath(nn.Module):
    """Stochastic depth — drops entire residual branches at random during training.

    ``drop_prob`` is the probability of dropping a residual branch; identity
    when ``drop_prob == 0`` or the module is in ``eval()`` mode.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


__all__ = ['ViTPatchEmbedding', 'SwTPatchEmbedding', 'FFN', 'MLP', 'DropPath',
           'PatchMerging', 'LEH']
