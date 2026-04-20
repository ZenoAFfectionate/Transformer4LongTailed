"""MoE4ViT — ViT backbone with shared+routed MoE in most blocks.

Uses the CLS-token patch embedding + standard multi-head self-attention,
with one vanilla ``MLP`` block at index 0 followed by ``depth - 2`` MoE
blocks. The MoE classes live in :mod:`model.moe` so they can be shared
with :class:`model.MoE4SwT.MoE4SwT`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .moe import TransformerEncoderBlock_MoE
from .utils import MLP, ViTPatchEmbedding


class TransformerEncoderBlock_MLP(nn.Module):
    """Standard pre-norm Transformer block: attention + MLP (no MoE)."""

    def __init__(self, emb_size, head_num, expansion=4, drop_p=0.1, attn_drop=0.0,
                 qk_norm=True):
        super().__init__()
        self.mlp_norm = nn.LayerNorm(emb_size)
        self.att_norm = nn.LayerNorm(emb_size)

        self.mlp = MLP(emb_size, expansion, drop_p)
        self.attention = MultiHeadAttention(emb_size, head_num,
                                            attn_dropout=attn_drop, proj_dropout=drop_p,
                                            qk_norm=qk_norm)

    def forward(self, x):
        residual = x
        x = self.att_norm(x)
        x = self.attention(x)
        x = x + residual

        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + residual
        return x

    def get_expert_loads(self):
        return 0

    def get_auxiliary_loss(self):
        return 0


class TransformerEncoder(nn.Module):
    """Block 0: plain MLP; blocks 1..depth-2: MoE."""

    def __init__(self, emb_size, head_num, expert_num, share_experts, top_k, depth=12,
                 drop_p=0.1, attn_drop=0.0, qk_norm=True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock_MLP(emb_size, head_num, drop_p=drop_p,
                                         attn_drop=attn_drop, qk_norm=qk_norm)]
            + [TransformerEncoderBlock_MoE(emb_size, head_num, expert_num, share_experts, top_k,
                                           drop_p=drop_p, attn_drop=attn_drop,
                                           qk_norm=qk_norm)
               for _ in range(depth - 2)]
        )

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            if i == 0:
                self.shallow_features = x[:, 0]
            x = block(x)
        return x

    def get_expert_load(self):
        return [block.get_expert_loads() for block in self.blocks]

    def get_auxiliary_loss(self):
        return sum(block.get_auxiliary_loss() for block in self.blocks)

    def update_expert_bias(self):
        for block in self.blocks:
            if hasattr(block, 'update_expert_bias'):
                block.update_expert_bias()


class ClassificationHead(nn.Module):
    """Concatenate shallow + semantic CLS tokens, LayerNorm, Linear."""

    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.layer_norm = nn.LayerNorm(2 * emb_size)
        self.classification = nn.Linear(2 * emb_size, n_classes)

    def forward(self, shallow_features, semantic_features):
        cls_token = torch.cat([shallow_features, semantic_features], dim=-1)
        return self.classification(self.layer_norm(cls_token))


class MoE4ViT(nn.Module):
    """Mixture-of-Experts Vision Transformer."""

    def __init__(self, in_channels, image_size, patch_size, emb_size, depth, n_classes,
                 head_num, expert_num, share_experts, top_k,
                 drop_p=0.1, attn_drop=0.0, num_registers=0, qk_norm=True):
        super().__init__()
        self.patch_embedding = ViTPatchEmbedding(image_size, patch_size, emb_size, in_channels,
                                                 drop_p=drop_p, num_registers=num_registers)
        self.vision_transformer = TransformerEncoder(
            emb_size, head_num, expert_num, share_experts, top_k, depth,
            drop_p=drop_p, attn_drop=attn_drop, qk_norm=qk_norm,
        )
        self.classification = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.vision_transformer(x)
        shallow_features = self.vision_transformer.shallow_features
        semantic_features = x[:, 0]
        x = self.classification(shallow_features, semantic_features)
        return x

    def get_expert_load(self):
        return self.vision_transformer.get_expert_load()

    def get_auxiliary_loss(self):
        return self.vision_transformer.get_auxiliary_loss()

    def update_expert_bias(self):
        self.vision_transformer.update_expert_bias()


__all__ = ['MoE4ViT', 'ClassificationHead', 'TransformerEncoder',
           'TransformerEncoderBlock_MLP']
