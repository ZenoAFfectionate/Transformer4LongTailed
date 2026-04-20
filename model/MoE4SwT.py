"""MoE4SwT — MoE4ViT with LEH local-enhancement in the first block.

Structure mirrors ``model/MoE4ViT.py`` (CLS-token ViT + sparse shared-expert
MoE), with one difference: the first encoder block uses ``LEH`` over the
patch tokens for early local feature enhancement, then attention, then FFN.

Reuses ``TransformerEncoderBlock_MoE`` from :mod:`model.moe` so MoE code
lives in exactly one place — the two backbones only differ in their
first-block design.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .moe import TransformerEncoderBlock_MoE
from .utils import FFN, LEH, ViTPatchEmbedding


class TransformerEncoderBlock_LEH(nn.Module):
    """LEH (on patch tokens) -> attention -> FFN, with residuals and dropout.

    LEH requires a square spatial grid, so the non-spatial prefix tokens
    (CLS + registers) are split off before LEH and re-prepended afterwards.
    """

    def __init__(self, emb_size, head_num, expansion=4, drop_p=0.1, attn_drop=0.0,
                 num_prefix=1, qk_norm=True):
        super().__init__()
        self.num_prefix = int(num_prefix)
        self.leh_norm = nn.LayerNorm(emb_size)
        self.att_norm = nn.LayerNorm(emb_size)
        self.ffn_norm = nn.LayerNorm(emb_size)

        self.leh_drop = nn.Dropout(drop_p)
        self.att_drop = nn.Dropout(drop_p)
        self.ffn_drop = nn.Dropout(drop_p)

        self.leh = LEH(emb_size, expansion, drop_p)
        self.att = MultiHeadAttention(emb_size, head_num,
                                      attn_dropout=attn_drop, proj_dropout=drop_p,
                                      qk_norm=qk_norm)
        self.ffn = FFN(emb_size, expansion, drop_p)

    def forward(self, x):
        prefix, patches = x[:, :self.num_prefix], x[:, self.num_prefix:]
        residual = patches
        patches = self.leh_norm(patches)
        patches = self.leh(patches)
        patches = self.leh_drop(patches) + residual
        x = torch.cat([prefix, patches], dim=1)

        residual = x
        x = self.att_norm(x)
        x = self.att(x)
        x = self.att_drop(x) + residual

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.ffn_drop(x) + residual
        return x

    def get_expert_loads(self):
        return 0

    def get_auxiliary_loss(self):
        return 0


class TransformerEncoder(nn.Module):
    """Stack: one LEH block + (depth - 2) MoE blocks."""

    def __init__(self, emb_size, head_num, expert_num, share_experts, top_k, depth=12,
                 drop_p=0.1, attn_drop=0.0, num_prefix=1, qk_norm=True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock_LEH(emb_size, head_num,
                                         drop_p=drop_p, attn_drop=attn_drop,
                                         num_prefix=num_prefix, qk_norm=qk_norm)]
            + [TransformerEncoderBlock_MoE(emb_size, head_num, expert_num, share_experts, top_k,
                                           drop_p=drop_p, attn_drop=attn_drop,
                                           qk_norm=qk_norm)
               for _ in range(depth - 2)]
        )
        self.shallow_features = None

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
    """Concatenated (shallow_cls, semantic_cls) + LayerNorm + Linear."""

    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.layer_norm = nn.LayerNorm(2 * emb_size)
        self.classification = nn.Linear(2 * emb_size, n_classes)

    def forward(self, shallow_features, semantic_features):
        cls_token = torch.cat([shallow_features, semantic_features], dim=-1)
        return self.classification(self.layer_norm(cls_token))


class MoE4SwT(nn.Module):
    """ViT-style CLS-token backbone + LEH first block + MoE blocks."""

    def __init__(self, in_channels, image_size, patch_size, emb_size, depth, n_classes,
                 head_num, expert_num, share_experts, top_k,
                 drop_p=0.1, attn_drop=0.0, num_registers=0, qk_norm=True):
        super().__init__()
        self.patch_embedding = ViTPatchEmbedding(image_size, patch_size, emb_size, in_channels,
                                                 drop_p=drop_p, num_registers=num_registers)
        self.vision_transformer = TransformerEncoder(
            emb_size, head_num, expert_num, share_experts, top_k, depth,
            drop_p=drop_p, attn_drop=attn_drop,
            num_prefix=1 + int(num_registers), qk_norm=qk_norm,
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


__all__ = ['MoE4SwT', 'ClassificationHead', 'TransformerEncoder',
           'TransformerEncoderBlock_LEH']
