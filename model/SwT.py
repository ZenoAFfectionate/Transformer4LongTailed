"""Swin Transformer — multi-stage shifted-window attention backbone.

Port of ``MoE4SwT/model/SwT.py`` into the MoE4ViT codebase, refactored to
match the ``VisionTransformer`` style (LayerNorm, ``(logits, None)`` return)
and with the original forward typos fixed (``self.Embedding`` / ``self.head``).

The classifier consumes ``cat(mean(shallow), mean(semantic))`` where shallow
comes from after ``merges[0]`` (dim ``2 * emb_size``) and semantic from the
final stage (dim ``8 * emb_size``) — total feature dim ``10 * emb_size``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import ShiftedWindowAttention
from .utils import DropPath, FFN, LEH, PatchMerging, SwTPatchEmbedding


class SwinEncoderBlock(nn.Module):
    """Swin Transformer block: optional LEH -> W-MSA/SW-MSA -> FFN.

    Uses stochastic depth (``DropPath``) on each residual branch — matches
    the standard Swin V1 recipe (linear schedule over blocks).
    """

    def __init__(self, emb_size, head_num, window_size, drop_p=0.1,
                 shift=False, enhance=False, drop_path_rate=0.0, qk_norm=True):
        super().__init__()
        self.enhance = enhance
        if self.enhance:
            self.leh_norm = nn.LayerNorm(emb_size)
            self.leh = LEH(emb_size, expansion=4, drop_p=drop_p)

        self.att_norm = nn.LayerNorm(emb_size)
        self.att = ShiftedWindowAttention(emb_size, head_num, window_size,
                                          shift=shift, qk_norm=qk_norm)

        self.ffn_norm = nn.LayerNorm(emb_size)
        self.ffn = FFN(emb_size, expansion=4, drop_p=drop_p)

        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        if self.enhance:
            x = x + self.drop_path(self.leh(self.leh_norm(x)))
        x = x + self.drop_path(self.att(self.att_norm(x)))
        x = x + self.drop_path(self.ffn(self.ffn_norm(x)))
        return x


class AlternatingEncoderBlock(nn.Module):
    """W-MSA block followed by SW-MSA block."""

    def __init__(self, emb_size, head_num, window_size=2, drop_p=0.1, enhance=False,
                 drop_path_rates=(0.0, 0.0), qk_norm=True):
        super().__init__()
        self.regular = SwinEncoderBlock(emb_size, head_num, window_size, drop_p,
                                        shift=False, enhance=enhance,
                                        drop_path_rate=drop_path_rates[0],
                                        qk_norm=qk_norm)
        self.shifted = SwinEncoderBlock(emb_size, head_num, window_size, drop_p,
                                        shift=True, enhance=enhance,
                                        drop_path_rate=drop_path_rates[1],
                                        qk_norm=qk_norm)

    def forward(self, x):
        x = self.regular(x)
        x = self.shifted(x)
        return x


class ClassificationHead(nn.Module):
    """Mean-pool shallow + semantic, concat, LayerNorm + Linear."""

    def __init__(self, feat_dim, n_classes):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.classification = nn.Linear(feat_dim, n_classes)

    def forward(self, shallow_features, semantic_features):
        shallow = shallow_features.mean(dim=1)
        semantic = semantic_features.mean(dim=1)
        features = torch.cat([shallow, semantic], dim=-1)
        return self.classification(self.norm(features))


class SwinTransformer(nn.Module):
    """Swin Transformer with 4 stages and shallow+semantic fusion.

    Feature dim for stage2: ``10 * emb_size`` (= 2 + 8 from shallow-after-merge1
    and semantic-after-stage4).
    """

    def __init__(self, channels=3, emb_size=96, depth=12, patch_size=4,
                 image_size=224, window_size=7, head_num=3, n_classes=1000,
                 drop_p=0.1, max_drop_path=0.1, qk_norm=True):
        super().__init__()
        assert depth >= 8, 'SwT depth must be >= 8 (needs at least 1 stage-3 block)'
        self.depth = depth
        self.Embedding = SwTPatchEmbedding(channels, emb_size, image_size, patch_size,
                                           drop_p=drop_p)

        # Stochastic-depth linear schedule: 0 -> max_drop_path over the
        # ``depth`` SwinEncoderBlocks (each AlternatingEncoderBlock holds 2).
        total_inner_blocks = depth  # depth // 2 AlternatingEncoderBlock * 2
        dpr = [max_drop_path * i / max(1, total_inner_blocks - 1)
               for i in range(total_inner_blocks)]

        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()

        idx = 0
        # Stage 1
        self.stages.append(AlternatingEncoderBlock(
            1 * emb_size, head_num, window_size, drop_p, enhance=True,
            drop_path_rates=(dpr[idx], dpr[idx + 1]), qk_norm=qk_norm))
        idx += 2
        self.merges.append(PatchMerging(1 * emb_size))

        # Stage 2
        self.stages.append(AlternatingEncoderBlock(
            2 * emb_size, 2 * head_num, window_size, drop_p, enhance=True,
            drop_path_rates=(dpr[idx], dpr[idx + 1]), qk_norm=qk_norm))
        idx += 2
        self.merges.append(PatchMerging(2 * emb_size))

        # Stage 3: (depth // 2 - 3) blocks
        for _ in range(depth // 2 - 3):
            self.stages.append(AlternatingEncoderBlock(
                4 * emb_size, 4 * head_num, window_size, drop_p,
                drop_path_rates=(dpr[idx], dpr[idx + 1]), qk_norm=qk_norm))
            idx += 2
        self.merges.append(PatchMerging(4 * emb_size))

        # Stage 4
        self.stages.append(AlternatingEncoderBlock(
            8 * emb_size, 8 * head_num, window_size, drop_p,
            drop_path_rates=(dpr[idx], dpr[idx + 1]), qk_norm=qk_norm))

        self.head = ClassificationHead(10 * emb_size, n_classes)

    def forward(self, x):
        x = self.Embedding(x)

        # Stage 1
        x = self.stages[0](x)
        x = self.merges[0](x)
        shallow_features = x

        # Stage 2
        x = self.stages[1](x)
        x = self.merges[1](x)

        # Stage 3: indices 2 .. depth//2-2
        for i in range(2, self.depth // 2 - 1):
            x = self.stages[i](x)
        x = self.merges[2](x)

        # Stage 4
        x = self.stages[-1](x)
        semantic_features = x

        logits = self.head(shallow_features, semantic_features)
        return logits, None

    def get_auxiliary_loss(self):
        return torch.tensor(0.0)


__all__ = ['SwinTransformer', 'ClassificationHead', 'AlternatingEncoderBlock',
           'SwinEncoderBlock']
