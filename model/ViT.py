"""Vision Transformer (clean CLS-token variant).

Post-cleanup highlights:

* CLS-token only — no mean-pool branch.
* ``LayerNorm`` + ``GELU`` throughout (fused-kernel friendly under AMP).
* Stochastic depth actually wired up (``DropPath`` with a linear
  ``0 -> max_drop_path_rate`` schedule).
* Attention dropout enabled.
* Removed ``enhance`` / ``LEH`` / ``FeatureFusion`` / ``RotaryEmbedding`` /
  ``manifold_*`` dead code.
* ``ClassificationHead`` keeps the shallow+semantic concat (user-chosen
  design knob), but uses ``LayerNorm``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .utils import DropPath, FFN, ViTPatchEmbedding


class TransformerEncoderBlock(nn.Module):
    """Pre-norm attention block with stochastic depth on both residuals."""

    def __init__(self, emb_size, head_num, drop_p=0.1, attn_drop=0.0,
                 drop_path_rate=0.0, qk_norm=True):
        super().__init__()
        self.att_norm = nn.LayerNorm(emb_size)
        self.att = MultiHeadAttention(emb_size, head_num,
                                      attn_dropout=attn_drop, proj_dropout=drop_p,
                                      qk_norm=qk_norm)
        self.ffn_norm = nn.LayerNorm(emb_size)
        self.ffn = FFN(emb_size, expansion=4, drop_p=drop_p)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        x = x + self.drop_path(self.att(self.att_norm(x)))
        x = x + self.drop_path(self.ffn(self.ffn_norm(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of ``depth`` encoder blocks with a linear stochastic-depth schedule.

    ``shallow_features`` is the output of the 2nd block (index 1) — used by
    ``ClassificationHead`` to fuse with the final-layer feature.
    """

    def __init__(self, emb_size, head_num, depth, drop_p=0.1, attn_drop=0.0,
                 max_drop_path=0.1, qk_norm=True):
        super().__init__()
        self.depth = depth
        self.shallow_features = None
        # linear schedule: layer i gets i/(depth-1) * max_drop_path
        self.rates = [max_drop_path * x / max(1, depth - 1) for x in range(depth)]
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(emb_size, head_num,
                                    drop_p=drop_p,
                                    attn_drop=attn_drop,
                                    drop_path_rate=self.rates[i],
                                    qk_norm=qk_norm)
            for i in range(depth)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 1:
                self.shallow_features = x
        return x


class ClassificationHead(nn.Module):
    """Linear head over ``cat(shallow_cls, semantic_cls)`` — dim ``2 * emb_size``."""

    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.norm = nn.LayerNorm(2 * emb_size)
        self.classification = nn.Linear(2 * emb_size, n_classes)

    def forward(self, shallow, semantic):
        fused = torch.cat([shallow, semantic], dim=-1)
        return self.classification(self.norm(fused))


class VisionTransformer(nn.Module):
    """Vision Transformer with shallow+deep CLS-token fusion."""

    def __init__(self, channels, image_size, depth, head_num, emb_size, patch_size,
                 n_classes, drop_p=0.1, attn_drop=0.0, max_drop_path=0.1,
                 num_registers=0, qk_norm=True):
        super().__init__()
        self.patch_embedding = ViTPatchEmbedding(
            image_size=image_size, patch_size=patch_size,
            emb_size=emb_size, in_channels=channels, drop_p=drop_p,
            num_registers=num_registers,
        )
        self.vision_transformer = TransformerEncoder(
            emb_size=emb_size, head_num=head_num, depth=depth,
            drop_p=drop_p, attn_drop=attn_drop, max_drop_path=max_drop_path,
            qk_norm=qk_norm,
        )
        self.classification = ClassificationHead(emb_size, n_classes)

    def forward(self, x, classify=True):
        x = self.patch_embedding(x)
        x = self.vision_transformer(x)
        semantic = x[:, 0]
        shallow = self.vision_transformer.shallow_features[:, 0]
        output = self.classification(shallow, semantic)
        return output, None


__all__ = ['VisionTransformer', 'ClassificationHead', 'TransformerEncoder', 'TransformerEncoderBlock']
