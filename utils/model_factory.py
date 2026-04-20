"""Model, loss, classifier, and feature-extraction factories for MoE4ViT.

Keeps ``train_stage1.py`` / ``train_stage2.py`` free of conditional clutter —
all dispatches on ``cfg.model_name``, ``cfg.loss_type``, and
``cfg.classifier`` live here.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from classifier.linear import LinearClassifier, LWSHead, LWSPlusHead
from model.ViT import VisionTransformer
from .loss import (
    BalancedSoftmaxLoss,
    ClassBalancedLoss,
    FocalLoss,
    LabelAwareSmoothing,
    LDAMLoss,
    LogitAdjustmentLoss,
    WeightedCrossEntropyLoss,
)


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

def build_model(cfg) -> nn.Module:
    """Dispatch on ``cfg.model_name``; returns an ``nn.Module``.

    ``VisionTransformer`` comes from ``model/ViT.py``; ``MoE4ViT`` is
    imported lazily so users who never request it do not pay the cost (and
    would not surface any latent issues inside that module).
    """
    if cfg.model_name == 'ViT':
        return VisionTransformer(
            channels=cfg.channels,
            image_size=cfg.image_size,
            depth=cfg.depth,
            head_num=cfg.head_num,
            emb_size=cfg.emb_size,
            patch_size=cfg.patch_size,
            n_classes=cfg.num_classes,
            drop_p=float(getattr(cfg, 'drop_p', 0.1)),
            attn_drop=float(getattr(cfg, 'attn_drop', 0.0)),
            max_drop_path=float(getattr(cfg, 'max_drop_path', 0.1)),
            num_registers=int(getattr(cfg, 'num_registers', 0)),
            qk_norm=bool(getattr(cfg, 'qk_norm', True)),
        )
    if cfg.model_name == 'MoE4ViT':
        from model.MoE4ViT import MoE4ViT  # lazy import
        return MoE4ViT(
            in_channels=cfg.channels,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            emb_size=cfg.emb_size,
            depth=cfg.depth,
            n_classes=cfg.num_classes,
            head_num=cfg.head_num,
            expert_num=cfg.num_experts,
            share_experts=cfg.share_experts,
            top_k=cfg.top_k,
            drop_p=float(getattr(cfg, 'drop_p', 0.1)),
            attn_drop=float(getattr(cfg, 'attn_drop', 0.0)),
            num_registers=int(getattr(cfg, 'num_registers', 0)),
            qk_norm=bool(getattr(cfg, 'qk_norm', True)),
        )
    if cfg.model_name == 'SwT':
        from model.SwT import SwinTransformer  # lazy import
        return SwinTransformer(
            channels=cfg.channels,
            image_size=cfg.image_size,
            depth=cfg.depth,
            patch_size=cfg.patch_size,
            window_size=int(getattr(cfg, 'window_size', 7)),
            head_num=cfg.head_num,
            emb_size=cfg.emb_size,
            n_classes=cfg.num_classes,
            drop_p=float(getattr(cfg, 'drop_p', 0.1)),
            max_drop_path=float(getattr(cfg, 'max_drop_path', 0.1)),
            qk_norm=bool(getattr(cfg, 'qk_norm', True)),
        )
    if cfg.model_name == 'MoE4SwT':
        from model.MoE4SwT import MoE4SwT  # lazy import
        return MoE4SwT(
            in_channels=cfg.channels,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            emb_size=cfg.emb_size,
            depth=cfg.depth,
            n_classes=cfg.num_classes,
            head_num=cfg.head_num,
            expert_num=cfg.num_experts,
            share_experts=cfg.share_experts,
            top_k=cfg.top_k,
            drop_p=float(getattr(cfg, 'drop_p', 0.1)),
            attn_drop=float(getattr(cfg, 'attn_drop', 0.0)),
            num_registers=int(getattr(cfg, 'num_registers', 0)),
            qk_norm=bool(getattr(cfg, 'qk_norm', True)),
        )
    raise ValueError(f'Unknown cfg.model_name: {cfg.model_name!r}')


def model_feature_dim(cfg) -> int:
    """Feature dim produced by :func:`extract_features`.

    * ViT / MoE4ViT / MoE4SwT — CLS-token-based, dim = ``2 * emb_size``.
    * SwT — mean-pooled multi-scale fusion, dim = ``10 * emb_size``
      (``2 * emb_size`` shallow after merge1 + ``8 * emb_size`` semantic
      after stage4).
    """
    if cfg.model_name == 'SwT':
        return 10 * int(cfg.emb_size)
    return 2 * int(cfg.emb_size)


def forward_logits(model, x):
    """Normalize model outputs — ViT returns ``(logits, None)``; MoE4ViT a tensor."""
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def extract_features(model, x):
    """Run the backbone forward and return the pooled pre-classifier feature.

    * ViT / MoE4ViT — ``cat(shallow_cls, semantic_cls)``, dim ``2 * emb_size``.
    * SwT — ``cat(mean(shallow), mean(semantic))``, dim ``10 * emb_size``.
    * MoE4SwT — same as MoE4ViT; shallow comes from the first-block CLS.

    Unwraps ``DataParallel`` / ``DistributedDataParallel`` so callers can pass
    the wrapped model straight through.
    """
    target = model.module if hasattr(model, 'module') else model

    if isinstance(target, VisionTransformer):
        z = target.patch_embedding(x)
        z = target.vision_transformer(z)
        semantic = z[:, 0]
        shallow = target.vision_transformer.shallow_features[:, 0]
        return torch.cat([shallow, semantic], dim=-1)

    try:
        from model.MoE4ViT import MoE4ViT
    except ImportError:
        MoE4ViT = None  # type: ignore[assignment]
    if MoE4ViT is not None and isinstance(target, MoE4ViT):
        z = target.patch_embedding(x)
        z = target.vision_transformer(z)
        semantic = z[:, 0]
        shallow = target.vision_transformer.shallow_features
        return torch.cat([shallow, semantic], dim=-1)

    try:
        from model.MoE4SwT import MoE4SwT
    except ImportError:
        MoE4SwT = None  # type: ignore[assignment]
    if MoE4SwT is not None and isinstance(target, MoE4SwT):
        z = target.patch_embedding(x)
        z = target.vision_transformer(z)
        semantic = z[:, 0]
        shallow = target.vision_transformer.shallow_features
        return torch.cat([shallow, semantic], dim=-1)

    try:
        from model.SwT import SwinTransformer
    except ImportError:
        SwinTransformer = None  # type: ignore[assignment]
    if SwinTransformer is not None and isinstance(target, SwinTransformer):
        z = target.Embedding(x)
        z = target.stages[0](z)
        z = target.merges[0](z)
        shallow = z.mean(dim=1)

        z = target.stages[1](z)
        z = target.merges[1](z)
        for i in range(2, target.depth // 2 - 1):
            z = target.stages[i](z)
        z = target.merges[2](z)

        z = target.stages[-1](z)
        semantic = z.mean(dim=1)

        return torch.cat([shallow, semantic], dim=-1)

    raise TypeError(f'extract_features: unsupported model type {type(target).__name__}')


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------

def build_loss(cfg, cls_num_list: Sequence[int], device) -> nn.Module:
    """Dispatch on ``cfg.loss_type``; returns an ``nn.Module`` on ``device``.

    Losses that need per-class priors (LDAM/BalancedSoftmax/LogitAdjustment/
    CB/LAS) receive ``cls_num_list``. Plain CE / Focal ignore it.
    """
    kind = str(cfg.loss_type).upper()
    smoothing = float(getattr(cfg, 'label_smoothing', 0.0))
    if kind == 'CE':
        crit: nn.Module = WeightedCrossEntropyLoss(reduction='mean',
                                                   label_smoothing=smoothing)
    elif kind == 'LDAM':
        crit = LDAMLoss(cls_num_list, max_m=float(cfg.max_m), s=30.0)
    elif kind in ('BALANCEDSOFTMAX', 'BS'):
        crit = BalancedSoftmaxLoss(cls_num_list)
    elif kind in ('LOGITADJUSTMENT', 'LA'):
        crit = LogitAdjustmentLoss(cls_num_list, tau=float(cfg.tau))
    elif kind == 'FOCAL':
        crit = FocalLoss(gamma=2.0)
    elif kind == 'LAS':
        crit = LabelAwareSmoothing(cls_num_list, float(cfg.smooth_head), float(cfg.smooth_tail))
    elif kind == 'CB':
        crit = ClassBalancedLoss(cls_num_list)
    else:
        raise ValueError(f'Unknown cfg.loss_type: {cfg.loss_type!r}')
    return crit.to(device)


# ------------------------------------------------------------------
# Stage2 components
# ------------------------------------------------------------------

def build_stage1_components_vit(cfg) -> Tuple[nn.Module, nn.Module, int]:
    """Return ``(backbone, classifier, feat_in)`` for stage2 retraining.

    ``classifier`` is always a fresh :class:`LinearClassifier` sized to the
    pooled feature dim. For LWS / LWS+, callers will wrap it after loading
    the stage1 classifier state into it.
    """
    model = build_model(cfg)
    feat_in = model_feature_dim(cfg)
    classifier = LinearClassifier(feat_in, int(cfg.num_classes))
    return model, classifier, feat_in


def wrap_lws_head(classifier_name: str, stage1_classifier: nn.Module,
                  feat_in: int, num_classes: int) -> nn.Module:
    """Wrap a stage1 linear head in an LWS / LWS+ head with base frozen."""
    if classifier_name == 'lws':
        head = LWSHead(feat_in, num_classes)
    elif classifier_name == 'lws_plus':
        head = LWSPlusHead(feat_in, num_classes)
    else:
        raise ValueError(f'wrap_lws_head: unsupported classifier {classifier_name}')
    stage1_linear = getattr(stage1_classifier, 'fc', stage1_classifier)
    head.load_base_from_stage1(stage1_linear)
    head.freeze_base()
    return head


__all__ = [
    'build_model',
    'model_feature_dim',
    'forward_logits',
    'extract_features',
    'build_loss',
    'build_stage1_components_vit',
    'wrap_lws_head',
]
