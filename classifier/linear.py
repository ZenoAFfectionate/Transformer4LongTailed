"""Linear-family classifier heads for stage2 training.

This module collects the PyTorch ``nn.Module`` classifier heads that operate on
frozen stage1 features:

* :class:`LinearClassifier` — a plain fully-connected head.
* :class:`LearnableWeightScaling` — LWS (learnable per-class scaling on logits).
* :class:`LWSPlus` — LWS-plus (learnable per-class scaling and bias on logits).
* :class:`LWSHead` / :class:`LWSPlusHead` — composite heads used by stage2: a
  (frozen at stage2) linear base that maps features -> logits, stacked with
  :class:`LearnableWeightScaling` / :class:`LWSPlus`. These share the same
  ``forward(features) -> logits`` signature as :class:`LinearClassifier`, so
  ``train_stage2`` can treat them uniformly.

All heads expose the same ``forward(x) -> logits`` signature expected by
``train_stage2``'s feature-retrain loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """Plain linear classification head ``z = W x + b``."""

    def __init__(self, feat_in: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(feat_in, num_classes, bias=bias)
        nn.init.kaiming_normal_(self.fc.weight)
        if bias:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class LearnableWeightScaling(nn.Module):
    """LWS head: rescale stage1 logits with per-class learnable weights."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.learned_norm * x


class LWSPlus(nn.Module):
    """LWS-plus head: learnable per-class scale *and* bias on stage1 logits.

    ``adjusted = f * x + g`` with ``f, g`` of shape ``(1, num_classes)``.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.scaling_factor = nn.Parameter(torch.ones(1, num_classes))
        self.bias = nn.Parameter(torch.zeros(1, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaling_factor * x + self.bias


class _LWSBaseHead(nn.Module):
    """Shared base: frozen linear maps features -> logits, scale head is trained.

    ``base`` is loaded from the stage1 classifier and frozen so that only the
    ``scale`` module's parameters receive gradients at stage2.
    """

    def __init__(self, feat_in: int, num_classes: int, scale: nn.Module):
        super().__init__()
        self.base = nn.Linear(feat_in, num_classes)
        nn.init.kaiming_normal_(self.base.weight)
        nn.init.zeros_(self.base.bias)
        self.scale = scale

    def freeze_base(self) -> None:
        for param in self.base.parameters():
            param.requires_grad = False
        self.base.eval()

    def load_base_from_stage1(self, stage1_linear: nn.Linear) -> None:
        """Copy a stage1 ``nn.Linear`` (or stage1 ``Classifier.fc``) into ``base``."""
        self.base.weight.data.copy_(stage1_linear.weight.data)
        if stage1_linear.bias is not None and self.base.bias is not None:
            self.base.bias.data.copy_(stage1_linear.bias.data)

    def trainable_parameters(self):
        return self.scale.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale(self.base(x))


class LWSHead(_LWSBaseHead):
    """Frozen linear + :class:`LearnableWeightScaling`."""

    def __init__(self, feat_in: int, num_classes: int):
        super().__init__(feat_in, num_classes, LearnableWeightScaling(num_classes))


class LWSPlusHead(_LWSBaseHead):
    """Frozen linear + :class:`LWSPlus`."""

    def __init__(self, feat_in: int, num_classes: int):
        super().__init__(feat_in, num_classes, LWSPlus(num_classes))


__all__ = [
    'LinearClassifier',
    'LearnableWeightScaling',
    'LWSPlus',
    'LWSHead',
    'LWSPlusHead',
]
