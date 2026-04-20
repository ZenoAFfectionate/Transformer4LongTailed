"""Loss functions for balanced and long-tail classification.

Exposed classes (all ``nn.Module``, ``(logits, target) -> loss``):

* :class:`WeightedCrossEntropyLoss` — effective-number reweighted CE.
* :class:`LabelAwareSmoothing` — per-class label smoothing (Zhang 2021).
* :class:`FocalLoss` — down-weight easy samples via ``(1 - p_t)^gamma``.
* :class:`LDAMLoss` — label-distribution-aware margin (Cao 2019).
* :class:`BalancedSoftmaxLoss` — softmax calibrated to balanced test (Ren 2020).
* :class:`LogitAdjustmentLoss` — additive ``tau * log(prior)`` (Menon 2021).
* :class:`ClassBalancedLoss` — effective-number reweighting wrapper (Cui 2019).

All classes accept ``cls_num_list`` (per-class training count) when relevant
and register their per-class tensors as buffers so ``.cuda()`` / ``.to()``
propagate correctly.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _effective_num_weights(cls_num_list: Sequence[int], beta: float = 0.9999) -> torch.Tensor:
    """Effective-number class weights (Cui et al., 2019)."""
    cls_num = np.asarray(cls_num_list, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, cls_num)
    per_cls_weights = (1.0 - beta) / np.clip(effective_num, a_min=1e-12, a_max=None)
    per_cls_weights = per_cls_weights / per_cls_weights.sum() * len(cls_num)
    return torch.tensor(per_cls_weights, dtype=torch.float)


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy with optional effective-number reweighting + label smoothing.

    ``label_smoothing`` is forwarded to ``F.cross_entropy`` (PyTorch native
    since 1.10). When combined with mixup, the per-target CE is computed per
    mix side and then ``lam``-weighted in ``mixup_criterion`` — smoothing
    composes correctly because it is applied on each side independently.
    """

    def __init__(
        self,
        cls_num_list: Optional[Sequence[int]] = None,
        reweight_CE: bool = False,
        reduction: str = 'mean',
        beta: float = 0.9999,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if reduction not in {'mean', 'sum', 'none'}:
            raise ValueError(f"Unsupported reduction type: {reduction}")
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

        if reweight_CE:
            if cls_num_list is None:
                raise ValueError('reweight_CE=True requires cls_num_list.')
            weights = _effective_num_weights(cls_num_list, beta=beta)
            self.register_buffer('per_cls_weights', weights)
        else:
            self.per_cls_weights = None

    def forward(self, output_logits, target, extra_info=None):
        per_sample = F.cross_entropy(
            output_logits,
            target,
            weight=self.per_cls_weights,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        if self.reduction == 'none':
            return per_sample
        if self.reduction == 'mean':
            return per_sample.mean()
        return per_sample.sum()


CrossEntropyLoss = WeightedCrossEntropyLoss


class LabelAwareSmoothing(nn.Module):
    """Smoothing strength per-class, heavier on head, lighter on tail.

    ``shape`` ∈ {'concave', 'linear', 'convex', 'exp'}. ``exp`` requires ``power``.
    """

    def __init__(
        self,
        cls_num_list: Sequence[int],
        smooth_head: float,
        smooth_tail: float,
        shape: str = 'concave',
        power: Optional[float] = None,
    ):
        super().__init__()
        cls_num_array = np.asarray(cls_num_list, dtype=np.float64)
        n_1 = cls_num_array.max()
        n_K = cls_num_array.min()
        if n_1 == n_K:
            smooth = np.full_like(cls_num_array, smooth_tail, dtype=np.float64)
        elif shape == 'concave':
            smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin(
                (cls_num_array - n_K) * np.pi / (2 * (n_1 - n_K))
            )
        elif shape == 'linear':
            smooth = smooth_tail + (smooth_head - smooth_tail) * (cls_num_array - n_K) / (n_1 - n_K)
        elif shape == 'convex':
            smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(
                1.5 * np.pi + (cls_num_array - n_K) * np.pi / (2 * (n_1 - n_K))
            )
        elif shape == 'exp':
            if power is None:
                raise ValueError("shape='exp' requires a positive `power`.")
            smooth = smooth_tail + (smooth_head - smooth_tail) * np.power(
                (cls_num_array - n_K) / (n_1 - n_K), power
            )
        else:
            raise ValueError(f'Unsupported LabelAwareSmoothing shape: {shape}')

        self.register_buffer('smooth', torch.tensor(smooth, dtype=torch.float))

    def forward(self, x, target):
        smoothing = self.smooth[target]
        confidence = 1.0 - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


class FocalLoss(nn.Module):
    """``loss = alpha_t * (1 - p_t)^gamma * -log(p_t)`` (Lin et al., 2017)."""

    def __init__(
        self,
        gamma: float = 2.0,
        cls_num_list: Optional[Sequence[int]] = None,
        use_effective_num: bool = False,
        beta: float = 0.9999,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

        if use_effective_num:
            if cls_num_list is None:
                raise ValueError('use_effective_num=True requires cls_num_list.')
            alpha = _effective_num_weights(cls_num_list, beta=beta)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        logp_t = logp.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        focal_weight = (1.0 - p_t).pow(self.gamma)
        loss = -focal_weight * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin (Cao et al., 2019)."""

    def __init__(
        self,
        cls_num_list: Sequence[int],
        max_m: float = 0.5,
        weight: Optional[torch.Tensor] = None,
        s: float = 30.0,
    ):
        super().__init__()
        cls_num_array = np.asarray(cls_num_list, dtype=np.float64)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_array))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer('m_list', torch.tensor(m_list, dtype=torch.float))
        self.s = float(s)
        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight, dtype=torch.float)
            self.register_buffer('weight', weight.float())
        else:
            self.weight = None

    def forward(self, logits, target):
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, target.unsqueeze(1), True)
        batch_m = self.m_list[target].unsqueeze(1)
        logits_m = logits - index.float() * batch_m
        return F.cross_entropy(self.s * logits_m, target, weight=self.weight)


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax — softmax calibrated to a balanced test distribution."""

    def __init__(self, cls_num_list: Sequence[int]):
        super().__init__()
        cls_num_array = np.asarray(cls_num_list, dtype=np.float64)
        prior = cls_num_array / cls_num_array.sum()
        log_prior = np.log(np.clip(prior, a_min=1e-12, a_max=None))
        self.register_buffer('log_prior', torch.tensor(log_prior, dtype=torch.float))

    def forward(self, logits, target):
        adjusted_logits = logits + self.log_prior.unsqueeze(0)
        return F.cross_entropy(adjusted_logits, target)


class LogitAdjustmentLoss(nn.Module):
    """``CE(logits + tau * log prior, y)`` (Menon et al., 2021)."""

    def __init__(self, cls_num_list: Sequence[int], tau: float = 1.0):
        super().__init__()
        cls_num_array = np.asarray(cls_num_list, dtype=np.float64)
        prior = cls_num_array / cls_num_array.sum()
        log_prior = np.log(np.clip(prior, a_min=1e-12, a_max=None))
        self.register_buffer('log_prior', torch.tensor(log_prior, dtype=torch.float))
        self.tau = float(tau)

    def forward(self, logits, target):
        return F.cross_entropy(logits + self.tau * self.log_prior.unsqueeze(0), target)


class ClassBalancedLoss(nn.Module):
    """Effective-number reweighted wrapper around a per-sample base loss."""

    def __init__(
        self,
        cls_num_list: Sequence[int],
        base_loss: Optional[nn.Module] = None,
        beta: float = 0.9999,
        reduction: str = 'mean',
    ):
        super().__init__()
        weights = _effective_num_weights(cls_num_list, beta=beta)
        self.register_buffer('per_cls_weights', weights)
        self.base_loss = base_loss
        self.reduction = reduction

    def forward(self, logits, target):
        if self.base_loss is None:
            per_sample = F.cross_entropy(logits, target, reduction='none')
        else:
            per_sample = self.base_loss(logits, target)
            if per_sample.ndim == 0:
                raise ValueError(
                    'ClassBalancedLoss requires base_loss with reduction="none".'
                )
        weighted = per_sample * self.per_cls_weights[target]
        if self.reduction == 'mean':
            return weighted.mean()
        if self.reduction == 'sum':
            return weighted.sum()
        return weighted


__all__ = [
    'WeightedCrossEntropyLoss',
    'CrossEntropyLoss',
    'LabelAwareSmoothing',
    'FocalLoss',
    'LDAMLoss',
    'BalancedSoftmaxLoss',
    'LogitAdjustmentLoss',
    'ClassBalancedLoss',
]
