"""Mixup / ReMix data augmentation utilities.

Shapes and conventions match DS-AWBN/utils/mixup.py so long-tail losses and
training loops carry over unchanged.
"""

from __future__ import absolute_import

import numpy as np
import torch


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Classic mixup — returns mixed inputs, pair of targets, and scalar lambda."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    batch_size = x.size(0)
    if use_cuda and torch.cuda.is_available():
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Standard mixup loss — scalar lambda interpolation."""
    loss_a = criterion(pred, y_a)
    loss_b = criterion(pred, y_b)
    if torch.is_tensor(loss_a) and loss_a.ndim > 0:
        return (lam * loss_a + (1 - lam) * loss_b).mean()
    return lam * loss_a + (1 - lam) * loss_b


def remix_data(x, y, class_counts=None, alpha=1.0, kappa=3.0, tau=0.5):
    """ReMix — mixup in the input with class-frequency-aware label coefficient.

    Returns ``(mixed_x, y_a, y_b, lam_x, lam_y)``. ``lam_x`` is the per-sample
    Beta sample; ``lam_y`` is the adjusted label weight that preserves tail
    samples when a head sample dominates the mix.
    """
    B, device = x.size(0), x.device

    if alpha > 0:
        lam_x = torch.distributions.Beta(alpha, alpha).sample((B,)).to(device)
    else:
        lam_x = torch.ones(B, device=device)

    index = torch.randperm(B, device=device)
    x2, y2 = x[index], y[index]

    mixed_x = lam_x.view(B, 1, 1, 1) * x + (1.0 - lam_x).view(B, 1, 1, 1) * x2

    if class_counts is None:
        return mixed_x, y, y2, lam_x, lam_x

    counts = class_counts if isinstance(class_counts, torch.Tensor) else torch.tensor(
        class_counts, device=device
    )
    counts = counts.to(dtype=torch.float, device=device)
    n_i = counts[y]
    n_j = counts[y2]
    ratio = n_i / (n_j + 1e-12)

    cond_major = (ratio >= kappa) & (lam_x < tau)
    cond_minor = (ratio <= 1.0 / kappa) & ((1.0 - lam_x) < tau)
    lam_y = torch.where(
        cond_major,
        torch.zeros_like(lam_x),
        torch.where(cond_minor, torch.ones_like(lam_x), lam_x),
    )
    return mixed_x, y, y2, lam_x, lam_y


def remix_criterion(criterion, pred, y_a, y_b, lam):
    """Per-sample ReMix loss — requires ``criterion`` with ``reduction='none'``."""
    loss_a = criterion(pred, y_a)
    loss_b = criterion(pred, y_b)
    if loss_a.ndim == 0 or loss_b.ndim == 0:
        raise ValueError("Remix 需要 criterion(reduction='none') 以获得逐样本损失。")

    lam = lam.to(loss_a.dtype).to(loss_a.device)
    if lam.ndim == 0:
        lam = lam.expand_as(loss_a)
    return (lam * loss_a + (1.0 - lam) * loss_b).mean()


__all__ = [
    'mixup_data',
    'mixup_criterion',
    'remix_data',
    'remix_criterion',
]
