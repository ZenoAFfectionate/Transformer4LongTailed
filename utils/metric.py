"""Classification metrics: top-k accuracy and calibration (ECE)."""

import numpy as np
import torch


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy as percentages."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def calibration(true_labels, pred_labels, confidences, num_bins=15):
    """Reliability diagram bins + expected/max calibration error.

    Returns a dict with per-bin accuracy/confidence/count, as well as
    scalar ``expected_calibration_error`` and ``max_calibration_error``.
    All inputs are 1-D numpy arrays.
    """
    assert len(confidences) == len(pred_labels)
    assert len(confidences) == len(true_labels)
    assert num_bins > 0

    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    confidences = np.asarray(confidences, dtype=float)

    finite_mask = np.isfinite(confidences)
    true_labels = true_labels[finite_mask]
    pred_labels = pred_labels[finite_mask]
    confidences = confidences[finite_mask]

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=float)
    bin_confidences = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    total_count = int(np.sum(bin_counts))
    if total_count == 0:
        return {
            "accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "counts": bin_counts,
            "bins": bins,
            "avg_accuracy": np.nan,
            "avg_confidence": np.nan,
            "expected_calibration_error": np.nan,
            "max_calibration_error": np.nan,
        }

    avg_acc = np.sum(bin_accuracies * bin_counts) / total_count
    avg_conf = np.sum(bin_confidences * bin_counts) / total_count

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / total_count
    mce = float(np.max(gaps))

    return {
        "accuracies": bin_accuracies,
        "confidences": bin_confidences,
        "counts": bin_counts,
        "bins": bins,
        "avg_accuracy": avg_acc,
        "avg_confidence": avg_conf,
        "expected_calibration_error": ece,
        "max_calibration_error": mce,
    }
