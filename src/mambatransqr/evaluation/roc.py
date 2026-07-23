"""Receiver-operating-characteristic utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def roc_curve(scores: Tensor, targets: Tensor, thresholds: int = 101) -> tuple[Tensor, Tensor, Tensor]:
    """Compute binary ROC false-positive and true-positive rates.

    Args:
        scores: Predicted positive-class probabilities.
        targets: Binary target labels.
        thresholds: Number of thresholds evenly spaced from 0 to 1.

    Returns:
        False-positive rates, true-positive rates, and threshold values.
    """
    if thresholds < 2:
        raise ValueError("thresholds must be at least 2")
    levels = torch.linspace(0, 1, thresholds, device=scores.device)
    actual = targets.bool()
    positives = actual.sum().clamp_min(1)
    negatives = (~actual).sum().clamp_min(1)
    true_positive_rates: list[Tensor] = []
    false_positive_rates: list[Tensor] = []
    for level in levels:
        predicted = scores >= level
        true_positive_rates.append((predicted & actual).sum().float() / positives)
        false_positive_rates.append((predicted & ~actual).sum().float() / negatives)
    return torch.stack(false_positive_rates), torch.stack(true_positive_rates), levels


def auc(false_positive_rate: Tensor, true_positive_rate: Tensor) -> Tensor:
    """Compute ROC area under the curve using trapezoidal integration."""
    order = torch.argsort(false_positive_rate)
    return torch.trapz(true_positive_rate[order], false_positive_rate[order])
