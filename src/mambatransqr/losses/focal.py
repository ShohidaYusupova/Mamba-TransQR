"""Focal classification loss."""

from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor, nn


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    Args:
        gamma: Focusing exponent.
        alpha: Optional scalar class-balance multiplier.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        """Initialize focal-loss parameters."""
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Return mean focal cross-entropy."""
        cross_entropy = functional.cross_entropy(predictions, targets, reduction="none")
        probability = (-cross_entropy).exp()
        loss = (1 - probability).pow(self.gamma) * cross_entropy
        return (self.alpha * loss if self.alpha is not None else loss).mean()
