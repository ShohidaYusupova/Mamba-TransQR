"""Weighted composition of modular losses."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn


class CombinedLoss(nn.Module):
    """Combine named loss modules with configurable weights.

    Args:
        losses: Mapping of name to loss module.
        weights: Optional mapping of loss multipliers.
    """

    def __init__(self, losses: Mapping[str, nn.Module], weights: Mapping[str, float] | None = None) -> None:
        """Register all loss modules and validate weight names."""
        super().__init__()
        if not losses:
            raise ValueError("at least one loss is required")
        self.losses = nn.ModuleDict(losses)
        self.weights = dict(weights or {})
        unknown = set(self.weights) - set(losses)
        if unknown:
            raise ValueError(f"unknown loss weights: {sorted(unknown)}")

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Return the weighted sum of all configured loss values."""
        return sum(
            loss(predictions, targets) * self.weights.get(name, 1.0)
            for name, loss in self.losses.items()
        )
