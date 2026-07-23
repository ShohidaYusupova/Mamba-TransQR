"""Composable weighted-loss management."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn


class LossManager:
    """Combine named PyTorch loss functions into one optimization loss.

    Args:
        losses: Mapping from metric names to loss modules.
        weights: Optional multiplier for each named loss.
    """

    def __init__(
        self,
        losses: Mapping[str, nn.Module],
        weights: Mapping[str, float] | None = None,
    ) -> None:
        """Validate and store configured loss functions."""
        if not losses:
            raise ValueError("at least one loss function is required")
        self.losses = dict(losses)
        self.weights = dict(weights or {})
        unknown = set(self.weights) - set(self.losses)
        if unknown:
            raise ValueError(
                f"weights configured for unknown losses: {sorted(unknown)}"
            )

    def __call__(
        self, predictions: Tensor, targets: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute weighted total and unweighted named scalar losses.

        Args:
            predictions: Model output tensor.
            targets: Supervision tensor.

        Returns:
            Total differentiable loss and scalar metric mapping.
        """
        total: Tensor | None = None
        metrics: dict[str, float] = {}
        for name, loss_fn in self.losses.items():
            value = loss_fn(predictions, targets)
            weight = self.weights.get(name, 1.0)
            total = value * weight if total is None else total + value * weight
            metrics[name] = float(value.detach())
        if total is None:
            raise RuntimeError("no loss values were computed")
        metrics["loss"] = float(total.detach())
        return total, metrics
