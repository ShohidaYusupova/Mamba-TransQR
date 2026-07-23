"""Robust Charbonnier loss."""

from __future__ import annotations

from torch import Tensor, nn


class CharbonnierLoss(nn.Module):
    """Smooth robust L1 loss.

    Args:
        epsilon: Stability constant inside the square root.
    """

    def __init__(self, epsilon: float = 1e-3) -> None:
        """Initialize robust loss settings."""
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Return mean Charbonnier error."""
        return ((predictions - targets).square() + self.epsilon**2).sqrt().mean()
