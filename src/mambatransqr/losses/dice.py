"""Dice segmentation loss."""

from __future__ import annotations

from torch import Tensor, nn


class DiceLoss(nn.Module):
    """Binary Dice loss with logits or probabilities.

    Args:
        from_logits: Apply sigmoid to predictions before scoring.
        smooth: Numerical stability constant.
    """

    def __init__(self, from_logits: bool = True, smooth: float = 1.0) -> None:
        """Initialize Dice-loss configuration."""
        super().__init__()
        self.from_logits = from_logits
        self.smooth = smooth

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Return one minus batch-mean Dice score."""
        values = predictions.sigmoid() if self.from_logits else predictions
        values = values.flatten(1)
        expected = targets.flatten(1)
        score = (2 * (values * expected).sum(1) + self.smooth) / (
            values.sum(1) + expected.sum(1) + self.smooth
        )
        return 1.0 - score.mean()
