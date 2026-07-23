"""Structural similarity loss."""

from __future__ import annotations

from torch import Tensor, nn


class SSIMLoss(nn.Module):
    """Minimize one minus global structural similarity.

    Args:
        data_range: Pixel value range.
    """

    def __init__(self, data_range: float = 1.0) -> None:
        """Initialize SSIM loss."""
        super().__init__()
        self.data_range = data_range

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Return structural dissimilarity."""
        dimensions = tuple(range(2, predictions.ndim))
        mean_x = predictions.mean(dim=dimensions, keepdim=True)
        mean_y = targets.mean(dim=dimensions, keepdim=True)
        variance_x = ((predictions - mean_x).square()).mean(
            dim=dimensions, keepdim=True
        )
        variance_y = ((targets - mean_y).square()).mean(dim=dimensions, keepdim=True)
        covariance = ((predictions - mean_x) * (targets - mean_y)).mean(
            dim=dimensions, keepdim=True
        )
        c1, c2 = (0.01 * self.data_range) ** 2, (0.03 * self.data_range) ** 2
        score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
            (mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2)
        )
        return 1.0 - score.mean()
