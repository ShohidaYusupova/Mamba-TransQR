"""Pixel reconstruction losses."""

from __future__ import annotations

from torch import Tensor, nn


class ReconstructionLoss(nn.Module):
    """Configurable L1 or L2 reconstruction loss.

    Args:
        mode: Either ``l1`` or ``mse``.
    """

    def __init__(self, mode: str = "l1") -> None:
        """Initialize the requested reconstruction criterion."""
        super().__init__()
        if mode not in {"l1", "mse"}:
            raise ValueError("mode must be 'l1' or 'mse'")
        self.criterion: nn.Module = nn.L1Loss() if mode == "l1" else nn.MSELoss()

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Return pixel reconstruction error."""
        return self.criterion(predictions, targets)
