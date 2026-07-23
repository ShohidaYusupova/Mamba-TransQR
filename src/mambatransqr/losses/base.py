"""Base interfaces for modular losses."""

from __future__ import annotations

from torch import Tensor, nn


class BaseLoss(nn.Module):
    """Base class for named loss modules."""

    name: str = "loss"

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Calculate a scalar loss value."""
        raise NotImplementedError
