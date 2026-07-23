"""Edge-preserving image loss."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EdgeLoss(nn.Module):
    """Match Sobel edge responses between prediction and target images."""

    def __init__(self) -> None:
        """Initialize fixed Sobel kernels."""
        super().__init__()
        self.register_buffer(
            "horizontal",
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(
                1, 1, 3, 3
            ),
        )
        self.register_buffer(
            "vertical",
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(
                1, 1, 3, 3
            ),
        )

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Return L1 difference of channelwise Sobel edge magnitudes."""
        return torch.nn.functional.l1_loss(
            self._edges(predictions), self._edges(targets)
        )

    def _edges(self, images: Tensor) -> Tensor:
        """Calculate per-channel Sobel magnitude."""
        channels = images.shape[1]
        horizontal = self.horizontal.expand(channels, 1, 3, 3)
        vertical = self.vertical.expand(channels, 1, 3, 3)
        x = torch.nn.functional.conv2d(images, horizontal, padding=1, groups=channels)
        y = torch.nn.functional.conv2d(images, vertical, padding=1, groups=channels)
        return torch.sqrt(x.square() + y.square() + 1e-6)
