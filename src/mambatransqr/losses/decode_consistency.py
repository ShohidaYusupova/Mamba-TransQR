"""Differentiable QR decoding surrogate; it does not call external decoders."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DecodeConsistencyLoss(nn.Module):
    """Encourage confident binarization and local module contrast differentiably."""

    def __init__(self, temperature: float = 0.1, threshold: float = 0.5) -> None:
        super().__init__()
        if temperature <= 0 or not 0 <= threshold <= 1:
            raise ValueError("temperature must be positive and threshold in [0, 1]")
        self.temperature, self.threshold = temperature, threshold

    def forward(self, predictions: Tensor, targets: Tensor | None = None) -> Tensor:
        gray = predictions.mean(dim=1, keepdim=True)
        confidence = torch.sigmoid((gray - self.threshold).abs() / self.temperature)
        ambiguity = 1 - confidence.mean()
        contrast = torch.nn.functional.avg_pool2d(gray, 3, 1, 1)
        local_contrast = torch.relu(0.15 - (gray - contrast).abs()).mean()
        target_term = torch.zeros_like(ambiguity)
        if targets is not None:
            target_term = torch.nn.functional.l1_loss(
                torch.sigmoid((gray - self.threshold) / self.temperature),
                (targets.mean(dim=1, keepdim=True) > self.threshold).to(gray.dtype),
            )
        return ambiguity + local_contrast + target_term
