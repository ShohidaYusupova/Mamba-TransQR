"""PSNR metric."""

import torch
from torch import Tensor


def psnr(predictions: Tensor, targets: Tensor, data_range: float = 1.0) -> Tensor:
    """Return peak signal-to-noise ratio in decibels."""
    error = (
        (predictions - targets)
        .square()
        .mean()
        .clamp_min(torch.finfo(predictions.dtype).eps)
    )
    return 20 * torch.log10(
        torch.tensor(data_range, device=predictions.device)
    ) - 10 * torch.log10(error)
