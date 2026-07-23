"""Binary precision metric."""

from torch import Tensor


def precision(predictions: Tensor, targets: Tensor, threshold: float = 0.5) -> Tensor:
    """Return binary precision."""
    predicted, actual = predictions >= threshold, targets.bool()
    tp = (predicted & actual).sum().float()
    return tp / (tp + (predicted & ~actual).sum()).clamp_min(1)
