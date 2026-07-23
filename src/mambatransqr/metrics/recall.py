"""Binary recall metric."""
from torch import Tensor
def recall(predictions: Tensor, targets: Tensor, threshold: float = .5) -> Tensor:
    """Return binary recall."""
    predicted, actual = predictions >= threshold, targets.bool(); tp = (predicted & actual).sum().float(); return tp / (tp + (~predicted & actual).sum()).clamp_min(1)
