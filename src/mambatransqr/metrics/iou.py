"""Binary IoU metric."""
from torch import Tensor
def iou(predictions: Tensor, targets: Tensor, threshold: float = .5) -> Tensor:
    """Return binary intersection-over-union."""
    predicted, actual = predictions >= threshold, targets.bool(); return (predicted & actual).sum().float() / (predicted | actual).sum().clamp_min(1)
