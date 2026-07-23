"""Binary F1 metric."""

from torch import Tensor

from mambatransqr.metrics.precision import precision
from mambatransqr.metrics.recall import recall


def f1_score(predictions: Tensor, targets: Tensor, threshold: float = 0.5) -> Tensor:
    """Return binary F1 score."""
    p, r = (
        precision(predictions, targets, threshold),
        recall(predictions, targets, threshold),
    )
    return 2 * p * r / (p + r).clamp_min(1e-8)
