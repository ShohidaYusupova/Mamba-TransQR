"""SSIM metric."""

from torch import Tensor


def ssim(predictions: Tensor, targets: Tensor, data_range: float = 1.0) -> Tensor:
    """Return global structural similarity."""
    dims = tuple(range(2, predictions.ndim))
    x = predictions.mean(dims, keepdim=True)
    y = targets.mean(dims, keepdim=True)
    vx = ((predictions - x).square()).mean(dims, keepdim=True)
    vy = ((targets - y).square()).mean(dims, keepdim=True)
    cov = ((predictions - x) * (targets - y)).mean(dims, keepdim=True)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    return (
        ((2 * x * y + c1) * (2 * cov + c2))
        / ((x.square() + y.square() + c1) * (vx + vy + c2))
    ).mean()
