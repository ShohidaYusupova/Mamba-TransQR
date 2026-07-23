"""Image and classification metric implementations."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import Tensor


def mse(predictions: Tensor, targets: Tensor) -> Tensor:
    """Compute mean squared error.

    Args:
        predictions: Predicted tensor.
        targets: Reference tensor.

    Returns:
        Scalar MSE tensor.
    """
    return torch.mean((predictions - targets).square())


def mae(predictions: Tensor, targets: Tensor) -> Tensor:
    """Compute mean absolute error."""
    return torch.mean(torch.abs(predictions - targets))


def psnr(predictions: Tensor, targets: Tensor, data_range: float = 1.0) -> Tensor:
    """Compute peak signal-to-noise ratio in decibels."""
    error = mse(predictions, targets)
    return 20 * torch.log10(torch.tensor(data_range, device=error.device)) - 10 * torch.log10(
        error.clamp_min(torch.finfo(error.dtype).eps)
    )


def ssim(predictions: Tensor, targets: Tensor, data_range: float = 1.0) -> Tensor:
    """Compute a global structural similarity approximation.

    Args:
        predictions: Predicted image tensor of shape ``(N, C, H, W)``.
        targets: Reference image tensor with matching shape.
        data_range: Pixel value range.

    Returns:
        Mean SSIM score over batch and channels.
    """
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have matching shapes")
    dimensions = tuple(range(2, predictions.ndim))
    mean_x = predictions.mean(dim=dimensions, keepdim=True)
    mean_y = targets.mean(dim=dimensions, keepdim=True)
    variance_x = ((predictions - mean_x).square()).mean(dim=dimensions, keepdim=True)
    variance_y = ((targets - mean_y).square()).mean(dim=dimensions, keepdim=True)
    covariance = ((predictions - mean_x) * (targets - mean_y)).mean(
        dim=dimensions, keepdim=True
    )
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2)
    )
    return score.mean()


def accuracy(logits: Tensor, targets: Tensor) -> Tensor:
    """Compute classification accuracy from logits and class indices."""
    return (logits.argmax(dim=1) == targets).float().mean()


def top_k_accuracy(logits: Tensor, targets: Tensor, k: int) -> Tensor:
    """Compute top-k classification accuracy."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, classes)")
    if not 1 <= k <= logits.shape[1]:
        raise ValueError("k must be between 1 and the number of classes")
    matches = logits.topk(k, dim=1).indices.eq(targets.unsqueeze(1))
    return matches.any(dim=1).float().mean()


def binary_confusion(predictions: Tensor, targets: Tensor, threshold: float = 0.5) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return true positives, false positives, false negatives, and true negatives."""
    predicted = predictions >= threshold
    actual = targets.bool()
    true_positive = (predicted & actual).sum()
    false_positive = (predicted & ~actual).sum()
    false_negative = (~predicted & actual).sum()
    true_negative = (~predicted & ~actual).sum()
    return true_positive, false_positive, false_negative, true_negative


def precision(predictions: Tensor, targets: Tensor, threshold: float = 0.5) -> Tensor:
    """Compute binary precision."""
    true_positive, false_positive, _, _ = binary_confusion(predictions, targets, threshold)
    return true_positive.float() / (true_positive + false_positive).clamp_min(1)


def recall(predictions: Tensor, targets: Tensor, threshold: float = 0.5) -> Tensor:
    """Compute binary recall."""
    true_positive, _, false_negative, _ = binary_confusion(predictions, targets, threshold)
    return true_positive.float() / (true_positive + false_negative).clamp_min(1)


def f1_score(predictions: Tensor, targets: Tensor, threshold: float = 0.5) -> Tensor:
    """Compute binary F1 score."""
    score_precision = precision(predictions, targets, threshold)
    score_recall = recall(predictions, targets, threshold)
    return 2 * score_precision * score_recall / (score_precision + score_recall).clamp_min(
        torch.finfo(score_precision.dtype).eps
    )


def iou(predictions: Tensor, targets: Tensor, threshold: float = 0.5) -> Tensor:
    """Compute binary intersection-over-union."""
    predicted = predictions >= threshold
    actual = targets.bool()
    intersection = (predicted & actual).sum().float()
    union = (predicted | actual).sum().float()
    return intersection / union.clamp_min(1.0)


class MetricsManager:
    """Calculate a configured collection of scalar evaluation metrics.

    Args:
        metrics: Metric names to calculate. Defaults include MSE, MAE, PSNR,
            and SSIM.
        data_range: Pixel value range used by image metrics.
    """

    def __init__(
        self,
        metrics: tuple[str, ...] = ("mse", "mae", "psnr", "ssim"),
        data_range: float = 1.0,
    ) -> None:
        """Validate metric names and settings."""
        if data_range <= 0:
            raise ValueError("data_range must be positive")
        self.metrics = tuple(name.lower() for name in metrics)
        self.data_range = data_range
        available = set(self.available_metrics())
        unknown = set(self.metrics) - available
        if unknown:
            raise ValueError(f"unsupported metrics: {sorted(unknown)}")

    @staticmethod
    def available_metrics() -> tuple[str, ...]:
        """Return supported metric names."""
        return (
            "psnr",
            "ssim",
            "lpips",
            "mse",
            "mae",
            "precision",
            "recall",
            "f1",
            "iou",
            "accuracy",
            "top1",
            "top5",
        )

    def compute(self, predictions: Tensor, targets: Tensor) -> dict[str, float]:
        """Calculate configured metrics.

        Args:
            predictions: Model predictions.
            targets: Ground-truth tensor.

        Returns:
            Mapping of metric name to Python float.
        """
        functions: Mapping[str, Callable[[], Tensor]] = {
            "mse": lambda: mse(predictions, targets),
            "mae": lambda: mae(predictions, targets),
            "psnr": lambda: psnr(predictions, targets, self.data_range),
            "ssim": lambda: ssim(predictions, targets, self.data_range),
            "precision": lambda: precision(predictions, targets),
            "recall": lambda: recall(predictions, targets),
            "f1": lambda: f1_score(predictions, targets),
            "iou": lambda: iou(predictions, targets),
            "accuracy": lambda: accuracy(predictions, targets),
            "top1": lambda: top_k_accuracy(predictions, targets, 1),
            "top5": lambda: top_k_accuracy(predictions, targets, 5),
        }
        results: dict[str, float] = {}
        for name in self.metrics:
            if name == "lpips":
                results[name] = self._lpips(predictions, targets)
            else:
                results[name] = float(functions[name]().detach())
        return results

    @staticmethod
    def _lpips(predictions: Tensor, targets: Tensor) -> float:
        """Compute LPIPS through its optional third-party dependency."""
        try:
            import lpips
        except ImportError as error:
            raise ImportError("LPIPS metric requires the optional lpips package.") from error
        model = lpips.LPIPS(net="alex").to(predictions.device)
        model.eval()
        with torch.no_grad():
            value = model(predictions * 2 - 1, targets * 2 - 1).mean()
        return float(value)
