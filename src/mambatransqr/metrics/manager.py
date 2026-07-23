"""Metric collection manager."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from torch import Tensor

from mambatransqr.metrics.accuracy import accuracy, top_k_accuracy
from mambatransqr.metrics.f1 import f1_score
from mambatransqr.metrics.iou import iou
from mambatransqr.metrics.precision import precision
from mambatransqr.metrics.psnr import psnr
from mambatransqr.metrics.recall import recall
from mambatransqr.metrics.ssim import ssim


class MetricsManager:
    """Calculate configured named metrics."""

    def __init__(self, names: tuple[str, ...] = ("psnr", "ssim")) -> None:
        """Store metric names after validation."""
        self.names = names
        unknown = set(names) - {
            "psnr",
            "ssim",
            "precision",
            "recall",
            "f1",
            "iou",
            "accuracy",
            "top1",
            "top5",
        }
        if unknown:
            raise ValueError(f"unsupported metrics: {sorted(unknown)}")

    def compute(self, predictions: Tensor, targets: Tensor) -> dict[str, float]:
        """Return scalar metric values."""
        functions: Mapping[str, Callable[[], Tensor]] = {
            "psnr": lambda: psnr(predictions, targets),
            "ssim": lambda: ssim(predictions, targets),
            "precision": lambda: precision(predictions, targets),
            "recall": lambda: recall(predictions, targets),
            "f1": lambda: f1_score(predictions, targets),
            "iou": lambda: iou(predictions, targets),
            "accuracy": lambda: accuracy(predictions, targets),
            "top1": lambda: top_k_accuracy(predictions, targets, 1),
            "top5": lambda: top_k_accuracy(predictions, targets, 5),
        }
        return {name: float(functions[name]().detach()) for name in self.names}
