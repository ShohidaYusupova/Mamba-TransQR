"""Confusion-matrix computation."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_confusion_matrix(
    predictions: Tensor, targets: Tensor, num_classes: int
) -> Tensor:
    """Compute a class-by-class confusion matrix.

    Args:
        predictions: Predicted class indices or class logits.
        targets: Ground-truth class indices.
        num_classes: Number of target classes.

    Returns:
        Integer matrix where rows are actual classes and columns are predictions.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    predicted = (
        predictions.argmax(dim=1) if predictions.ndim > targets.ndim else predictions
    )
    predicted = predicted.reshape(-1).long()
    actual = targets.reshape(-1).long()
    if predicted.numel() != actual.numel():
        raise ValueError("predictions and targets must have matching element counts")
    valid = (
        (actual >= 0)
        & (actual < num_classes)
        & (predicted >= 0)
        & (predicted < num_classes)
    )
    indices = actual[valid] * num_classes + predicted[valid]
    return torch.bincount(indices, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )
