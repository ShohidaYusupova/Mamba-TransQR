"""Batch collation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor


def qr_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate QR samples while preserving heterogeneous metadata.

    Args:
        batch: Sequence of samples returned by :class:`QRDataset`.

    Returns:
        Dictionary mapping tensor fields to stacked tensors and metadata fields
        to lists. Missing optional values remain a list containing ``None``.

    Raises:
        ValueError: If ``batch`` is empty.
    """
    if not batch:
        raise ValueError("batch must contain at least one sample")
    keys = set().union(*(sample.keys() for sample in batch))
    return {
        key: _collate_values([sample.get(key) for sample in batch])
        for key in sorted(keys)
    }


def _collate_values(values: list[Any]) -> Any:
    """Stack homogeneous tensors while retaining heterogeneous metadata."""
    if values and all(isinstance(value, Tensor) for value in values):
        return torch.stack(values)
    return values
