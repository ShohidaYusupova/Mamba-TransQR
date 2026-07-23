"""Batch collation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def qr_collate(batch: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
    """Collate QR samples while preserving heterogeneous metadata.

    Args:
        batch: Sequence of samples returned by :class:`QRDataset`.

    Returns:
        Dictionary mapping every observed key to a list of per-sample values.

    Raises:
        ValueError: If ``batch`` is empty.
    """
    if not batch:
        raise ValueError("batch must contain at least one sample")
    keys = set().union(*(sample.keys() for sample in batch))
    return {key: [sample.get(key) for sample in batch] for key in sorted(keys)}
