"""PyTorch DataLoader construction helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mambatransqr.data.collate import qr_collate


def create_dataloader(
    dataset: Any,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    sampler: Any | None = None,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] = qr_collate,
) -> Any:
    """Create a configured PyTorch DataLoader.

    Args:
        dataset: Dataset object implementing ``__len__`` and ``__getitem__``.
        batch_size: Samples per batch.
        shuffle: Whether to shuffle every epoch.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin tensor memory for accelerators.
        drop_last: Whether to discard an incomplete final batch.
        sampler: Optional sampler yielding dataset indices. Cannot be combined
            with ``shuffle=True``.
        collate_fn: Function used to combine individual samples.

    Returns:
        A ``torch.utils.data.DataLoader`` instance.

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If numeric options are invalid.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if sampler is not None and shuffle:
        raise ValueError("sampler cannot be combined with shuffle=True")
    try:
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise ImportError("PyTorch is required to create a DataLoader.") from error
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=sampler,
        collate_fn=collate_fn,
    )


def create_train_dataloader(dataset: Any, **kwargs: Any) -> Any:
    """Create a training loader with shuffling enabled by default."""
    return create_dataloader(dataset, shuffle=True, **kwargs)


def create_evaluation_dataloader(dataset: Any, **kwargs: Any) -> Any:
    """Create a validation or test loader with shuffling disabled by default."""
    return create_dataloader(dataset, shuffle=False, **kwargs)
