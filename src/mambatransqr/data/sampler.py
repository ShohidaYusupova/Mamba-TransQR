"""Deterministic sampling utilities."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from random import Random


@dataclass(slots=True)
class EpochSampler:
    """Yield dataset indices, optionally shuffled for each epoch.

    Args:
        size: Number of dataset items.
        shuffle: Whether to shuffle indices.
        seed: Base random seed.
    """

    size: int
    shuffle: bool = True
    seed: int = 0
    _epoch: int = 0

    def __post_init__(self) -> None:
        """Validate sampler size."""
        if self.size < 0:
            raise ValueError("size must be non-negative")

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to derive the shuffle order.

        Args:
            epoch: Non-negative epoch number.
        """
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """Yield indices in stable or epoch-shuffled order."""
        indices = list(range(self.size))
        if self.shuffle:
            Random(self.seed + self._epoch).shuffle(indices)
        yield from indices

    def __len__(self) -> int:
        """Return the number of yielded indices."""
        return self.size
