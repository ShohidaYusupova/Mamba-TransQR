"""Training metric logger implementations."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol


class TrainingLogger(Protocol):
    """Protocol implemented by metric loggers."""

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log scalar metrics at a global step."""

    def close(self) -> None:
        """Release logger resources."""


class CSVLogger:
    """Append scalar metrics to a CSV file.

    Args:
        path: Destination CSV file path.
    """

    def __init__(self, path: str | Path) -> None:
        """Create the destination directory and initialize state."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Append metrics, creating the header on first write."""
        row = {"step": float(step), **metrics}
        if self._fieldnames is None:
            self._fieldnames = list(row)
        if set(row) != set(self._fieldnames):
            raise ValueError("CSV metric keys must remain stable across writes")
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def close(self) -> None:
        """Close the logger (CSV writes are opened per record)."""


class TensorBoardLogger:
    """Log metrics through TensorBoard when its optional dependency is installed.

    Args:
        log_dir: Directory for TensorBoard event files.
    """

    def __init__(self, log_dir: str | Path) -> None:
        """Create a TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise ImportError("TensorBoard support requires tensorboard.") from error
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Write metrics at a global step."""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        """Flush and close the TensorBoard event writer."""
        self.writer.close()
