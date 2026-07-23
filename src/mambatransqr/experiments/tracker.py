"""Experiment metric tracking."""

import csv
from pathlib import Path


class ExperimentTracker:
    """Append experiment metrics to CSV."""

    def __init__(self, path: str | Path) -> None:
        """Create tracking path."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields: list[str] | None = None

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Append one metric row."""
        row = {"step": float(step), **metrics}
        self.fields = self.fields or list(row)
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
