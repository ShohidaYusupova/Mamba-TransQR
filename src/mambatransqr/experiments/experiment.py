"""Experiment lifecycle management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mambatransqr.experiments.config import save_config
from mambatransqr.experiments.tracker import ExperimentTracker


@dataclass(slots=True)
class Experiment:
    """Create isolated directories, logs, configs, and JSON summaries."""

    name: str
    root: str | Path = "runs"
    config: dict[str, Any] = field(default_factory=dict)
    directory: Path = field(init=False)
    tracker: ExperimentTracker = field(init=False)

    def __post_init__(self) -> None:
        """Create automatic experiment directory structure."""
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.directory = Path(self.root) / f"{timestamp}_{self.name}"
        self.directory.mkdir(parents=True, exist_ok=False)
        save_config(self.config, self.directory / "config.json")
        self.tracker = ExperimentTracker(self.directory / "metrics.csv")

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to CSV."""
        self.tracker.log(metrics, step)

    def summary(self, values: dict[str, Any]) -> Path:
        """Write JSON experiment summary."""
        path = self.directory / "summary.json"
        path.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")
        return path
