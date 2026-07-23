"""Experiment lifecycle management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
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
        path.write_text(
            json.dumps(
                {**values, **_architecture_identity(self.config)},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path


def _architecture_identity(config: dict[str, Any]) -> dict[str, str | None]:
    """Extract mandatory Mamba traceability fields from experiment configuration."""
    model = config.get("model", config)
    if not isinstance(model, dict):
        model = {}
    backend = model.get("mamba_backend")
    implementation = model.get("mamba_implementation")
    if implementation is None:
        implementation = (
            "official_mamba_ssm"
            if backend == "mamba_ssm"
            else "lightweight_state_space"
            if backend == "lightweight"
            else None
        )
    package_version = model.get("mamba_ssm_version")
    if package_version is None and backend == "mamba_ssm":
        try:
            package_version = version("mamba-ssm")
        except PackageNotFoundError:
            package_version = None
    return {
        "mamba_backend": backend if isinstance(backend, str) else None,
        "mamba_implementation": (
            implementation if isinstance(implementation, str) else None
        ),
        "mamba_ssm_version": (
            package_version if isinstance(package_version, str) else None
        ),
    }
