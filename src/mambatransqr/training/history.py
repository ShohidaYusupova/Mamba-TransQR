"""Training metric history."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TrainingHistory:
    """Store scalar metrics emitted during a training run."""

    records: list[dict[str, float]] = field(default_factory=list)

    def append(self, epoch: int, metrics: dict[str, float]) -> None:
        """Append metrics for one completed epoch.

        Args:
            epoch: One-based epoch number.
            metrics: Scalar metrics for the epoch.
        """
        self.records.append({"epoch": float(epoch), **metrics})

    def values(self, name: str) -> list[float]:
        """Return all recorded values for a metric name."""
        return [record[name] for record in self.records if name in record]

    def latest(self) -> dict[str, float]:
        """Return the most recent record, or an empty mapping."""
        return dict(self.records[-1]) if self.records else {}
