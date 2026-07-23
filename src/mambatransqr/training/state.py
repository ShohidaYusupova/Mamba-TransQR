"""Serializable training state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TrainingState:
    """Mutable state required to resume a training run.

    Attributes:
        epoch: Completed epoch count.
        global_step: Number of optimizer updates.
        best_metric: Best monitored metric seen so far.
        best_epoch: Epoch at which the best metric was recorded.
        stopped_early: Whether an early-stopping callback halted training.
        metrics: Most recent scalar metrics.
    """

    epoch: int = 0
    global_step: int = 0
    best_metric: float | None = None
    best_epoch: int = 0
    stopped_early: bool = False
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to a checkpoint-safe mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrainingState:
        """Create state from a checkpoint mapping.

        Args:
            values: Serialized state values.

        Returns:
            Restored training state.
        """
        return cls(
            epoch=int(values.get("epoch", 0)),
            global_step=int(values.get("global_step", 0)),
            best_metric=(
                float(values["best_metric"])
                if values.get("best_metric") is not None
                else None
            ),
            best_epoch=int(values.get("best_epoch", 0)),
            stopped_early=bool(values.get("stopped_early", False)),
            metrics={
                str(key): float(value)
                for key, value in values.get("metrics", {}).items()
            },
        )
