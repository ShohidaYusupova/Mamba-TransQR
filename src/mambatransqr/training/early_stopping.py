"""Early-stopping callback."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mambatransqr.training.callbacks import Callback

if TYPE_CHECKING:
    from mambatransqr.training.trainer import Trainer


class EarlyStopping(Callback):
    """Stop training when a monitored metric no longer improves.

    Args:
        monitor: Metric name to monitor.
        patience: Consecutive non-improving epochs to tolerate.
        mode: ``min`` for losses or ``max`` for scores.
        min_delta: Minimum change considered an improvement.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        """Initialize early-stopping state."""
        if patience < 0:
            raise ValueError("patience must be non-negative")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.wait = 0

    def on_epoch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        """Update patience and halt the trainer when appropriate."""
        if self.monitor not in metrics:
            return
        value = metrics[self.monitor]
        if self.best is None or self._improved(value):
            self.best = value
            self.wait = 0
            return
        self.wait += 1
        if self.wait > self.patience:
            trainer.state.stopped_early = True

    def _improved(self, value: float) -> bool:
        """Return whether a new monitored value is an improvement."""
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta
