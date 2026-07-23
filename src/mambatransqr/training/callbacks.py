"""Callback interfaces for training lifecycle events."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mambatransqr.training.trainer import Trainer


class Callback:
    """Base class for optional training lifecycle behavior."""

    def on_fit_start(self, trainer: Trainer) -> None:
        """Run before the first training epoch."""

    def on_epoch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        """Run after each completed training epoch."""

    def on_fit_end(self, trainer: Trainer) -> None:
        """Run after training finishes."""


class CallbackList:
    """Dispatch lifecycle calls to a collection of callbacks."""

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        """Store callbacks in invocation order."""
        self.callbacks = callbacks or []

    def on_fit_start(self, trainer: Trainer) -> None:
        """Dispatch the fit-start event."""
        for callback in self.callbacks:
            callback.on_fit_start(trainer)

    def on_epoch_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        """Dispatch the epoch-end event."""
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, metrics)

    def on_fit_end(self, trainer: Trainer) -> None:
        """Dispatch the fit-end event."""
        for callback in self.callbacks:
            callback.on_fit_end(trainer)
