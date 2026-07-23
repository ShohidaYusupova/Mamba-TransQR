"""Model checkpoint persistence and restoration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn, optim

from mambatransqr.training.state import TrainingState


class Stateful(Protocol):
    """Protocol for checkpointable optimizer-adjacent state."""

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state."""

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore serializable state."""


class CheckpointManager:
    """Save and restore latest and best model checkpoints.

    Args:
        directory: Directory where checkpoint files are stored.
        monitor: Metric used to determine the best checkpoint.
        mode: ``min`` for losses or ``max`` for scores.
    """

    def __init__(
        self,
        directory: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        """Initialize checkpoint destination and comparison behavior."""
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode

    @property
    def latest_path(self) -> Path:
        """Return the standard latest-checkpoint path."""
        return self.directory / "latest.pt"

    @property
    def best_path(self) -> Path:
        """Return the standard best-checkpoint path."""
        return self.directory / "best.pt"

    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        state: TrainingState,
        *,
        scheduler: Stateful | None = None,
        ema_state: dict[str, object] | None = None,
        is_best: bool = False,
    ) -> None:
        """Persist latest state and optionally update the best checkpoint."""
        architecture = _architecture_identity(model)
        payload: dict[str, Any] = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": state.to_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "ema": ema_state,
            "architecture": architecture,
            **architecture,
        }
        self._atomic_save(payload, self.latest_path)
        if is_best:
            self._atomic_save(payload, self.best_path)

    def load(
        self,
        path: str | Path,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        *,
        scheduler: Stateful | None = None,
        map_location: str | torch.device = "cpu",
    ) -> tuple[TrainingState, dict[str, object] | None]:
        """Restore checkpoint content into supplied training components.

        Args:
            path: Checkpoint to restore.
            model: Model receiving stored parameters.
            optimizer: Optional optimizer receiving stored state.
            scheduler: Optional scheduler receiving stored state.
            map_location: Device mapping passed to :func:`torch.load`.

        Returns:
            Restored training state and optional EMA state.
        """
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        ema = payload.get("ema")
        return TrainingState.from_dict(payload["state"]), ema if isinstance(
            ema, dict
        ) else None

    def _atomic_save(self, payload: dict[str, Any], path: Path) -> None:
        """Write a checkpoint atomically to reduce corruption risk."""
        temporary_path = path.with_suffix(".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(path)


def _architecture_identity(model: nn.Module) -> dict[str, str | None]:
    """Extract backend traceability metadata without constraining generic models."""
    identity = getattr(model, "architecture_identity", None)
    if callable(identity):
        value = identity()
        if isinstance(value, dict):
            return {
                "mamba_backend": value.get("mamba_backend"),
                "mamba_implementation": value.get("mamba_implementation"),
                "mamba_ssm_version": value.get("mamba_ssm_version"),
            }
    return {
        "mamba_backend": None,
        "mamba_implementation": None,
        "mamba_ssm_version": None,
    }
