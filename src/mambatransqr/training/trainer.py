"""High-level training orchestration."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from mambatransqr.training.amp import AMPManager
from mambatransqr.training.callbacks import Callback, CallbackList
from mambatransqr.training.checkpoint import CheckpointManager, Stateful
from mambatransqr.training.ema import ExponentialMovingAverage
from mambatransqr.training.engine import TrainingEngine
from mambatransqr.training.history import TrainingHistory
from mambatransqr.training.logger import TrainingLogger
from mambatransqr.training.loss_manager import LossManager
from mambatransqr.training.state import TrainingState


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Configuration controlling a training run."""

    epochs: int = 100
    device: str = "auto"
    seed: int = 42
    gradient_accumulation: int = 1
    gradient_clip_norm: float | None = 1.0
    mixed_precision: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    multi_gpu: bool = False
    checkpoint_dir: str = "checkpoints"
    monitor: str = "val_loss"
    monitor_mode: str = "min"


class Trainer:
    """Coordinate model optimization, validation, logging, and checkpoints.

    Args:
        model: Model to optimize.
        optimizer: Optimizer bound to model parameters.
        loss_manager: Objective function manager.
        config: Training behavior configuration.
        scheduler: Optional learning-rate scheduler.
        callbacks: Optional lifecycle callbacks.
        loggers: Optional metric loggers.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_manager: LossManager,
        config: TrainerConfig | None = None,
        *,
        scheduler: Stateful | None = None,
        callbacks: list[Callback] | None = None,
        loggers: list[TrainingLogger] | None = None,
    ) -> None:
        """Initialize devices and training components."""
        self.config = config or TrainerConfig()
        if self.config.epochs < 1:
            raise ValueError("epochs must be positive")
        self.set_seed(self.config.seed)
        self.device = self.select_device(self.config.device)
        self.model = model.to(self.device)
        if self.config.multi_gpu and self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.state = TrainingState()
        self.history = TrainingHistory()
        self.callbacks = CallbackList(callbacks)
        self.loggers = loggers or []
        self.ema = (
            ExponentialMovingAverage(self.model, self.config.ema_decay)
            if self.config.use_ema
            else None
        )
        self.engine = TrainingEngine(
            self.model,
            optimizer,
            loss_manager,
            self.device,
            gradient_accumulation=self.config.gradient_accumulation,
            gradient_clip_norm=self.config.gradient_clip_norm,
            amp=AMPManager(self.device, self.config.mixed_precision),
            ema=self.ema,
        )
        self.checkpoints = CheckpointManager(
            Path(self.config.checkpoint_dir),
            monitor=self.config.monitor,
            mode=self.config.monitor_mode,
        )

    def fit(
        self,
        train_loader: Iterable[Any],
        validation_loader: Iterable[Any] | None = None,
        *,
        resume_from: str | Path | None = None,
    ) -> TrainingHistory:
        """Train through configured epochs, saving latest and best checkpoints.

        Args:
            train_loader: Iterable yielding train batches.
            validation_loader: Optional iterable yielding validation batches.
            resume_from: Optional checkpoint path to restore before training.

        Returns:
            Complete training metric history.
        """
        if resume_from is not None:
            self.resume(resume_from)
        self.callbacks.on_fit_start(self)
        for epoch in range(self.state.epoch + 1, self.config.epochs + 1):
            train_metrics, updates = self.engine.train_epoch(train_loader)
            self.state.global_step += updates
            metrics = {f"train_{key}": value for key, value in train_metrics.items()}
            if validation_loader is not None:
                if self.ema is not None:
                    self.ema.apply_to(self.model)
                validation_metrics = self.engine.evaluate(validation_loader)
                if self.ema is not None:
                    self.ema.restore(self.model)
                metrics.update({f"val_{key}": value for key, value in validation_metrics.items()})
            self._step_scheduler(metrics)
            self.state.epoch = epoch
            self.state.metrics = metrics
            is_best = self._is_best(metrics)
            if is_best:
                self.state.best_metric = metrics[self.config.monitor]
                self.state.best_epoch = epoch
            self.history.append(epoch, metrics)
            for logger in self.loggers:
                logger.log_metrics(metrics, self.state.global_step)
            self.checkpoints.save(
                self.model,
                self.optimizer,
                self.state,
                scheduler=self.scheduler,
                ema_state=self.ema.state_dict() if self.ema is not None else None,
                is_best=is_best,
            )
            self.callbacks.on_epoch_end(self, metrics)
            if self.state.stopped_early:
                break
        self.callbacks.on_fit_end(self)
        for logger in self.loggers:
            logger.close()
        return self.history

    def resume(self, path: str | Path) -> None:
        """Restore a saved checkpoint into the current training run."""
        self.state, ema_state = self.checkpoints.load(
            path,
            self.model,
            self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )
        if self.ema is not None and ema_state is not None:
            self.ema.load_state_dict(ema_state)

    @staticmethod
    def select_device(requested: str = "auto") -> torch.device:
        """Select an available compute device.

        Args:
            requested: ``auto``, ``cuda``, ``mps``, or ``cpu``.

        Returns:
            Selected PyTorch device.
        """
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device

    @staticmethod
    def set_seed(seed: int) -> None:
        """Seed Python, NumPy, and PyTorch random generators."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _is_best(self, metrics: dict[str, float]) -> bool:
        """Return whether current monitored metrics improve the checkpoint."""
        if self.config.monitor not in metrics:
            return False
        current = metrics[self.config.monitor]
        if self.state.best_metric is None:
            return True
        if self.config.monitor_mode == "min":
            return current < self.state.best_metric
        return current > self.state.best_metric

    def _step_scheduler(self, metrics: dict[str, float]) -> None:
        """Step regular or metric-driven schedulers."""
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            value = metrics.get(self.config.monitor)
            if value is not None:
                self.scheduler.step(value)
            return
        self.scheduler.step()
