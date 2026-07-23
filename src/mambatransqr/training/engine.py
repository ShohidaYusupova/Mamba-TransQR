"""Low-level train and evaluation epoch execution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor, nn, optim

from mambatransqr.training.amp import AMPManager
from mambatransqr.training.ema import ExponentialMovingAverage
from mambatransqr.training.loss_manager import LossManager


class TrainingEngine:
    """Execute train and validation epochs for an image-to-image model.

    Args:
        model: Model receiving image batches.
        optimizer: Optimizer updated during training.
        loss_manager: Loss-combination helper.
        device: Device where model and tensors are placed.
        gradient_accumulation: Number of batches per optimizer update.
        gradient_clip_norm: Optional maximum global gradient norm.
        amp: AMP manager controlling mixed precision.
        ema: Optional moving average updated after optimizer steps.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_manager: LossManager,
        device: torch.device,
        *,
        gradient_accumulation: int = 1,
        gradient_clip_norm: float | None = None,
        amp: AMPManager | None = None,
        ema: ExponentialMovingAverage | None = None,
    ) -> None:
        """Initialize the epoch runner."""
        if gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be positive")
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        self.model = model
        self.optimizer = optimizer
        self.loss_manager = loss_manager
        self.device = device
        self.gradient_accumulation = gradient_accumulation
        self.gradient_clip_norm = gradient_clip_norm
        self.amp = amp or AMPManager(device, enabled=False)
        self.ema = ema

    def train_epoch(self, loader: Iterable[Any]) -> tuple[dict[str, float], int]:
        """Run one optimization epoch.

        Args:
            loader: Iterable yielding mappings with ``image`` and ``target``.

        Returns:
            Mean metrics and number of optimizer updates made.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        aggregates: dict[str, float] = {}
        batch_count = 0
        updates = 0
        for batch_count, batch in enumerate(loader, start=1):
            images, targets = self._unpack(batch)
            with self.amp.autocast():
                predictions = self.model(images)
                loss, metrics = self.loss_manager(predictions, targets)
                scaled_loss = loss / self.gradient_accumulation
            self.amp.backward(scaled_loss)
            self._accumulate(aggregates, metrics)
            if batch_count % self.gradient_accumulation == 0:
                self._optimizer_step()
                updates += 1
        if batch_count == 0:
            raise ValueError("training loader yielded no batches")
        if batch_count % self.gradient_accumulation:
            self._optimizer_step()
            updates += 1
        return self._average(aggregates, batch_count), updates

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Any]) -> dict[str, float]:
        """Run one evaluation epoch without gradient updates."""
        self.model.eval()
        aggregates: dict[str, float] = {}
        batch_count = 0
        for batch_count, batch in enumerate(loader, start=1):
            images, targets = self._unpack(batch)
            with self.amp.autocast():
                predictions = self.model(images)
                _, metrics = self.loss_manager(predictions, targets)
            self._accumulate(aggregates, metrics)
        if batch_count == 0:
            raise ValueError("evaluation loader yielded no batches")
        return self._average(aggregates, batch_count)

    def _optimizer_step(self) -> None:
        """Clip gradients, apply an update, and refresh EMA state."""
        if self.gradient_clip_norm is not None:
            self.amp.unscale(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        self.amp.step(self.optimizer)
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema is not None:
            self.ema.update(self.model)

    def _unpack(self, batch: Any) -> tuple[Tensor, Tensor]:
        """Move a supported batch representation onto the training device."""
        if isinstance(batch, Mapping):
            if "image" not in batch or "target" not in batch:
                raise KeyError("mapping batches must contain 'image' and 'target'")
            images, targets = batch["image"], batch["target"]
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            images, targets = batch[0], batch[1]
        else:
            raise TypeError("batch must be a mapping or a pair of tensors")
        if not isinstance(images, Tensor) or not isinstance(targets, Tensor):
            raise TypeError("image and target batch values must be tensors")
        return images.to(self.device, non_blocking=True), targets.to(
            self.device, non_blocking=True
        )

    @staticmethod
    def _accumulate(aggregates: dict[str, float], metrics: dict[str, float]) -> None:
        """Add batch metrics into an aggregate mapping."""
        for name, value in metrics.items():
            aggregates[name] = aggregates.get(name, 0.0) + value

    @staticmethod
    def _average(aggregates: dict[str, float], count: int) -> dict[str, float]:
        """Return mean metrics from summed values."""
        return {name: value / count for name, value in aggregates.items()}
