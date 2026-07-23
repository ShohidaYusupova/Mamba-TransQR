"""Metadata-aware QR restoration trainer."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor, nn, optim

from mambatransqr.losses.multiscale import MultiScaleRestorationLoss
from mambatransqr.training.loss_manager import LossManager
from mambatransqr.training.trainer import Trainer, TrainerConfig


class QRRestorationTrainer(Trainer):
    """Train paired QR restoration models with metadata-aware QR objectives."""

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, objective: MultiScaleRestorationLoss, config: TrainerConfig | None = None, **kwargs: Any) -> None:
        self.objective = objective
        super().__init__(model, optimizer, LossManager({"qr": objective}), config, **kwargs)

    def train_batch(self, batch: Mapping[str, Any]) -> dict[str, float]:
        """Run one AMP-compatible paired metadata-aware optimization step."""
        inputs, targets, metadata = self._batch(batch)
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        with self.engine.amp.autocast():
            output = self.model(inputs); loss = self.objective(output, targets, metadata)
        self.engine.amp.backward(loss); self.engine.amp.step(self.optimizer)
        if self.ema is not None: self.ema.update(self.model)
        return {"loss": float(loss.detach()), **{name: float(value.detach()) for name, value in self.objective.last_components.items()}}

    @torch.no_grad()
    def validate_qr(self, loader: Iterable[Mapping[str, Any]]) -> dict[str, float]:
        """Return validation loss, PSNR, SSIM proxy, and threshold decode proxy rate."""
        self.model.eval(); totals = {"val_loss": 0.0, "val_psnr": 0.0, "val_ssim": 0.0, "val_decode_rate": 0.0}; count = 0
        for batch in loader:
            inputs, targets, metadata = self._batch(batch); output = self.model(inputs); mse = torch.nn.functional.mse_loss(output, targets)
            totals["val_loss"] += float(self.objective(output, targets, metadata)); totals["val_psnr"] += float((-10 * torch.log10(mse.clamp_min(1e-8))))
            totals["val_ssim"] += float(1 - self.objective.ssim(output, targets)); totals["val_decode_rate"] += float(((output - 0.5).abs() > 0.25).float().mean()); count += 1
        return {name: value / max(count, 1) for name, value in totals.items()}

    def _batch(self, batch: Mapping[str, Any]) -> tuple[Tensor, Tensor, list[dict[str, Any]] | None]:
        inputs, targets = batch.get("image"), batch.get("target")
        if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor): raise TypeError("QR batches require tensor image and target fields")
        metadata = batch.get("metadata"); return inputs.to(self.device), targets.to(self.device), metadata if isinstance(metadata, list) else None
