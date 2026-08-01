"""Learning-rate scheduler factory helpers."""

from __future__ import annotations

from dataclasses import dataclass

from torch import optim


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Configuration used to construct a learning-rate scheduler."""

    name: str = "cosine"
    epochs: int = 100
    step_size: int = 30
    gamma: float = 0.1
    patience: int = 10
    warmup_epochs: int = 0
    min_lr: float = 0.0


class SchedulerFactory:
    """Create standard PyTorch learning-rate schedulers."""

    @staticmethod
    def create(
        optimizer: optim.Optimizer,
        config: SchedulerConfig | None = None,
    ) -> optim.lr_scheduler.LRScheduler | optim.lr_scheduler.ReduceLROnPlateau:
        """Create a configured scheduler.

        Args:
            optimizer: Optimizer whose learning rate is managed.
            config: Scheduler configuration.

        Returns:
            A PyTorch scheduler.
        """
        settings = config or SchedulerConfig()
        name = settings.name.lower()
        if name == "warmup_cosine":
            if not 0 < settings.warmup_epochs < settings.epochs:
                raise ValueError("warmup_epochs must be between 1 and epochs - 1")
            peak_lr = optimizer.param_groups[0]["lr"]
            if not 0.0 <= settings.min_lr < peak_lr:
                raise ValueError("min_lr must be non-negative and below peak LR")
            warmup = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / settings.warmup_epochs,
                end_factor=1.0,
                total_iters=settings.warmup_epochs,
            )
            cosine = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=settings.epochs - settings.warmup_epochs,
                eta_min=settings.min_lr,
            )
            return optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[settings.warmup_epochs],
            )
        if name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=settings.epochs
            )
        if name == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=settings.step_size,
                gamma=settings.gamma,
            )
        if name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=settings.gamma,
                patience=settings.patience,
            )
        raise ValueError(f"unsupported scheduler: {settings.name}")
