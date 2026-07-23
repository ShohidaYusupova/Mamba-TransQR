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
        if name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)
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
