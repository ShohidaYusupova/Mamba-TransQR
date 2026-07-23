"""Optimizer factory helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from torch import Tensor, nn, optim


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Configuration used to construct an optimizer."""

    name: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9


class OptimizerFactory:
    """Create standard PyTorch optimizers from configuration."""

    @staticmethod
    def create(
        parameters: Iterable[Tensor] | Iterable[nn.Parameter],
        config: OptimizerConfig | None = None,
    ) -> optim.Optimizer:
        """Create a configured optimizer.

        Args:
            parameters: Trainable model parameters.
            config: Optimizer configuration.

        Returns:
            Constructed PyTorch optimizer.

        Raises:
            ValueError: If the optimizer name or learning rate is invalid.
        """
        settings = config or OptimizerConfig()
        if settings.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        name = settings.name.lower()
        if name == "adamw":
            return optim.AdamW(
                parameters,
                lr=settings.learning_rate,
                weight_decay=settings.weight_decay,
            )
        if name == "adam":
            return optim.Adam(
                parameters,
                lr=settings.learning_rate,
                weight_decay=settings.weight_decay,
            )
        if name == "sgd":
            return optim.SGD(
                parameters,
                lr=settings.learning_rate,
                momentum=settings.momentum,
                weight_decay=settings.weight_decay,
            )
        raise ValueError(f"unsupported optimizer: {settings.name}")
