"""Shared neural-network layers and model utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn


class DropPath(nn.Module):
    """Stochastic depth applied independently to each batch item.

    Args:
        drop_prob: Probability of dropping a residual branch during training.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        """Initialize the stochastic-depth layer."""
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob = drop_prob

    def forward(self, inputs: Tensor) -> Tensor:
        """Drop complete residual paths during training."""
        if self.drop_prob == 0.0 or not self.training:
            return inputs
        keep_prob = 1.0 - self.drop_prob
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=inputs.dtype, device=inputs.device
        )
        random_tensor.floor_()
        return inputs.div(keep_prob) * random_tensor


def get_activation(name: str) -> Callable[[], nn.Module]:
    """Return an activation-layer factory.

    Args:
        name: One of ``gelu``, ``relu``, ``silu``, or ``leaky_relu``.

    Returns:
        Callable that creates the requested activation layer.

    Raises:
        ValueError: If the activation name is unsupported.
    """
    activations: dict[str, Callable[[], nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
    }
    try:
        return activations[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported activation: {name}") from error


def initialize_weights(module: nn.Module) -> None:
    """Apply transformer-friendly parameter initialization.

    Args:
        module: Module visited by :meth:`torch.nn.Module.apply`.
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


@dataclass(frozen=True, slots=True)
class ModelSummary:
    """Compact parameter summary for a model.

    Attributes:
        total_parameters: Number of all parameters.
        trainable_parameters: Number of parameters requiring gradients.
    """

    total_parameters: int
    trainable_parameters: int

    def __str__(self) -> str:
        """Format a readable summary."""
        return (
            "ModelSummary("
            f"total_parameters={self.total_parameters:,}, "
            f"trainable_parameters={self.trainable_parameters:,})"
        )


def summarize_model(model: nn.Module) -> ModelSummary:
    """Return a parameter-count summary for a PyTorch model.

    Args:
        model: Model to inspect.

    Returns:
        Total and trainable parameter counts.
    """
    parameters = tuple(model.parameters())
    return ModelSummary(
        total_parameters=sum(parameter.numel() for parameter in parameters),
        trainable_parameters=sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
    )
