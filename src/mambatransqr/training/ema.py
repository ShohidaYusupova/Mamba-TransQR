"""Exponential moving average of model parameters."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


class ExponentialMovingAverage:
    """Maintain a moving-average copy of trainable model parameters.

    Args:
        model: Model whose parameters are tracked.
        decay: EMA decay in the half-open interval [0, 1).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        """Initialize shadow parameter tensors."""
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup: dict[str, Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters after an optimizer step."""
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        """Swap averaged parameters into a model for evaluation."""
        self.backup = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore parameters saved by :meth:`apply_to`."""
        for name, parameter in model.named_parameters():
            if name in self.backup:
                parameter.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, object]:
        """Return serializable EMA state."""
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Load EMA state from a checkpoint."""
        self.decay = float(state["decay"])
        shadow = state["shadow"]
        if not isinstance(shadow, dict):
            raise ValueError("invalid EMA shadow state")
        self.shadow = {
            str(name): value
            for name, value in shadow.items()
            if isinstance(value, Tensor)
        }
