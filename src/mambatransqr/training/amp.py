"""Automatic mixed-precision support."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch import Tensor, optim


class AMPManager:
    """Manage autocasting and gradient scaling for a selected device.

    Args:
        device: Target training device.
        enabled: Whether AMP should be enabled when supported.
    """

    def __init__(self, device: torch.device, enabled: bool = True) -> None:
        """Initialize AMP state."""
        self.device = device
        self.enabled = enabled and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled)

    def autocast(self) -> AbstractContextManager[None]:
        """Return the appropriate autocast context manager."""
        if not self.enabled:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.float16)

    def backward(self, loss: Tensor) -> None:
        """Backpropagate a scaled or full-precision loss."""
        self.scaler.scale(loss).backward()

    def unscale(self, optimizer: optim.Optimizer) -> None:
        """Unscale gradients before clipping them."""
        self.scaler.unscale_(optimizer)

    def step(self, optimizer: optim.Optimizer) -> None:
        """Perform an optimizer update and refresh scaling."""
        self.scaler.step(optimizer)
        self.scaler.update()

    def state_dict(self) -> dict[str, Any]:
        """Return GradScaler state, including an empty disabled scaler state."""
        return self.scaler.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore GradScaler state."""
        self.scaler.load_state_dict(state)
