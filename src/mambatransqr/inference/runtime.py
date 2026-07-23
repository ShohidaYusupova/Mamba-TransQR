"""Runtime device and mixed-precision helpers."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch


def select_device(requested: str = "auto") -> torch.device:
    """Select a CPU or CUDA device.

    Args:
        requested: ``auto``, ``cpu``, or a CUDA device specifier.

    Returns:
        Available PyTorch device.
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def autocast_context(device: torch.device, enabled: bool = True) -> AbstractContextManager[None]:
    """Return an AMP context when CUDA mixed precision is available."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()
