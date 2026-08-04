"""Checkpoint resume selection, identity validation, and RNG state helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


class IncompatibleCheckpointError(ValueError):
    """Raised before mutation when a checkpoint belongs to another run."""


def resolve_resume_checkpoint(
    resume: str | Path, checkpoint_dir: str | Path
) -> Path | None:
    """Resolve ``auto``, ``never``, or an explicit checkpoint path."""
    value = str(resume)
    if value == "never":
        return None
    if value == "auto":
        latest = Path(checkpoint_dir) / "latest.pt"
        return latest if latest.is_file() else None
    candidate = Path(value)
    if not candidate.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {candidate}")
    return candidate


def validate_resume_identity(
    payload: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Refuse resume unless every required identity field matches exactly."""
    actual = payload.get("resume_identity")
    if not isinstance(actual, dict):
        raise IncompatibleCheckpointError("checkpoint has no resume_identity")
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: expected {value['expected']!r}, got {value['actual']!r}"
            for key, value in mismatches.items()
        )
        raise IncompatibleCheckpointError(f"incompatible resume checkpoint ({details})")


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and all CUDA RNG states."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore all RNG states captured by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
