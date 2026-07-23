"""Reproducibility metadata for QR restoration recipes."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch


def reproducibility_metadata(
    config: dict[str, Any],
    manifest: str | Path | None = None,
    model: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Capture hashes, dependency versions, hardware, Git, and Mamba identity."""
    manifest_hash = _hash_file(manifest) if manifest else None
    identity: dict[str, object] = getattr(model, "architecture_identity", lambda: {})()
    return {
        "seed": config.get("seed"),
        "deterministic": config.get("deterministic", False),
        "config_hash": hashlib.sha256(
            repr(sorted(config.items())).encode()
        ).hexdigest(),
        "dataset_manifest_hash": manifest_hash,
        "git_commit": _git_commit(),
        "mamba_backend": identity.get("mamba_backend"),
        "mamba_implementation": identity.get("mamba_implementation"),
        "mamba_ssm_version": identity.get("mamba_ssm_version"),
        "dependencies": {
            name: _version(name) for name in ("torch", "mamba-ssm", "qrcode", "Pillow")
        },
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cuda": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }


def _hash_file(path: str | Path) -> str | None:
    candidate = Path(path)
    return (
        hashlib.sha256(candidate.read_bytes()).hexdigest()
        if candidate.is_file()
        else None
    )


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
