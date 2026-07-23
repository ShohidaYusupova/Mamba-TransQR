"""Factory helpers for Mamba-TransQR models."""

from __future__ import annotations

from mambatransqr.models.transqr import MambaTransQR, ModelConfig


def build_model(config: ModelConfig | None = None) -> MambaTransQR:
    """Build a Mamba-TransQR model from a configuration.

    Args:
        config: Optional model configuration. Defaults create the standard model.

    Returns:
        Initialized :class:`MambaTransQR` instance.
    """
    return MambaTransQR(config)
