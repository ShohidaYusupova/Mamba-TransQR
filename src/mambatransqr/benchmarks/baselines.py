"""Baseline declaration helpers; implementations must be supplied explicitly."""

from mambatransqr.benchmarks.registry import BaselineRegistry, BaselineSpec


def default_registry() -> BaselineRegistry:
    """Return an empty registry; benchmark values require explicit local models."""
    return BaselineRegistry()


__all__ = ["BaselineRegistry", "BaselineSpec", "default_registry"]
