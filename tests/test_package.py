"""Smoke tests for the public package interface."""

from mambatransqr import __version__


def test_version_is_defined() -> None:
    """The package exposes a non-empty version string."""
    assert __version__ == "0.1.0"
