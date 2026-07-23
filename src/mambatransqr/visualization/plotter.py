"""Lazy Matplotlib plotting helpers."""

from pathlib import Path
from typing import Any


def pyplot() -> Any:
    """Return pyplot or raise a clear dependency error."""
    try:
        import matplotlib.pyplot as value
    except ImportError as error:
        raise ImportError("Visualization requires matplotlib.") from error
    return value


def save(figure: Any, path: str | Path) -> Path:
    """Persist a figure and return its path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    pyplot().close(figure)
    return destination
