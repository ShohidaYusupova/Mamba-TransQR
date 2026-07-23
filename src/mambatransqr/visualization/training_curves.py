"""Training curve plots."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from mambatransqr.visualization.plotter import pyplot, save


def plot_training_curves(
    history: Mapping[str, Sequence[float]], path: str | Path
) -> Path:
    """Plot named metric sequences."""
    figure, axis = pyplot().subplots()
    for name, values in history.items():
        axis.plot(values, label=name)
    axis.legend()
    axis.set_xlabel("Epoch")
    return save(figure, path)
