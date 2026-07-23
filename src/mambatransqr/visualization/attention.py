"""Attention-map visualizations."""

from pathlib import Path

from torch import Tensor

from mambatransqr.visualization.plotter import pyplot, save


def plot_attention(attention: Tensor, path: str | Path) -> Path:
    """Plot an attention matrix heatmap."""
    figure, axis = pyplot().subplots()
    axis.imshow(attention.detach().cpu(), cmap="magma")
    return save(figure, path)
