"""Feature-map visualizations."""

from pathlib import Path

from torch import Tensor

from mambatransqr.visualization.plotter import pyplot, save


def plot_feature_maps(features: Tensor, path: str | Path, count: int = 8) -> Path:
    """Plot a selection of feature maps from the first batch item."""
    shown = min(count, features.shape[1])
    figure, axes = pyplot().subplots(1, shown, squeeze=False)
    for index in range(shown):
        axes[0, index].imshow(features[0, index].detach().cpu(), cmap="viridis")
        axes[0, index].axis("off")
    return save(figure, path)
