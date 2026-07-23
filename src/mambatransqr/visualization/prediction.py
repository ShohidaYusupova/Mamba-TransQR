"""Prediction comparison plots."""
from pathlib import Path
from torch import Tensor
from mambatransqr.visualization.plotter import pyplot, save
def plot_prediction(inputs: Tensor, predictions: Tensor, targets: Tensor, path: str | Path) -> Path:
    """Plot one input, prediction, and target triplet."""
    figure, axes = pyplot().subplots(1, 3)
    for axis, image, title in zip(axes, (inputs, predictions, targets), ("Input", "Prediction", "Target"), strict=True): axis.imshow(image.detach().cpu().clamp(0, 1).permute(1, 2, 0)); axis.set_title(title); axis.axis("off")
    return save(figure, path)
