"""Optional Matplotlib evaluation visualizations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from torch import Tensor


class EvaluationVisualizer:
    """Create plots for evaluation metrics and model predictions.

    Matplotlib is imported lazily so headless evaluation does not require it.
    """

    @staticmethod
    def metric_curves(history: Mapping[str, Sequence[float]], path: str | Path) -> Path:
        """Plot one curve per metric history sequence."""
        pyplot = _pyplot()
        figure, axis = pyplot.subplots()
        for name, values in history.items():
            axis.plot(values, label=name)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Value")
        axis.legend()
        return _save(figure, path)

    @staticmethod
    def training_history(
        records: Sequence[Mapping[str, float]], path: str | Path
    ) -> Path:
        """Plot metrics extracted from trainer history records."""
        history: dict[str, list[float]] = {}
        for record in records:
            for name, value in record.items():
                if name != "epoch":
                    history.setdefault(name, []).append(value)
        return EvaluationVisualizer.metric_curves(history, path)

    @staticmethod
    def confusion_matrix(matrix: Tensor, path: str | Path) -> Path:
        """Plot a confusion matrix heatmap."""
        pyplot = _pyplot()
        figure, axis = pyplot.subplots()
        image = axis.imshow(matrix.detach().cpu().numpy(), cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        return _save(figure, path)

    @staticmethod
    def roc_curve(
        false_positive_rate: Tensor, true_positive_rate: Tensor, path: str | Path
    ) -> Path:
        """Plot a ROC curve."""
        pyplot = _pyplot()
        figure, axis = pyplot.subplots()
        axis.plot(false_positive_rate.cpu(), true_positive_rate.cpu(), label="ROC")
        axis.plot([0, 1], [0, 1], "--", color="gray")
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.legend()
        return _save(figure, path)

    @staticmethod
    def sample_predictions(
        inputs: Tensor,
        predictions: Tensor,
        targets: Tensor,
        path: str | Path,
        count: int = 4,
    ) -> Path:
        """Plot input, prediction, and target triplets for sample images."""
        if count < 1:
            raise ValueError("count must be positive")
        pyplot = _pyplot()
        samples = min(count, inputs.shape[0], predictions.shape[0], targets.shape[0])
        figure, axes = pyplot.subplots(
            samples, 3, squeeze=False, figsize=(9, 3 * samples)
        )
        for index in range(samples):
            for axis, image, title in zip(
                axes[index],
                (inputs[index], predictions[index], targets[index]),
                ("Input", "Prediction", "Target"),
                strict=True,
            ):
                axis.imshow(_image_array(image))
                axis.set_title(title)
                axis.axis("off")
        return _save(figure, path)


def _pyplot() -> Any:
    """Import Matplotlib pyplot with a clear optional-dependency message."""
    try:
        import matplotlib.pyplot as pyplot
    except ImportError as error:
        raise ImportError(
            "Visualization requires the optional matplotlib package."
        ) from error
    return pyplot


def _save(figure: Any, path: str | Path) -> Path:
    """Save a Matplotlib figure to a prepared path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    _pyplot().close(figure)
    return destination


def _image_array(image: Tensor) -> Tensor:
    """Convert CHW image tensors to HWC tensors suitable for Matplotlib."""
    value = image.detach().cpu().float().clamp(0, 1)
    return value.permute(1, 2, 0) if value.ndim == 3 else value
