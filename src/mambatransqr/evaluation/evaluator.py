"""End-to-end model evaluation orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor, nn

from mambatransqr.evaluation.metrics import MetricsManager


class Evaluator:
    """Evaluate image-to-image or classification models over batch iterables.

    Args:
        model: Model to evaluate.
        metrics: Metric manager used for each batch.
        device: Optional evaluation device. Defaults to the model's device.
    """

    def __init__(
        self,
        model: nn.Module,
        metrics: MetricsManager | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the evaluator."""
        self.model = model
        self.metrics = metrics or MetricsManager()
        parameter = next(model.parameters(), None)
        self.device = device or (
            parameter.device if parameter is not None else torch.device("cpu")
        )
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Any]) -> dict[str, float]:
        """Evaluate a model over all batches.

        Args:
            loader: Iterable yielding ``(inputs, targets)`` pairs or mappings
                with ``image`` and ``target`` entries.

        Returns:
            Mean metric mapping.
        """
        self.model.eval()
        totals: dict[str, float] = {}
        batches = 0
        for batches, batch in enumerate(loader, start=1):
            inputs, targets = self._unpack(batch)
            values = self.metrics.compute(self.model(inputs), targets)
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value
        if batches == 0:
            raise ValueError("evaluation loader yielded no batches")
        return {name: value / batches for name, value in totals.items()}

    def predict(self, inputs: Tensor) -> Tensor:
        """Run model inference for a tensor batch."""
        self.model.eval()
        with torch.no_grad():
            return self.model(inputs.to(self.device))

    def _unpack(self, batch: Any) -> tuple[Tensor, Tensor]:
        """Validate and move a supported batch representation to the device."""
        if isinstance(batch, Mapping):
            inputs, targets = batch.get("image"), batch.get("target")
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        else:
            raise TypeError("batch must be a mapping or a pair of tensors")
        if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
            raise TypeError("batch inputs and targets must be tensors")
        return inputs.to(self.device), targets.to(self.device)
