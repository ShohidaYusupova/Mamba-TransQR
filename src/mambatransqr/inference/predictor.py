"""Single-image model inference."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn

from mambatransqr.evaluation.decoder import QRDecoderEvaluator
from mambatransqr.inference.postprocessing import tensor_to_image
from mambatransqr.inference.preprocessing import load_image, preprocess_image
from mambatransqr.inference.result import PredictionResult
from mambatransqr.inference.runtime import autocast_context, select_device
from mambatransqr.models import MambaTransQR, ModelConfig, build_model


class Predictor:
    """Load a Mamba-TransQR checkpoint and restore individual images.

    Args:
        model: Optional already-created model.
        config: Model configuration used when constructing a model.
        checkpoint: Optional checkpoint containing ``model`` or raw state dict.
        device: CPU, CUDA, or ``auto``.
        mixed_precision: Enable CUDA automatic mixed precision.
        decoder_backend: Optional QR decoder backend.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        config: ModelConfig | None = None,
        *,
        checkpoint: str | Path | None = None,
        device: str = "auto",
        mixed_precision: bool = True,
        decoder_backend: str | None = None,
    ) -> None:
        """Initialize the model and optional checkpoint state."""
        self.device = select_device(device)
        self.model = (model or build_model(config)).to(self.device).eval()
        model_config = getattr(self.model, "config", None)
        self.config = config or (
            model_config if isinstance(model_config, ModelConfig) else ModelConfig()
        )
        self.mixed_precision = mixed_precision
        self.decoder = QRDecoderEvaluator(decoder_backend) if decoder_backend else None
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model parameters from a Trainer or raw state-dict checkpoint."""
        payload: Any = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_path(self, path: str | Path) -> tuple[Image.Image, PredictionResult]:
        """Load and restore an image from disk."""
        image = load_image(path)
        restored, result = self.predict_image(image)
        result.source_path = str(path)
        return restored, result

    def predict_image(self, image: Image.Image) -> tuple[Image.Image, PredictionResult]:
        """Restore one PIL image and return timing and optional decode metadata."""
        inputs = preprocess_image(image, self.config.image_size).to(self.device)
        _synchronize(self.device)
        started = perf_counter()
        with torch.no_grad(), autocast_context(self.device, self.mixed_precision):
            output: Tensor = self.model(inputs)
        _synchronize(self.device)
        restored = tensor_to_image(output)
        decoded = self.decoder.decode(restored) if self.decoder is not None else None
        return restored, PredictionResult(
            decoded_text=decoded.value if decoded is not None else None,
            latency_ms=(perf_counter() - started) * 1_000,
            metadata={"device": str(self.device)},
        )


def _synchronize(device: torch.device) -> None:
    """Synchronize CUDA work before time measurement."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
