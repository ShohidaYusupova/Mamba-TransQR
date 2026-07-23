"""Unit tests for core inference helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from mambatransqr.inference import (  # noqa: E402
    InferencePipeline,
    PredictionResult,
    Predictor,
    export_torchscript,
)
from mambatransqr.inference.preprocessing import preprocess_image  # noqa: E402
from mambatransqr.models import ModelConfig  # noqa: E402


def test_preprocessing_creates_normalized_batch_tensor() -> None:
    """PIL preprocessing returns a normalized BCHW tensor."""
    tensor = preprocess_image(Image.new("RGB", (10, 10), "white"), 8)
    assert tensor.shape == (1, 3, 8, 8)
    assert tensor.max() == 1.0


def test_prediction_result_is_serializable() -> None:
    """Prediction results include decode success in serialized output."""
    result = PredictionResult(decoded_text="hello", latency_ms=1.2)
    assert result.to_dict()["decode_success"] is True


def test_predictor_runs_single_image_on_cpu() -> None:
    """Predictor restores a compact configured image on CPU."""
    config = ModelConfig(
        image_size=16,
        patch_size=8,
        embed_dim=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    restored, result = Predictor(
        config=config, device="cpu", mixed_precision=False
    ).predict_image(Image.new("RGB", (16, 16), "white"))
    assert restored.size == (16, 16)
    assert result.latency_ms >= 0


def test_pipeline_writes_json_and_csv_reports(tmp_path: Path) -> None:
    """Pipeline report writer emits both requested report formats."""
    json_path, csv_path = InferencePipeline.write_reports(
        [PredictionResult()], tmp_path
    )
    assert json_path.is_file()
    assert csv_path.is_file()


def test_torchscript_export_is_validated(tmp_path: Path) -> None:
    """TorchScript exporter writes a loadable small module."""
    model = torch.nn.Identity()
    path = export_torchscript(model, torch.ones(1, 3), tmp_path / "model.pt")
    assert path.is_file()
