"""CPU end-to-end smoke test for the core restoration workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from mambatransqr.data.dataset import TrainDataset  # noqa: E402
from mambatransqr.data.loader import (  # noqa: E402
    create_evaluation_dataloader,
    create_train_dataloader,
)
from mambatransqr.evaluation.evaluator import Evaluator  # noqa: E402
from mambatransqr.evaluation.metrics import MetricsManager  # noqa: E402
from mambatransqr.inference.predictor import Predictor  # noqa: E402
from mambatransqr.losses import ReconstructionLoss  # noqa: E402
from mambatransqr.models import ModelConfig, build_model  # noqa: E402
from mambatransqr.training import (  # noqa: E402
    LossManager,
    OptimizerFactory,
    Trainer,
    TrainerConfig,
)


def _to_tensor(image: Image.Image) -> object:
    """Convert an RGB PIL image to a CHW float tensor without torchvision."""
    values = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(values).permute(2, 0, 1)


def _make_paired_images(root: Path) -> tuple[Path, Path]:
    """Create a two-sample paired synthetic restoration dataset."""
    source, target = root / "source", root / "target"
    source.mkdir()
    target.mkdir()
    for index, value in enumerate((32, 224)):
        image = Image.new("RGB", (16, 16), (value, value, value))
        image.save(source / f"sample_{index}.png")
        image.save(target / f"sample_{index}.png")
    return source, target


def test_cpu_restoration_workflow_smoke(tmp_path: Path) -> None:
    """Exercise data, training, checkpoint, inference, and evaluation on CPU."""
    source, target = _make_paired_images(tmp_path)
    dataset = TrainDataset(
        source, _to_tensor, target_root=target, target_transform=_to_tensor
    )
    train_loader = create_train_dataloader(dataset, batch_size=2)
    evaluation_loader = create_evaluation_dataloader(dataset, batch_size=2)
    config = ModelConfig(
        image_size=16,
        patch_size=8,
        embed_dim=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path_rate=0.0,
        mamba_backend="lightweight",
    )
    model = build_model(config)
    optimizer = OptimizerFactory.create(model.parameters())
    trainer = Trainer(
        model,
        optimizer,
        LossManager({"reconstruction": ReconstructionLoss("mse")}),
        TrainerConfig(
            epochs=1,
            device="cpu",
            mixed_precision=False,
            use_ema=False,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        ),
    )
    history = trainer.fit(train_loader, evaluation_loader)
    checkpoint = tmp_path / "checkpoints" / "latest.pt"
    assert history.records and checkpoint.is_file()

    restored_model = build_model(config)
    restored_trainer = Trainer(
        restored_model,
        OptimizerFactory.create(restored_model.parameters()),
        LossManager({"reconstruction": ReconstructionLoss("mse")}),
        TrainerConfig(device="cpu", mixed_precision=False, use_ema=False),
    )
    restored_trainer.resume(checkpoint)
    predictor = Predictor(restored_model, config, device="cpu", mixed_precision=False)
    _, result = predictor.predict_path(source / "sample_0.png")
    assert result.latency_ms >= 0.0

    metrics = Evaluator(
        restored_model, MetricsManager(("mse",)), torch.device("cpu")
    ).evaluate(evaluation_loader)
    assert "mse" in metrics
