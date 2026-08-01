"""Unit tests for the training framework."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mambatransqr.models import ModelConfig, build_model  # noqa: E402
from mambatransqr.training import (  # noqa: E402
    CheckpointManager,
    CSVLogger,
    EarlyStopping,
    ExponentialMovingAverage,
    LossManager,
    OptimizerConfig,
    OptimizerFactory,
    SchedulerConfig,
    SchedulerFactory,
    Trainer,
    TrainerConfig,
    TrainingState,
)


def _batches() -> list[dict[str, object]]:
    """Create deterministic tiny regression batches."""
    return [
        {
            "image": torch.tensor([[1.0], [2.0]]),
            "target": torch.tensor([[2.0], [4.0]]),
        },
        {
            "image": torch.tensor([[3.0], [4.0]]),
            "target": torch.tensor([[6.0], [8.0]]),
        },
    ]


def _trainer(directory: Path) -> Trainer:
    """Create a compact CPU trainer."""
    model = torch.nn.Linear(1, 1)
    optimizer = OptimizerFactory.create(
        model.parameters(), OptimizerConfig(learning_rate=0.1)
    )
    return Trainer(
        model,
        optimizer,
        LossManager({"mse": torch.nn.MSELoss()}),
        TrainerConfig(
            epochs=2,
            device="cpu",
            mixed_precision=False,
            use_ema=True,
            checkpoint_dir=str(directory),
        ),
    )


def test_optimizer_factory_creates_adamw() -> None:
    """The optimizer factory creates the requested optimizer type."""
    optimizer = OptimizerFactory.create(torch.nn.Linear(1, 1).parameters())
    assert isinstance(optimizer, torch.optim.AdamW)


def test_loss_manager_returns_weighted_total() -> None:
    """Loss manager combines named loss functions and exposes metrics."""
    manager = LossManager({"mse": torch.nn.MSELoss()}, {"mse": 2.0})
    loss, metrics = manager(torch.tensor([1.0]), torch.tensor([3.0]))
    assert loss.item() == pytest.approx(8.0)
    assert metrics["mse"] == pytest.approx(4.0)
    assert metrics["loss"] == pytest.approx(8.0)


def test_ema_updates_and_restores_parameters() -> None:
    """EMA can swap averaged weights into and out of a model."""
    model = torch.nn.Linear(1, 1, bias=False)
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)
    original = model.weight.detach().clone()
    ema.apply_to(model)
    assert not torch.equal(model.weight, original)
    ema.restore(model)
    assert torch.equal(model.weight, original)


def test_checkpoint_manager_restores_state(tmp_path: Path) -> None:
    """Saved checkpoints restore model, optimizer, and training state."""
    model = torch.nn.Linear(1, 1)
    optimizer = OptimizerFactory.create(model.parameters())
    state = TrainingState(epoch=3, global_step=12, best_metric=0.5)
    manager = CheckpointManager(tmp_path)
    manager.save(model, optimizer, state, is_best=True)
    restored, _ = manager.load(manager.latest_path, model, optimizer)
    assert restored.epoch == 3
    assert manager.best_path.is_file()


def test_checkpoint_can_deploy_ema_and_resume_raw_weights(tmp_path: Path) -> None:
    """Deployable weights and raw resume weights remain distinct."""
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = OptimizerFactory.create(model.parameters())
    raw = model.state_dict()["weight"].clone()
    deployable = {"weight": raw + 1.0}
    manager = CheckpointManager(tmp_path)
    manager.save(
        model,
        optimizer,
        TrainingState(),
        deployable_model_state=deployable,
        is_best=True,
    )
    manager.load(manager.best_path, model)
    assert torch.equal(model.weight, deployable["weight"])
    manager.load(manager.best_path, model, optimizer)
    assert torch.equal(model.weight, raw)


def test_warmup_cosine_scheduler_reaches_minimum_lr() -> None:
    """Warmup-cosine scheduling warms up and decays to the configured floor."""
    optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.2)
    scheduler = SchedulerFactory.create(
        optimizer,
        SchedulerConfig(
            name="warmup_cosine", epochs=6, warmup_epochs=2, min_lr=0.01
        ),
    )
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] < rates[2]
    assert rates[-1] == pytest.approx(0.01)


def test_checkpoint_records_mamba_architecture_identity(tmp_path: Path) -> None:
    """Mamba checkpoints retain enough identity to trace experimental results."""
    config = ModelConfig(
        image_size=16,
        patch_size=8,
        embed_dim=8,
        depth=1,
        num_heads=2,
        mamba_backend="lightweight",
    )
    with pytest.warns(UserWarning):
        model = build_model(config)
    optimizer = OptimizerFactory.create(model.parameters())
    manager = CheckpointManager(tmp_path)
    manager.save(model, optimizer, TrainingState())
    payload = torch.load(manager.latest_path, weights_only=False)
    assert payload["architecture"] == config.architecture_identity()
    assert payload["mamba_backend"] == "lightweight"


def test_trainer_saves_latest_and_best_checkpoints(tmp_path: Path) -> None:
    """A fit run writes latest and best checkpoints and records history."""
    trainer = _trainer(tmp_path)
    history = trainer.fit(_batches(), _batches())
    assert len(history.records) == 2
    assert (tmp_path / "latest.pt").is_file()
    assert (tmp_path / "best.pt").is_file()


def test_csv_logger_writes_metrics(tmp_path: Path) -> None:
    """CSV logger writes a header and a metric row."""
    path = tmp_path / "metrics.csv"
    logger = CSVLogger(path)
    logger.log_metrics({"loss": 1.0}, step=3)
    logger.close()
    assert "loss" in path.read_text(encoding="utf-8")


def test_early_stopping_marks_trainer_state(tmp_path: Path) -> None:
    """Early stopping requests a halt after exceeded patience."""
    trainer = _trainer(tmp_path)
    callback = EarlyStopping(patience=0)
    callback.on_epoch_end(trainer, {"val_loss": 1.0})
    callback.on_epoch_end(trainer, {"val_loss": 1.0})
    assert trainer.state.stopped_early
