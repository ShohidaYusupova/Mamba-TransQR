"""Automatic ablation resume selection and state restoration tests."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mambatransqr.data.dataset_builder import SyntheticQRDatasetBuilder  # noqa: E402
from mambatransqr.data.paired_dataset import PairedQRDataset  # noqa: E402
from mambatransqr.experiments.ablation_resume import (  # noqa: E402
    selected_resumable_variants,
)
from mambatransqr.training import (  # noqa: E402
    AMPManager,
    CheckpointManager,
    EarlyStopping,
    ExponentialMovingAverage,
    IncompatibleCheckpointError,
    TrainingState,
    capture_rng_state,
    resolve_resume_checkpoint,
    validate_resume_identity,
)


def test_fresh_run_when_latest_checkpoint_is_absent(tmp_path: Path) -> None:
    """Auto mode starts fresh when latest.pt does not exist."""
    assert resolve_resume_checkpoint("auto", tmp_path) is None
    assert resolve_resume_checkpoint("never", tmp_path) is None


def test_automatic_resume_selects_latest_checkpoint(tmp_path: Path) -> None:
    """Auto mode selects the variant-local latest.pt."""
    latest = tmp_path / "latest.pt"
    latest.touch()
    assert resolve_resume_checkpoint("auto", tmp_path) == latest


def test_explicit_resume_path_must_exist(tmp_path: Path) -> None:
    """An explicit missing checkpoint is rejected instead of starting fresh."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_resume_checkpoint(tmp_path / "missing.pt", tmp_path)


def test_checkpoint_restores_next_epoch_and_all_training_state(tmp_path: Path) -> None:
    """Resume restores optimizer, scheduler, EMA, scaler, early stop, and epoch."""
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.2)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.step()
    scheduler.step()
    ema = ExponentialMovingAverage(model, decay=0.8)
    ema.update(model)
    amp = AMPManager(torch.device("cpu"), enabled=False)
    early = EarlyStopping(patience=4)
    early.best, early.wait = 0.4, 2
    identity = {"variant_name": "without_ema", "configuration_hash": "abc"}
    manager = CheckpointManager(tmp_path)
    manager.save(
        model,
        optimizer,
        TrainingState(epoch=7, global_step=21, best_metric=0.4, best_epoch=6),
        scheduler=scheduler,
        ema_state=ema.state_dict(),
        scaler_state=amp.state_dict(),
        early_stopping_state=early.state_dict(),
        rng_state=capture_rng_state(),
        resume_identity=identity,
    )

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=9.0)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=1, gamma=0.5
    )
    restored_ema = ExponentialMovingAverage(restored_model, decay=0.1)
    restored_amp = AMPManager(torch.device("cpu"), enabled=False)
    restored_early = EarlyStopping(patience=4)
    state, ema_state = manager.load(
        manager.latest_path,
        restored_model,
        restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_amp,
        early_stopping=restored_early,
        restore_rng=True,
    )
    assert state.epoch == 7
    assert list(range(state.epoch + 1, 10)) == [8, 9]
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert restored_optimizer.state
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    assert ema_state is not None
    restored_ema.load_state_dict(ema_state)
    assert restored_ema.decay == pytest.approx(0.8)
    assert restored_early.best == pytest.approx(0.4)
    assert restored_early.wait == 2


def test_rng_state_is_restored(tmp_path: Path) -> None:
    """Python, NumPy, and Torch random streams resume exactly."""
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    random.random()
    np.random.rand()
    torch.rand(1)
    from mambatransqr.training import restore_rng_state

    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert actual == pytest.approx(expected)


def test_incompatible_checkpoint_is_rejected() -> None:
    """Every required identity mismatch fails before loading state."""
    expected = {
        "variant_name": "without_ema",
        "configuration_hash": "expected",
        "dataset_manifest_hash": "dataset",
        "random_seed": 42,
        "backend_identity": {"mamba_backend": "lightweight"},
        "model_architecture": {"depth": 6},
    }
    payload = {"resume_identity": {**expected, "random_seed": 7}}
    with pytest.raises(IncompatibleCheckpointError, match="random_seed"):
        validate_resume_identity(payload, expected)


@pytest.mark.parametrize(
    "variant", ("full_model", "without_mamba", "without_qr_structure_loss")
)
def test_completed_variants_are_skipped(variant: str) -> None:
    """Completed variants can never enter the resumable training loop."""
    assert selected_resumable_variants(variant) == []


def test_existing_dataset_is_loaded_without_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Starting/resuming only reads metadata and never invokes dataset generation."""
    metadata = tmp_path / "train" / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "000.json").write_text(
        json.dumps(
            {"damaged_image_path": "damaged.png", "clean_image_path": "clean.png"}
        ),
        encoding="utf-8",
    )

    def fail_generation(*args: object, **kwargs: object) -> None:
        raise AssertionError("dataset generation must not be called")

    monkeypatch.setattr(SyntheticQRDatasetBuilder, "build", fail_generation)
    dataset = PairedQRDataset(tmp_path, "train")
    assert len(dataset) == 1
