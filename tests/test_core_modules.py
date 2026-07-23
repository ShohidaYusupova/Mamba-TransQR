"""Tests for losses, metrics, and experiment utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mambatransqr.experiments import Experiment  # noqa: E402
from mambatransqr.losses import (  # noqa: E402
    CharbonnierLoss,
    CombinedLoss,
    ReconstructionLoss,
)
from mambatransqr.metrics import MetricsManager  # noqa: E402


def test_combined_loss_is_differentiable() -> None:
    """Combined loss returns a differentiable scalar."""
    values = torch.ones(1, 1, 2, 2, requires_grad=True)
    loss = CombinedLoss({"l1": ReconstructionLoss(), "char": CharbonnierLoss()})
    result = loss(values, torch.zeros_like(values))
    result.backward()
    assert result.item() > 0


def test_metrics_manager_scores_identical_images() -> None:
    """Image metrics score identical images correctly."""
    values = MetricsManager(("psnr", "ssim")).compute(
        torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4)
    )
    assert values["ssim"] == pytest.approx(1.0)


def test_experiment_creates_logs_and_summary(tmp_path: Path) -> None:
    """Experiment creates automatic directories and output artifacts."""
    experiment = Experiment("unit", tmp_path, {"seed": 1})
    experiment.log({"loss": 1.0}, 1)
    assert experiment.summary({"ok": True}).is_file()
    assert (experiment.directory / "metrics.csv").is_file()
