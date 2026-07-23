"""Tests for QR-aware restoration objectives."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mambatransqr.losses import (  # noqa: E402
    DecodeConsistencyLoss,
    MultiScaleRestorationLoss,
    QRStructureLoss,
    qr_structural_masks,
)


def test_qr_structure_masks_and_loss_are_metadata_aware() -> None:
    """Structural masks cover QR regions and produce a differentiable loss."""
    prediction = torch.rand(2, 3, 64, 64, requires_grad=True)
    target = torch.rand(2, 3, 64, 64)
    metadata = [{"qr_version": 1, "border": 4}, {"qr_version": 2, "border": 4}]
    masks = qr_structural_masks(target, metadata)
    assert masks["finder"].sum() > 0 and masks["quiet"].sum() > 0
    QRStructureLoss()(prediction, target, metadata).backward()
    assert prediction.grad is not None


def test_decode_surrogate_is_differentiable_not_external_decoder() -> None:
    """DecodeConsistencyLoss has no external decoder dependency or gradient path."""
    values = torch.rand(1, 3, 16, 16, requires_grad=True)
    DecodeConsistencyLoss(temperature=0.2)(values).backward()
    assert values.grad is not None


def test_weighted_multiscale_loss_forwards_and_backwards() -> None:
    """Combined objective exposes a scalar total with all weighted components."""
    prediction = torch.rand(1, 3, 32, 32, requires_grad=True)
    target = torch.rand(1, 3, 32, 32)
    objective = MultiScaleRestorationLoss()
    loss = objective(prediction, target, [{"qr_version": 1}])
    loss.backward()
    assert loss.ndim == 0 and {"structure", "decode_consistency"} <= set(
        objective.last_components
    )
