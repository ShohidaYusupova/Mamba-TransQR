"""Immutable completed/resumable Phase 4 ablation variant selection."""

from __future__ import annotations

COMPLETED_ABLATION_VARIANTS = frozenset(
    ("full_model", "without_mamba", "without_qr_structure_loss")
)
RESUMABLE_ABLATION_VARIANTS = (
    "without_decode_consistency_loss",
    "without_ema",
    "without_refinement_decoder",
)


def selected_resumable_variants(requested: str | None) -> list[str]:
    """Select remaining variants while always skipping completed variants."""
    if requested is None:
        return list(RESUMABLE_ABLATION_VARIANTS)
    if requested in COMPLETED_ABLATION_VARIANTS:
        return []
    if requested not in RESUMABLE_ABLATION_VARIANTS:
        raise ValueError(f"unknown ablation variant: {requested}")
    return [requested]
