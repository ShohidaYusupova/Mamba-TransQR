"""Leakage-safe deterministic QR dataset splitting."""

from __future__ import annotations

from random import Random


def split_payloads(
    payloads: list[str],
    seed: int,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, str]:
    """Assign each unique payload to one reproducible train/validation/test split."""
    if (
        len(ratios) != 3
        or any(value < 0 for value in ratios)
        or abs(sum(ratios) - 1) > 1e-9
    ):
        raise ValueError("split ratios must be non-negative and sum to 1")
    unique = sorted(set(payloads))
    shuffled = unique[:]
    Random(seed).shuffle(shuffled)
    train_end = round(len(shuffled) * ratios[0])
    validation_end = train_end + round(len(shuffled) * ratios[1])
    return {
        payload: (
            "train"
            if index < train_end
            else "validation" if index < validation_end else "test"
        )
        for index, payload in enumerate(shuffled)
    }
