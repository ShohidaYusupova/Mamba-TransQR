"""Measured image and QR decode metrics; no values are synthesized."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from mambatransqr.evaluation.metrics import MetricsManager


def image_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    """Compute measured PSNR, SSIM, MSE, and MAE."""
    return MetricsManager(("psnr", "ssim", "mse", "mae")).compute(prediction, target)


def aggregate(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Average only present numeric measurements over explicit grouping keys."""
    groups: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    for group, values in groups.items():
        result = dict(zip(keys, group, strict=True))
        numeric = {
            key
            for row in values
            for key, value in row.items()
            if isinstance(value, (int, float))
        }
        result.update(
            {
                key: sum(
                    float(row[key])
                    for row in values
                    if isinstance(row.get(key), (int, float))
                )
                / sum(isinstance(row.get(key), (int, float)) for row in values)
                for key in numeric
            }
        )
        output.append(result)
    return output
