"""Inference latency measurement utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean
from time import perf_counter

import torch


@dataclass(frozen=True, slots=True)
class LatencyResult:
    """Inference latency summary in milliseconds."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    samples: int


def measure_latency(
    function: Callable[[], object],
    *,
    warmup: int = 10,
    iterations: int = 100,
    device: torch.device | None = None,
) -> LatencyResult:
    """Measure synchronized function latency.

    Args:
        function: Zero-argument inference function to time.
        warmup: Unrecorded warmup calls.
        iterations: Recorded calls.
        device: Optional device synchronized around measurements.

    Returns:
        Millisecond latency summary.
    """
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    for _ in range(warmup):
        function()
    values: list[float] = []
    for _ in range(iterations):
        _synchronize(device)
        started = perf_counter()
        function()
        _synchronize(device)
        values.append((perf_counter() - started) * 1_000)
    ordered = sorted(values)
    return LatencyResult(
        mean_ms=mean(values),
        median_ms=ordered[len(ordered) // 2],
        p95_ms=ordered[min(len(ordered) - 1, round(0.95 * len(ordered)))],
        samples=iterations,
    )


def _synchronize(device: torch.device | None) -> None:
    """Synchronize CUDA work before wall-clock timing when appropriate."""
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
