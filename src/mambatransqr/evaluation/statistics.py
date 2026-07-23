"""Summary statistics for evaluation results."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev


@dataclass(frozen=True, slots=True)
class StatisticsResult:
    """Mean, standard deviation, and confidence interval summary."""

    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    count: int


def summarize(values: list[float]) -> StatisticsResult:
    """Calculate summary statistics and a normal-approximation 95% interval.

    Args:
        values: Non-empty scalar observations.

    Returns:
        Summary statistics for the observations.
    """
    if not values:
        raise ValueError("values must not be empty")
    average = mean(values)
    deviation = stdev(values) if len(values) > 1 else 0.0
    radius = 1.96 * deviation / sqrt(len(values))
    return StatisticsResult(
        average, deviation, average - radius, average + radius, len(values)
    )
