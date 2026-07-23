"""Evaluation and benchmark report generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path


class ReportGenerator:
    """Write JSON evaluation, CSV benchmark, and Markdown summary reports."""

    @staticmethod
    def evaluation_json(metrics: dict[str, float], path: str | Path) -> Path:
        """Write a JSON evaluation report and return its path."""
        destination = _prepare(path)
        destination.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return destination

    @staticmethod
    def benchmark_csv(metrics: dict[str, float], path: str | Path) -> Path:
        """Write a one-row CSV benchmark report and return its path."""
        destination = _prepare(path)
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(metrics))
            writer.writeheader()
            writer.writerow(metrics)
        return destination

    @staticmethod
    def summary_markdown(
        evaluation: dict[str, float], benchmark: dict[str, float], path: str | Path
    ) -> Path:
        """Write a concise Markdown evaluation summary and return its path."""
        destination = _prepare(path)
        rows = ["# Evaluation Summary", "", "## Evaluation", "", "| Metric | Value |", "|---|---:|"]
        rows.extend(f"| {name} | {value:.6f} |" for name, value in sorted(evaluation.items()))
        rows.extend(["", "## Benchmark", "", "| Metric | Value |", "|---|---:|"])
        rows.extend(f"| {name} | {value:.6f} |" for name, value in sorted(benchmark.items()))
        destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return destination


def _prepare(path: str | Path) -> Path:
    """Create parent directories and return a normalized destination path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination
