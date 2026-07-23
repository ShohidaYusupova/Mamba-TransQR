"""Evaluation and benchmark report generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class ReportGenerator:
    """Write JSON evaluation, CSV benchmark, and Markdown summary reports."""

    @staticmethod
    def evaluation_json(
        metrics: dict[str, float],
        path: str | Path,
        *,
        architecture: dict[str, str | None] | None = None,
    ) -> Path:
        """Write a JSON evaluation report and return its path."""
        destination = _prepare(path)
        destination.write_text(
            json.dumps(_with_architecture(metrics, architecture), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return destination

    @staticmethod
    def benchmark_csv(
        metrics: dict[str, float],
        path: str | Path,
        *,
        architecture: dict[str, str | None] | None = None,
    ) -> Path:
        """Write a one-row CSV benchmark report and return its path."""
        destination = _prepare(path)
        with destination.open("w", newline="", encoding="utf-8") as stream:
            row = _with_architecture(metrics, architecture)
            writer = csv.DictWriter(stream, fieldnames=sorted(row))
            writer.writeheader()
            writer.writerow(row)
        return destination

    @staticmethod
    def summary_markdown(
        evaluation: dict[str, float],
        benchmark: dict[str, float],
        path: str | Path,
        *,
        architecture: dict[str, str | None] | None = None,
    ) -> Path:
        """Write a concise Markdown evaluation summary and return its path."""
        destination = _prepare(path)
        rows = [
            "# Evaluation Summary",
            "",
            "## Evaluation",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
        rows.extend(
            f"| {name} | {value:.6f} |" for name, value in sorted(evaluation.items())
        )
        rows.extend(["", "## Benchmark", "", "| Metric | Value |", "|---|---:|"])
        rows.extend(
            f"| {name} | {value:.6f} |" for name, value in sorted(benchmark.items())
        )
        if architecture is not None:
            rows.extend(
                [
                    "",
                    "## Architecture identity",
                    "",
                    "| Field | Value |",
                    "|---|---|",
                ]
            )
            rows.extend(
                f"| {key} | {value if value is not None else 'not installed'} |"
                for key, value in _identity(architecture).items()
            )
        destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return destination


def _prepare(path: str | Path) -> Path:
    """Create parent directories and return a normalized destination path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _identity(architecture: dict[str, str | None] | None) -> dict[str, str | None]:
    """Normalize mandatory backend traceability fields."""
    architecture = architecture or {}
    return {
        "mamba_backend": architecture.get("mamba_backend"),
        "mamba_implementation": architecture.get("mamba_implementation"),
        "mamba_ssm_version": architecture.get("mamba_ssm_version"),
    }


def _with_architecture(
    metrics: dict[str, float], architecture: dict[str, str | None] | None
) -> dict[str, Any]:
    """Add mandatory architecture identity fields to a report row."""
    return {**metrics, **_identity(architecture)}
