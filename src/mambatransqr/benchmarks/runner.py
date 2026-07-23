"""Measured, resumable benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mambatransqr.benchmarks.latency import measure
from mambatransqr.benchmarks.metrics import aggregate, image_metrics
from mambatransqr.benchmarks.report import can_resume, write_rows
from mambatransqr.benchmarks.tables import export_table
from mambatransqr.training.recipes import reproducibility_metadata


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    output: str = "results/benchmark"
    seed: int = 42
    manifest: str | None = None
    resume: bool = True
    iterations: int = 10


class BenchmarkRunner:
    """Evaluate only supplied models and paired tensors; never invent results."""

    def __init__(self, config: BenchmarkConfig | None = None) -> None: self.config = config or BenchmarkConfig()

    def run(self, model: nn.Module, batches: list[dict[str, Any]], *, model_name: str = "Mamba-TransQR") -> list[dict[str, Any]]:
        root = Path(self.config.output); identity = reproducibility_metadata({"seed": self.config.seed}, self.config.manifest, model)
        if self.config.resume and can_resume(root, identity):
            import json
            return json.loads((root / "raw_results.json").read_text(encoding="utf-8"))
        model.eval(); rows: list[dict[str, Any]] = []
        for batch in batches:
            inputs, target = batch["image"], batch["target"]
            with torch.no_grad(): output = model(inputs)
            row = {"model": model_name, **image_metrics(output, target), **measure(model, inputs, self.config.iterations), **{key: batch.get(key) for key in ("degradation_type", "degradation_severity", "qr_version", "error_correction_level", "payload_type")}}
            rows.append(row)
        write_rows(root, rows, identity)
        for stem, keys in (("model_summary", ("model",)), ("degradation_summary", ("degradation_type",)), ("severity_summary", ("degradation_severity",)), ("decode_summary", ("model",)), ("latency_summary", ("model",))): export_table(aggregate(rows, keys), stem, root)
        return rows
