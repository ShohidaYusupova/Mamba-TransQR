"""Model inference benchmarking."""

from __future__ import annotations

import os
import tracemalloc
import warnings
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from mambatransqr.evaluation.latency import LatencyResult, measure_latency


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Model inference benchmark metrics."""

    throughput: float
    fps: float
    gpu_memory_mb: float
    cpu_memory_mb: float
    latency: LatencyResult

    def to_dict(self) -> dict[str, float]:
        """Return a flat report-ready metric mapping."""
        result = asdict(self)
        latency = result.pop("latency")
        return {**result, **{f"latency_{key}": value for key, value in latency.items()}}


class BenchmarkRunner:
    """Benchmark model inference throughput, latency, and memory.

    Args:
        model: Model to benchmark.
        device: Device executing inference.
    """

    def __init__(self, model: nn.Module, device: torch.device | None = None) -> None:
        """Store the benchmark target and choose its device."""
        parameter = next(model.parameters(), None)
        self.device = device or (
            parameter.device if parameter is not None else torch.device("cpu")
        )
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def run(
        self,
        inputs: Tensor,
        *,
        warmup: int = 10,
        iterations: int = 100,
    ) -> BenchmarkResult:
        """Benchmark repeated model inference.

        Args:
            inputs: Representative input batch.
            warmup: Unrecorded inference calls.
            iterations: Recorded inference calls.

        Returns:
            Throughput, FPS, memory, and latency metrics.
        """
        batch = inputs.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        tracemalloc.start()
        latency = measure_latency(
            lambda: self.model(batch),
            warmup=warmup,
            iterations=iterations,
            device=self.device,
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        throughput = batch.shape[0] / (latency.mean_ms / 1_000)
        gpu_memory = (
            torch.cuda.max_memory_allocated(self.device) / (1024**2)
            if self.device.type == "cuda"
            else 0.0
        )
        return BenchmarkResult(
            throughput=throughput,
            fps=throughput,
            gpu_memory_mb=gpu_memory,
            cpu_memory_mb=max(peak, _rss_bytes()) / (1024**2),
            latency=latency,
        )


def _rss_bytes() -> int:
    """Return process resident memory when psutil is available."""
    try:
        import psutil
    except ImportError:
        warnings.warn(
            "CPU RSS measurement requires the 'metrics' optional dependency; "
            "reporting traced Python allocations only.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0
    return int(psutil.Process(os.getpid()).memory_info().rss)
