"""Measured latency, throughput, memory, and complexity helpers."""

from __future__ import annotations

import time

import torch
from torch import Tensor, nn


def measure(
    model: nn.Module, inputs: Tensor, iterations: int = 10
) -> dict[str, float | None]:
    """Measure model latency without claiming unavailable FLOPs/memory values."""
    model.eval()
    device = inputs.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = (time.perf_counter() - started) / iterations
    return {
        "model_inference_latency_ms": seconds * 1000,
        "throughput": inputs.shape[0] / seconds,
        "fps": inputs.shape[0] / seconds,
        "parameter_count": float(sum(item.numel() for item in model.parameters())),
        "peak_gpu_memory": (
            float(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "peak_cpu_memory": None,
        "flops_or_macs": None,
    }
