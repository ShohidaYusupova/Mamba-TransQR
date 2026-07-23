"""QR decoding metrics."""

from __future__ import annotations

from collections.abc import Sequence


def decode_metrics(
    decoded: Sequence[bool], latencies_ms: Sequence[float]
) -> dict[str, float]:
    """Return QR decode success rate and mean latency."""
    if not decoded or len(decoded) != len(latencies_ms):
        raise ValueError("decoded and latencies must be equally non-empty")
    return {
        "decode_rate": sum(decoded) / len(decoded),
        "success_rate": sum(decoded) / len(decoded),
        "decode_latency_ms": sum(latencies_ms) / len(latencies_ms),
    }
