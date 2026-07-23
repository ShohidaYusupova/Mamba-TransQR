"""Inference result data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PredictionResult:
    """Serializable result of one image inference operation.

    Attributes:
        source_path: Optional input image path.
        restored_path: Optional saved output image path.
        decoded_text: Optional QR payload extracted from restored output.
        latency_ms: Model inference latency in milliseconds.
        metadata: Additional prediction metadata.
    """

    source_path: str | None = None
    restored_path: str | None = None
    decoded_text: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def decode_success(self) -> bool:
        """Return whether a QR payload was successfully decoded."""
        return self.decoded_text is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a report-ready dictionary."""
        return {**asdict(self), "decode_success": self.decode_success}
