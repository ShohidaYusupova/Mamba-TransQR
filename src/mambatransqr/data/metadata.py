"""Synthetic QR dataset metadata records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class QRMetadata:
    """Serializable paired-sample provenance."""

    sample_id: str
    payload: str
    payload_type: str
    qr_version: int
    error_correction_level: str
    image_dimensions: tuple[int, int]
    degradation_severity: str | None
    degradation_operators: tuple[str, ...]
    degradation_parameters: dict[str, Any]
    random_seed: int
    clean_image_path: str
    damaged_image_path: str | None
    generation_timestamp: str
    decoder_readable_before: bool | None
    decoder_readable_after: bool | None
    split: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QRMetadata":
        """Validate and reconstruct persisted metadata."""
        required = {field.name for field in cls.__dataclass_fields__.values()}
        missing = required - set(value)
        if missing:
            raise ValueError(f"metadata missing fields: {sorted(missing)}")
        return cls(
            **{**value, "image_dimensions": tuple(value["image_dimensions"]), "degradation_operators": tuple(value["degradation_operators"])}
        )
