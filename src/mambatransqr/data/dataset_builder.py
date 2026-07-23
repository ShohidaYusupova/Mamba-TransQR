"""Reproducible synthetic QR paired-dataset creation and validation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from typing import Any

from PIL import Image

from mambatransqr.data.metadata import QRMetadata
from mambatransqr.data.qr_degradation import QRDegradationEngine
from mambatransqr.data.qr_generator import QRGenerationSpec, QRGenerator, decode_readability
from mambatransqr.data.splits import split_payloads


@dataclass(frozen=True, slots=True)
class DatasetGenerationConfig:
    """Configuration for a deterministic synthetic QR dataset."""

    samples: int = 100
    seed: int = 42
    versions: tuple[int, ...] = tuple(range(1, 41))
    error_correction_levels: tuple[str, ...] = ("L", "M", "Q", "H")
    payload_types: tuple[str, ...] = ("alphanumeric", "numeric", "byte", "url")
    payload_length: int = 12
    image_size: int = 256
    border: int = 4
    foreground: str = "#000000"
    background: str = "#ffffff"
    severities: tuple[str, ...] = ("mild", "moderate", "severe")
    decoder_backend: str | None = None
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)


class DatasetIntegrityError(ValueError):
    """Raised when persisted synthetic QR data fails integrity validation."""


class SyntheticQRDatasetBuilder:
    """Build clean/damaged paired QR datasets grouped by non-leaking payload splits."""

    def __init__(self, config: DatasetGenerationConfig | None = None) -> None:
        self.config = config or DatasetGenerationConfig()
        if self.config.samples < 1:
            raise ValueError("samples must be positive")
        if not self.config.severities:
            raise ValueError("at least one degradation severity is required")

    def build(self, output: str | Path) -> list[QRMetadata]:
        """Create image pairs, per-sample metadata, and dataset summary artifacts."""
        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        rng = Random(self.config.seed)
        plans: list[tuple[int, QRGenerationSpec, int, Image.Image]] = []
        for index in range(self.config.samples):
            seed = rng.randrange(2**31)
            local = Random(seed)
            for _ in range(100):
                payload_type = local.choice(self.config.payload_types)
                spec = self._spec(payload_type, local)
                try:
                    clean = QRGenerator.render(spec)
                    plans.append((index, spec, seed, clean))
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("unable to generate a payload fitting the configured QR set")
        assignments = split_payloads(
            [spec.payload for _, spec, _, _ in plans],
            self.config.seed,
            self.config.split_ratios,
        )
        records: list[QRMetadata] = []
        for index, spec, seed, clean in plans:
            split = assignments[spec.payload]
            clean_relative = Path(split) / "clean" / f"qr_{index:06d}.png"
            self._save(clean, root / clean_relative)
            before = decode_readability(clean, self.config.decoder_backend)
            for severity in self.config.severities:
                engine = self._degrader(spec.version, spec.border, severity, seed)
                degraded = engine.degrade(clean)
                sample_id = f"qr_{index:06d}_{severity}"
                damaged_relative = Path(split) / "damaged" / severity / f"{sample_id}.png"
                self._save(degraded.image, root / damaged_relative)
                record = QRMetadata(
                    sample_id=sample_id,
                    payload=spec.payload,
                    payload_type=spec.payload_type,
                    qr_version=spec.version,
                    error_correction_level=spec.error_correction,
                    image_dimensions=clean.size,
                    degradation_severity=severity,
                    degradation_operators=degraded.operations,
                    degradation_parameters={
                        "module_count": 17 + 4 * spec.version,
                        "quiet_zone_modules": spec.border,
                        "operation_count": len(degraded.operations),
                    },
                    random_seed=seed,
                    clean_image_path=str(clean_relative),
                    damaged_image_path=str(damaged_relative),
                    generation_timestamp=datetime.fromtimestamp(seed, UTC).isoformat(),
                    decoder_readable_before=before,
                    decoder_readable_after=decode_readability(
                        degraded.image, self.config.decoder_backend
                    ),
                    split=split,
                )
                self._write_metadata(root / split / "metadata" / f"{sample_id}.json", record)
                records.append(record)
        self._write_artifacts(root, records)
        validate_dataset(root)
        return records

    def _spec(self, payload_type: str, rng: Random) -> QRGenerationSpec:
        version = rng.choice(self.config.versions)
        return QRGenerationSpec(
            version=version,
            error_correction=rng.choice(self.config.error_correction_levels),
            payload_type=payload_type,
            payload=QRGenerator.make_payload(payload_type, self.config.payload_length, rng),
            image_size=self.config.image_size,
            border=self.config.border,
            foreground=self.config.foreground,
            background=self.config.background,
        )

    @staticmethod
    def _save(image: Image.Image, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")

    @staticmethod
    def _write_metadata(path: Path, record: QRMetadata) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _degrader(
        version: int, border: int, severity: str, seed: int
    ) -> QRDegradationEngine:
        settings = {
            "mild": (("module_dropout", "ink_spread"), 1),
            "moderate": (("module_dropout", "module_occlusion", "print_scan"), 2),
            "severe": (QRDegradationEngine.available_degradations(), 4),
        }
        try:
            enabled, count = settings[severity]
        except KeyError as error:
            raise ValueError("severity must be mild, moderate, or severe") from error
        return QRDegradationEngine(
            module_count=17 + 4 * version,
            quiet_zone_modules=border,
            enabled=enabled,
            count=count,
            seed=seed,
        )

    def _write_artifacts(self, root: Path, records: list[QRMetadata]) -> None:
        rows = [record.to_dict() for record in records]
        with (root / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        split_counts = {name: sum(item.split == name for item in records) for name in ("train", "validation", "test")}
        payload_counts = {name: len({item.payload for item in records if item.split == name}) for name in split_counts}
        (root / "dataset_summary.json").write_text(json.dumps({"samples": len(records), "seed": self.config.seed, "severities": self.config.severities}, indent=2), encoding="utf-8")
        (root / "split_statistics.json").write_text(json.dumps({"samples": split_counts, "unique_payloads": payload_counts}, indent=2), encoding="utf-8")


def validate_dataset(root: str | Path) -> dict[str, int]:
    """Validate metadata, PNG readability, pairs, IDs, and payload split isolation."""
    directory = Path(root)
    paths = sorted(directory.glob("*/metadata/*.json"))
    if not paths:
        raise DatasetIntegrityError("no metadata files found")
    records: list[QRMetadata] = []
    for path in paths:
        try:
            records.append(QRMetadata.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise DatasetIntegrityError(f"invalid metadata: {path}") from error
    ids = [record.sample_id for record in records]
    if len(ids) != len(set(ids)):
        raise DatasetIntegrityError("duplicate sample IDs")
    ownership: dict[str, str] = {}
    for record in records:
        if ownership.setdefault(record.payload, record.split) != record.split:
            raise DatasetIntegrityError("payload leakage across splits")
        dimensions: list[tuple[int, int]] = []
        for relative in (record.clean_image_path, record.damaged_image_path):
            if relative is None or not (directory / relative).is_file():
                raise DatasetIntegrityError(f"missing paired image for {record.sample_id}")
            try:
                with Image.open(directory / relative) as image:
                    if image.size != record.image_dimensions:
                        raise DatasetIntegrityError(f"clean/damaged pair mismatch for {record.sample_id}")
                    dimensions.append(image.size)
                    image.verify()
            except (OSError, ValueError) as error:
                raise DatasetIntegrityError(f"corrupt image for {record.sample_id}") from error
        if len(dimensions) != 2 or dimensions[0] != dimensions[1]:
            raise DatasetIntegrityError(f"clean/damaged pair mismatch for {record.sample_id}")
    return {"records": len(records), "payloads": len(ownership)}


def load_generation_config(path: str | Path) -> DatasetGenerationConfig:
    """Load YAML configuration without making PyYAML a core runtime dependency."""
    try:
        import yaml
    except ImportError as error:
        raise ImportError(
            "YAML dataset configuration requires PyYAML. Install with "
            "`pip install -e \".[qr-generation]\"`."
        ) from error
    value: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for name in (
        "versions",
        "error_correction_levels",
        "payload_types",
        "severities",
        "split_ratios",
    ):
        if name in value:
            value[name] = tuple(value[name])
    return DatasetGenerationConfig(**value)
