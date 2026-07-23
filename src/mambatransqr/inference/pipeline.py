"""High-level inference pipeline and report output."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mambatransqr.inference.batch import BatchPredictor
from mambatransqr.inference.predictor import Predictor
from mambatransqr.inference.result import PredictionResult


class InferencePipeline:
    """Coordinate image inference, output saving, and JSON/CSV reports.

    Args:
        predictor: Initialized model predictor.
    """

    def __init__(self, predictor: Predictor) -> None:
        """Initialize batch support for the predictor."""
        self.predictor = predictor
        self.batch_predictor = BatchPredictor(predictor)

    def predict_image(
        self, source: str | Path, output_dir: str | Path, *, save_original: bool = True
    ) -> PredictionResult:
        """Restore one image and persist it with optional original copy."""
        return self.batch_predictor.predict_paths(
            [source], output_dir, save_original=save_original
        )[0]

    def predict_folder(
        self, source: str | Path, output_dir: str | Path, *, save_original: bool = False
    ) -> list[PredictionResult]:
        """Restore all images in a folder and return their results."""
        return self.batch_predictor.predict_folder(
            source, output_dir, save_original=save_original
        )

    @staticmethod
    def write_reports(
        results: list[PredictionResult], output_dir: str | Path
    ) -> tuple[Path, Path]:
        """Write JSON and CSV inference reports.

        Args:
            results: Prediction results to serialize.
            output_dir: Directory receiving ``predictions.json`` and ``predictions.csv``.

        Returns:
            JSON report path followed by CSV report path.
        """
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        records = [_with_architecture(result.to_dict()) for result in results]
        json_path = directory / "predictions.json"
        json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        csv_path = directory / "predictions.csv"
        columns = sorted({key for record in records for key in record})
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
        return json_path, csv_path


def _with_architecture(record: dict[str, object]) -> dict[str, object]:
    """Promote mandatory Mamba identity fields into each inference report row."""
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return {
        **record,
        "mamba_backend": metadata.get("mamba_backend"),
        "mamba_implementation": metadata.get("mamba_implementation"),
        "mamba_ssm_version": metadata.get("mamba_ssm_version"),
    }
