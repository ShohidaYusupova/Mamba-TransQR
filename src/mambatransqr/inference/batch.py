"""Batch and folder inference helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from mambatransqr.inference.predictor import Predictor
from mambatransqr.inference.result import PredictionResult


class BatchPredictor:
    """Run a predictor over image paths and write restored images.

    Args:
        predictor: Initialized single-image predictor.
    """

    def __init__(self, predictor: Predictor) -> None:
        """Store the predictor used for every path."""
        self.predictor = predictor

    def predict_paths(
        self,
        paths: Iterable[str | Path],
        output_dir: str | Path,
        *,
        save_original: bool = False,
    ) -> list[PredictionResult]:
        """Restore paths and save their outputs.

        Args:
            paths: Input image paths.
            output_dir: Destination directory for restored images.
            save_original: Whether to copy original images alongside outputs.

        Returns:
            Inference results in input order.
        """
        destination = Path(output_dir)
        results: list[PredictionResult] = []
        for path_value in paths:
            path = Path(path_value)
            restored, result = self.predictor.predict_path(path)
            restored_path = destination / f"{path.stem}_restored.png"
            restored_path.parent.mkdir(parents=True, exist_ok=True)
            restored.save(restored_path)
            if save_original:
                original_path = (
                    destination / f"{path.stem}_original{path.suffix.lower()}"
                )
                original_path.write_bytes(path.read_bytes())
            result.restored_path = str(restored_path)
            results.append(result)
        return results

    def predict_folder(
        self,
        folder: str | Path,
        output_dir: str | Path,
        *,
        recursive: bool = True,
        save_original: bool = False,
    ) -> list[PredictionResult]:
        """Restore all supported images found under a folder."""
        root = Path(folder)
        pattern = "**/*" if recursive else "*"
        extensions = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
        paths = sorted(
            path for path in root.glob(pattern) if path.suffix.lower() in extensions
        )
        return self.predict_paths(paths, output_dir, save_original=save_original)
