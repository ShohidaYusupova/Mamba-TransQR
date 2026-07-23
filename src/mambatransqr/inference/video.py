"""Optional OpenCV video-frame inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mambatransqr.inference.predictor import Predictor
from mambatransqr.inference.result import PredictionResult


class VideoPredictor:
    """Restore each frame of a video using an initialized predictor."""

    def __init__(self, predictor: Predictor) -> None:
        """Store the predictor used for video frames."""
        self.predictor = predictor

    def predict_video(
        self, source: str | Path, output: str | Path
    ) -> list[PredictionResult]:
        """Restore a video and return per-frame inference results.

        Raises:
            ImportError: If the optional OpenCV dependency is unavailable.
        """
        try:
            import cv2
        except ImportError as error:
            raise ImportError(
                "Video inference requires the optional opencv-python package."
            ) from error
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"could not open video: {source}")
        frames_per_second = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*"mp4v"),
            frames_per_second,
            (width, height),
        )
        results: list[PredictionResult] = []
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                restored, result = self.predictor.predict_image(image)
                restored = restored.resize((width, height))
                writer.write(cv2.cvtColor(np.asarray(restored), cv2.COLOR_RGB2BGR))
                results.append(result)
        finally:
            capture.release()
            writer.release()
        return results
