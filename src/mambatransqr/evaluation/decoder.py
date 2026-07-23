"""Optional QR decoding evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """Result of one QR decoder attempt."""

    decoded: bool
    value: str | None
    latency_ms: float
    backend: str


class QRDecoderEvaluator:
    """Evaluate QR decoder success rates using ZBar or ZXing.

    Args:
        backend: Decoder backend, either ``zbar`` or ``zxing``.
    """

    def __init__(self, backend: str = "zbar") -> None:
        """Validate the decoder backend."""
        if backend not in {"zbar", "zxing"}:
            raise ValueError("backend must be 'zbar' or 'zxing'")
        self.backend = backend

    def decode(self, image: Image.Image) -> DecodeResult:
        """Decode one PIL image and measure decoding latency."""
        started = perf_counter()
        value = (
            self._decode_zbar(image)
            if self.backend == "zbar"
            else self._decode_zxing(image)
        )
        return DecodeResult(
            decoded=value is not None,
            value=value,
            latency_ms=(perf_counter() - started) * 1_000,
            backend=self.backend,
        )

    def evaluate(self, images: list[Image.Image]) -> dict[str, float]:
        """Calculate decode rate, success rate, and mean decode latency."""
        if not images:
            raise ValueError("at least one image is required")
        results = [self.decode(image) for image in images]
        successes = sum(result.decoded for result in results)
        rate = successes / len(results)
        return {
            f"{self.backend}_decode_rate": rate,
            "success_rate": rate,
            "decode_latency_ms": sum(result.latency_ms for result in results)
            / len(results),
        }

    @staticmethod
    def _decode_zbar(image: Image.Image) -> str | None:
        """Decode using pyzbar's ZBar bindings."""
        try:
            from pyzbar.pyzbar import decode
        except ImportError as error:
            raise ImportError(
                "ZBar decoding requires the optional pyzbar package."
            ) from error
        decoded = decode(image)
        return decoded[0].data.decode("utf-8") if decoded else None

    @staticmethod
    def _decode_zxing(image: Image.Image) -> str | None:
        """Decode using the optional zxing-cpp bindings."""
        try:
            import zxingcpp
        except ImportError as error:
            raise ImportError(
                "ZXing decoding requires the optional zxing-cpp package."
            ) from error
        result = zxingcpp.read_barcode(np.asarray(image.convert("RGB")))
        return result.text if result is not None else None
