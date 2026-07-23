"""Deterministic QR-code payload and PNG generation."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from string import ascii_uppercase, digits

from PIL import Image


class OptionalQRDependencyError(ImportError):
    """Raised when QR rendering or decoding extras are not installed."""


@dataclass(frozen=True, slots=True)
class QRGenerationSpec:
    """Parameters defining one QR raster."""

    version: int
    error_correction: str
    payload_type: str
    payload: str
    image_size: int
    border: int = 4
    foreground: str = "#000000"
    background: str = "#ffffff"


class QRGenerator:
    """Render standard QR versions 1--40 with deterministic payload helpers."""

    _levels = ("L", "M", "Q", "H")
    _types = ("alphanumeric", "numeric", "byte", "url")

    @staticmethod
    def make_payload(payload_type: str, length: int, rng: Random) -> str:
        """Create a deterministic payload of the requested QR mode."""
        if length < 1:
            raise ValueError("payload length must be positive")
        if payload_type == "numeric":
            return "".join(rng.choice(digits) for _ in range(length))
        if payload_type == "alphanumeric":
            alphabet = digits + ascii_uppercase + " $%*+-./:"
            return "".join(rng.choice(alphabet) for _ in range(length))
        if payload_type == "byte":
            alphabet = ascii_uppercase + ascii_uppercase.lower() + digits + "-_"
            return "".join(rng.choice(alphabet) for _ in range(length))
        if payload_type == "url":
            suffix = "".join(
                rng.choice(ascii_uppercase.lower() + digits) for _ in range(length)
            )
            return f"https://example.test/{suffix}"
        raise ValueError(f"unsupported payload type: {payload_type}")

    @classmethod
    def render(cls, spec: QRGenerationSpec) -> Image.Image:
        """Render a PNG-compatible RGB image, failing clearly on missing extras."""
        if not 1 <= spec.version <= 40:
            raise ValueError("QR version must be between 1 and 40")
        if spec.error_correction not in cls._levels:
            raise ValueError("error correction must be one of L, M, Q, H")
        if spec.payload_type not in cls._types:
            raise ValueError(f"unsupported payload type: {spec.payload_type}")
        if spec.image_size < 1 or spec.border < 0:
            raise ValueError("image_size must be positive and border non-negative")
        try:
            import qrcode
        except ImportError as error:
            raise OptionalQRDependencyError(
                "QR generation requires the optional qrcode package. Install with "
                '`pip install -e ".[qr-generation]"`.'
            ) from error
        levels = {
            "L": qrcode.constants.ERROR_CORRECT_L,
            "M": qrcode.constants.ERROR_CORRECT_M,
            "Q": qrcode.constants.ERROR_CORRECT_Q,
            "H": qrcode.constants.ERROR_CORRECT_H,
        }
        code = qrcode.QRCode(
            version=spec.version,
            error_correction=levels[spec.error_correction],
            box_size=1,
            border=spec.border,
        )
        code.add_data(spec.payload)
        try:
            code.make(fit=False)
        except ValueError as error:
            raise ValueError(
                "payload does not fit the requested QR version/error-correction level"
            ) from error
        image = code.make_image(fill_color=spec.foreground, back_color=spec.background)
        return image.convert("RGB").resize(
            (spec.image_size, spec.image_size), Image.Resampling.NEAREST
        )


def decode_readability(image: Image.Image, backend: str | None = None) -> bool | None:
    """Return decoder readability, or ``None`` when decoding is not requested."""
    if backend is None:
        return None
    if backend != "pyzbar":
        raise ValueError("decoder backend must be 'pyzbar' or None")
    try:
        from pyzbar.pyzbar import decode
    except ImportError as error:
        raise OptionalQRDependencyError(
            "QR readability checks require pyzbar. Install with "
            '`pip install -e ".[qr-decode]"`.'
        ) from error
    return bool(decode(image))
