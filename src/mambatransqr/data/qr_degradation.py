"""Module-aware degradations for QR restoration training."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from mambatransqr.data.augmentations import (
    GaussianBlur,
    GaussianNoise,
    JPEGCompression,
)


@dataclass(frozen=True, slots=True)
class QRDegradationResult:
    """A degraded QR image and the operations used to produce it."""

    image: Image.Image
    operations: tuple[str, ...]


@dataclass(slots=True)
class QRDegradationEngine:
    """Generate QR-specific, module-aware synthetic degradations.

    The engine uses an explicit QR module count so corruption aligns to QR
    cells instead of arbitrary image pixels. ``module_count`` excludes any
    quiet zone; set ``quiet_zone_modules`` to match the rasterized source.

    Args:
        module_count: Number of QR data modules per side (for example 21 for
            version 1), excluding the quiet zone.
        quiet_zone_modules: Quiet-zone width represented in the source image.
        enabled: Operations eligible for each degradation sample.
        count: Number of distinct operations sampled per call.
        seed: Optional seed for reproducible degradation sequences.
    """

    module_count: int = 21
    quiet_zone_modules: int = 4
    enabled: tuple[str, ...] = (
        "module_dropout",
        "module_occlusion",
        "finder_damage",
        "ink_spread",
        "print_scan",
    )
    count: int = 2
    seed: int | None = None
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the QR grid configuration and initialize random state."""
        if self.module_count < 7:
            raise ValueError("module_count must be at least 7")
        if self.quiet_zone_modules < 0:
            raise ValueError("quiet_zone_modules must be non-negative")
        invalid = set(self.enabled) - set(self.available_degradations())
        if invalid:
            raise ValueError(f"unknown QR degradations: {sorted(invalid)}")
        if not 0 <= self.count <= len(self.enabled):
            raise ValueError(
                "count must be between 0 and the enabled degradation count"
            )
        self._rng = Random(self.seed)

    @staticmethod
    def available_degradations() -> tuple[str, ...]:
        """Return operation names supported by this QR-specific engine."""
        return (
            "module_dropout",
            "module_occlusion",
            "finder_damage",
            "ink_spread",
            "print_scan",
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return a degraded image for transform-style use."""
        return self.degrade(image).image

    def degrade(self, image: Image.Image) -> QRDegradationResult:
        """Apply sampled degradations and return the image plus provenance."""
        result = image.convert("RGB")
        self._validate_image(result)
        selected = tuple(self._rng.sample(self.enabled, self.count))
        for name in selected:
            result = self.apply(name, result)
        return QRDegradationResult(image=result, operations=selected)

    def apply(self, name: str, image: Image.Image) -> Image.Image:
        """Apply a named QR degradation while preserving image dimensions."""
        handlers = {
            "module_dropout": self._module_dropout,
            "module_occlusion": self._module_occlusion,
            "finder_damage": self._finder_damage,
            "ink_spread": self._ink_spread,
            "print_scan": self._print_scan,
        }
        self._validate_image(image)
        try:
            return handlers[name](image.convert("RGB"))
        except KeyError as error:
            raise ValueError(f"unsupported QR degradation: {name}") from error

    def _validate_image(self, image: Image.Image) -> None:
        width, height = image.size
        if width != height:
            raise ValueError("QR degradation requires a square image")
        total_modules = self.module_count + 2 * self.quiet_zone_modules
        if width < total_modules:
            raise ValueError("image is too small for the configured QR module grid")

    def _module_box(
        self, image_side: int, row: int, column: int, size: int = 1
    ) -> tuple[int, int, int, int]:
        """Return a module-aligned pixel box using rounded grid boundaries."""
        total_modules = self.module_count + 2 * self.quiet_zone_modules
        offset = self.quiet_zone_modules
        left = round((column + offset) * image_side / total_modules)
        top = round((row + offset) * image_side / total_modules)
        right = round((column + size + offset) * image_side / total_modules)
        bottom = round((row + size + offset) * image_side / total_modules)
        return left, top, max(left + 1, right), max(top + 1, bottom)

    def _module_dropout(self, image: Image.Image) -> Image.Image:
        """Flip a sparse, module-aligned subset of QR cells."""
        result = image.copy()
        draw = ImageDraw.Draw(result)
        dropout_count = max(1, round(self.module_count * self.module_count * 0.03))
        for _ in range(dropout_count):
            row = self._rng.randrange(self.module_count)
            column = self._rng.randrange(self.module_count)
            draw.rectangle(
                self._module_box(image.size[0], row, column),
                fill=self._rng.choice([(0, 0, 0), (255, 255, 255)]),
            )
        return result

    def _module_occlusion(self, image: Image.Image) -> Image.Image:
        """Apply one or two contiguous module-aligned occlusions."""
        result = image.copy()
        draw = ImageDraw.Draw(result)
        for _ in range(self._rng.randint(1, 2)):
            side = self._rng.randint(2, min(6, self.module_count))
            row = self._rng.randint(0, self.module_count - side)
            column = self._rng.randint(0, self.module_count - side)
            draw.rectangle(
                self._module_box(image.size[0], row, column, side),
                fill=self._rng.choice([(0, 0, 0), (255, 255, 255)]),
            )
        return result

    def _finder_damage(self, image: Image.Image) -> Image.Image:
        """Corrupt one finder pattern, a QR-specific decoding failure mode."""
        result = image.copy()
        row, column = self._rng.choice(
            ((0, 0), (0, self.module_count - 7), (self.module_count - 7, 0))
        )
        ImageDraw.Draw(result).rectangle(
            self._module_box(image.size[0], row + 2, column + 2, 3),
            fill=self._rng.choice([(0, 0, 0), (255, 255, 255)]),
        )
        return result

    def _ink_spread(self, image: Image.Image) -> Image.Image:
        """Simulate print ink spreading from dark QR modules."""
        return image.filter(ImageFilter.MinFilter(3))

    def _print_scan(self, image: Image.Image) -> Image.Image:
        """Simulate a low-resolution print/scan cycle and compression noise."""
        width, height = image.size
        scale = self._rng.uniform(0.55, 0.85)
        reduced = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        )
        result = reduced.resize((width, height), Image.Resampling.BILINEAR)
        result = ImageEnhance.Contrast(result).enhance(self._rng.uniform(0.55, 0.9))
        result = GaussianBlur(self._rng.uniform(0.2, 1.0))(result)
        result = GaussianNoise(std=self._rng.uniform(1.0, 10.0))(result)
        return JPEGCompression(quality=self._rng.randint(20, 65))(result)
