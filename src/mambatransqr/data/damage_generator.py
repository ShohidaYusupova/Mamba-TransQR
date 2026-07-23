"""Synthetic QR-code damage generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from PIL import Image, ImageDraw, ImageEnhance

from mambatransqr.data.augmentations import GaussianBlur, GaussianNoise, JPEGCompression


@dataclass(slots=True)
class QRDamageGenerator:
    """Generate realistic image degradation for QR robustness experiments.

    Args:
        enabled: Damage operation names allowed during generation.
        count: Number of independently sampled operations per image.
        seed: Optional random seed.
    """

    enabled: tuple[str, ...] = (
        "scratches",
        "missing_blocks",
        "blur",
        "low_illumination",
        "noise",
        "compression_artifacts",
    )
    count: int = 2
    seed: int | None = None
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the configured operations and initialize random state."""
        available = set(self.available_damages())
        invalid = set(self.enabled) - available
        if invalid:
            raise ValueError(f"unknown damage operations: {sorted(invalid)}")
        if not 0 <= self.count <= len(self.enabled):
            raise ValueError("count must be between 0 and the number of enabled damages")
        self._rng = Random(self.seed)

    @staticmethod
    def available_damages() -> tuple[str, ...]:
        """Return supported damage-operation names."""
        return (
            "scratches",
            "missing_blocks",
            "blur",
            "low_illumination",
            "noise",
            "compression_artifacts",
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply a sampled set of damage operations.

        Args:
            image: Input QR image.

        Returns:
            A damaged RGB image.
        """
        result = image.convert("RGB")
        for name in self._rng.sample(self.enabled, self.count):
            result = self.apply(name, result)
        return result

    def apply(self, name: str, image: Image.Image) -> Image.Image:
        """Apply one named damage operation.

        Args:
            name: A member of :meth:`available_damages`.
            image: Input image.

        Returns:
            Damaged RGB image.

        Raises:
            ValueError: If ``name`` is not supported.
        """
        handlers = {
            "scratches": self._scratches,
            "missing_blocks": self._missing_blocks,
            "blur": self._blur,
            "low_illumination": self._low_illumination,
            "noise": self._noise,
            "compression_artifacts": self._compression_artifacts,
        }
        try:
            return handlers[name](image)
        except KeyError as error:
            raise ValueError(f"unsupported damage operation: {name}") from error

    def _scratches(self, image: Image.Image) -> Image.Image:
        """Draw a small number of thin scratch lines."""
        result = image.copy()
        width, height = result.size
        draw = ImageDraw.Draw(result)
        for _ in range(self._rng.randint(1, 4)):
            x0, y0 = self._rng.randrange(width), self._rng.randrange(height)
            x1 = min(width - 1, max(0, x0 + self._rng.randint(-width // 2, width // 2)))
            y1 = min(height - 1, max(0, y0 + self._rng.randint(-height // 2, height // 2)))
            color = self._rng.choice([(0, 0, 0), (255, 255, 255)])
            draw.line((x0, y0, x1, y1), fill=color, width=self._rng.randint(1, 3))
        return result

    def _missing_blocks(self, image: Image.Image) -> Image.Image:
        """Remove random square regions from the QR image."""
        result = image.copy()
        width, height = result.size
        draw = ImageDraw.Draw(result)
        max_side = max(1, min(width, height) // 5)
        for _ in range(self._rng.randint(1, 4)):
            side = self._rng.randint(1, max_side)
            x = self._rng.randint(0, width - side)
            y = self._rng.randint(0, height - side)
            draw.rectangle((x, y, x + side, y + side), fill=(255, 255, 255))
        return result

    def _blur(self, image: Image.Image) -> Image.Image:
        """Apply Gaussian blur."""
        return GaussianBlur(self._rng.uniform(0.5, 2.0))(image)

    def _low_illumination(self, image: Image.Image) -> Image.Image:
        """Reduce image brightness."""
        return ImageEnhance.Brightness(image).enhance(self._rng.uniform(0.25, 0.75))

    def _noise(self, image: Image.Image) -> Image.Image:
        """Apply Gaussian sensor noise."""
        return GaussianNoise(std=self._rng.uniform(5.0, 30.0))(image)

    def _compression_artifacts(self, image: Image.Image) -> Image.Image:
        """Apply JPEG compression artifacts."""
        return JPEGCompression(quality=self._rng.randint(10, 50))(image)
