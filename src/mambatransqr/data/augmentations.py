"""PIL and NumPy based image augmentations for QR robustness training."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from random import Random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _rgb(image: Image.Image) -> Image.Image:
    """Return an RGB copy of an image."""
    return image.convert("RGB")


@dataclass(slots=True)
class GaussianNoise:
    """Add zero-mean Gaussian noise to an image.

    Args:
        std: Noise standard deviation on the 0-255 pixel scale.
        seed: Optional seed for reproducible augmentation.
    """

    std: float = 10.0
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize random state and validate configuration."""
        if self.std < 0:
            raise ValueError("std must be non-negative")
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        """Add noise to the image."""
        pixels = np.asarray(_rgb(image), dtype=np.float32)
        noise = self._rng.normal(0.0, self.std, size=pixels.shape)
        return Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))


@dataclass(frozen=True, slots=True)
class GaussianBlur:
    """Blur an image with a Gaussian kernel.

    Args:
        radius: Blur radius in pixels.
    """

    radius: float = 1.0

    def __post_init__(self) -> None:
        """Validate the blur radius."""
        if self.radius < 0:
            raise ValueError("radius must be non-negative")

    def __call__(self, image: Image.Image) -> Image.Image:
        """Blur the image."""
        return image.filter(ImageFilter.GaussianBlur(self.radius))


@dataclass(frozen=True, slots=True)
class MotionBlur:
    """Approximate a horizontal motion blur with a box kernel.

    Args:
        kernel_size: Odd kernel edge length in pixels.
    """

    kernel_size: int = 5

    def __post_init__(self) -> None:
        """Validate the kernel size."""
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply horizontal motion blur."""
        kernel = [0.0] * (self.kernel_size * self.kernel_size)
        offset = (self.kernel_size // 2) * self.kernel_size
        for index in range(self.kernel_size):
            kernel[offset + index] = 1.0 / self.kernel_size
        return image.filter(
            ImageFilter.Kernel((self.kernel_size, self.kernel_size), kernel)
        )


@dataclass(frozen=True, slots=True)
class JPEGCompression:
    """Round-trip an image through JPEG compression.

    Args:
        quality: JPEG quality, from 1 (lowest) to 95 (highest).
    """

    quality: int = 50

    def __post_init__(self) -> None:
        """Validate JPEG quality."""
        if not 1 <= self.quality <= 95:
            raise ValueError("quality must be between 1 and 95")

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply JPEG compression artifacts."""
        buffer = BytesIO()
        _rgb(image).save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()


@dataclass(slots=True)
class PerspectiveTransform:
    """Apply a random perspective warp.

    Args:
        distortion_scale: Maximum corner displacement as a size fraction.
        seed: Optional seed for reproducible augmentation.
    """

    distortion_scale: float = 0.1
    seed: int | None = None
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize random state and validate configuration."""
        if not 0.0 <= self.distortion_scale <= 1.0:
            raise ValueError("distortion_scale must be between 0 and 1")
        self._rng = Random(self.seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        """Warp the image using perturbed source corners."""
        width, height = image.size
        delta = int(min(width, height) * self.distortion_scale)
        source = [(0, 0), (width, 0), (width, height), (0, height)]
        target = [
            (x + self._rng.randint(-delta, delta), y + self._rng.randint(-delta, delta))
            for x, y in source
        ]
        coefficients = _perspective_coefficients(target, source)
        return image.transform(image.size, Image.Transform.PERSPECTIVE, coefficients)


def _perspective_coefficients(
    source: list[tuple[int, int]], target: list[tuple[int, int]]
) -> tuple[float, ...]:
    """Calculate PIL perspective coefficients from four point pairs."""
    matrix: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(source, target, strict=True):
        matrix.extend(
            [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        )
        values.extend([u, v])
    return tuple(np.linalg.solve(np.asarray(matrix), np.asarray(values)).tolist())


@dataclass(slots=True)
class RandomShadow:
    """Overlay a random semi-transparent elliptical shadow.

    Args:
        opacity: Maximum shadow opacity, from 0 to 1.
        seed: Optional seed for reproducible augmentation.
    """

    opacity: float = 0.4
    seed: int | None = None
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize random state and validate configuration."""
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be between 0 and 1")
        self._rng = Random(self.seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply a randomly placed shadow."""
        width, height = image.size
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        x0 = self._rng.randrange(width)
        y0 = self._rng.randrange(height)
        x1 = self._rng.randint(x0 + 1, width)
        y1 = self._rng.randint(y0 + 1, height)
        alpha = int(255 * self._rng.uniform(0.1, self.opacity))
        draw.ellipse((x0, y0, x1, y1), fill=(0, 0, 0, alpha))
        return Image.alpha_composite(_rgb(image).convert("RGBA"), overlay).convert(
            "RGB"
        )


@dataclass(slots=True)
class RandomRain:
    """Draw random rain streaks onto an image.

    Args:
        density: Number of streaks per 10,000 pixels.
        seed: Optional seed for reproducible augmentation.
    """

    density: float = 3.0
    seed: int | None = None
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize random state and validate configuration."""
        if self.density < 0:
            raise ValueError("density must be non-negative")
        self._rng = Random(self.seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply rain streaks."""
        result = _rgb(image).copy()
        draw = ImageDraw.Draw(result)
        width, height = result.size
        count = round(width * height * self.density / 10_000)
        for _ in range(count):
            x = self._rng.randrange(width)
            y = self._rng.randrange(height)
            length = self._rng.randint(3, max(3, height // 8))
            draw.line(
                (x, y, x - length // 3, min(height - 1, y + length)),
                fill=(180, 180, 180),
                width=1,
            )
        return result


@dataclass(slots=True)
class RandomOcclusion:
    """Cover a randomly placed rectangular portion of an image.

    Args:
        max_fraction: Maximum area fraction covered by the rectangle.
        fill: RGB fill color for the occlusion.
        seed: Optional seed for reproducible augmentation.
    """

    max_fraction: float = 0.2
    fill: tuple[int, int, int] = (255, 255, 255)
    seed: int | None = None
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize random state and validate configuration."""
        if not 0.0 < self.max_fraction <= 1.0:
            raise ValueError("max_fraction must be in (0, 1]")
        self._rng = Random(self.seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply rectangular occlusion."""
        result = _rgb(image).copy()
        width, height = result.size
        fraction = self._rng.uniform(0.01, self.max_fraction)
        side = max(1, int((width * height * fraction) ** 0.5))
        box_width = min(width, side)
        box_height = min(height, side)
        x = self._rng.randint(0, width - box_width)
        y = self._rng.randint(0, height - box_height)
        ImageDraw.Draw(result).rectangle(
            (x, y, x + box_width, y + box_height), fill=self.fill
        )
        return result
