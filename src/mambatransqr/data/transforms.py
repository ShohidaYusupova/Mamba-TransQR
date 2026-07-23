"""Composable torchvision-backed image transforms.

The wrappers in this module keep transform configuration serializable while
deferring the torchvision import until a transform is applied.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


ImageTransform = Callable[[Any], Any]


def _functional() -> Any:
    """Return torchvision's functional transform module.

    Raises:
        ImportError: If the optional torchvision dependency is unavailable.
    """
    try:
        from torchvision.transforms import functional
    except ImportError as error:
        message = "torchvision is required to apply data transforms."
        raise ImportError(message) from error
    return functional


@dataclass(frozen=True, slots=True)
class Resize:
    """Resize an image using torchvision.

    Args:
        size: Output height and width, or the shortest-edge size.
    """

    size: int | tuple[int, int]

    def __call__(self, image: Any) -> Any:
        """Apply the resize transform."""
        return _functional().resize(image, self.size)


@dataclass(frozen=True, slots=True)
class CenterCrop:
    """Crop the center of an image using torchvision.

    Args:
        size: Output height and width, or a square edge length.
    """

    size: int | tuple[int, int]

    def __call__(self, image: Any) -> Any:
        """Apply the center crop transform."""
        return _functional().center_crop(image, self.size)


@dataclass(frozen=True, slots=True)
class RandomCrop:
    """Apply torchvision's random crop transform.

    Args:
        size: Output height and width, or a square edge length.
        padding: Optional pixel padding applied before cropping.
    """

    size: int | tuple[int, int]
    padding: int | tuple[int, int] | tuple[int, int, int, int] | None = None

    def __call__(self, image: Any) -> Any:
        """Apply the random crop transform."""
        from torchvision.transforms import RandomCrop as TorchRandomCrop

        return TorchRandomCrop(self.size, padding=self.padding)(image)


@dataclass(frozen=True, slots=True)
class RandomHorizontalFlip:
    """Randomly mirror an image horizontally using torchvision.

    Args:
        probability: Probability of mirroring the input image.
    """

    probability: float = 0.5

    def __post_init__(self) -> None:
        """Validate the probability."""
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")

    def __call__(self, image: Any) -> Any:
        """Apply the random horizontal flip transform."""
        from torchvision.transforms import RandomHorizontalFlip as TorchFlip

        return TorchFlip(p=self.probability)(image)


@dataclass(frozen=True, slots=True)
class Normalize:
    """Normalize a tensor image using torchvision.

    Args:
        mean: Per-channel mean values.
        std: Per-channel standard-deviation values.
    """

    mean: Sequence[float]
    std: Sequence[float]

    def __post_init__(self) -> None:
        """Validate channel configuration."""
        if len(self.mean) != len(self.std):
            raise ValueError("mean and std must have the same number of channels")
        if any(value <= 0 for value in self.std):
            raise ValueError("std values must be positive")

    def __call__(self, image: Any) -> Any:
        """Apply normalization."""
        return _functional().normalize(image, list(self.mean), list(self.std))


@dataclass(frozen=True, slots=True)
class ToTensor:
    """Convert a PIL image or array to a tensor using torchvision."""

    def __call__(self, image: Any) -> Any:
        """Convert the input to a tensor."""
        return _functional().to_tensor(image)


@dataclass(frozen=True, slots=True)
class Compose:
    """Compose multiple image transforms.

    Args:
        transforms: Transforms applied in declaration order.
    """

    transforms: Sequence[ImageTransform]

    def __call__(self, image: Any) -> Any:
        """Apply all configured transforms."""
        result = image
        for transform in self.transforms:
            result = transform(result)
        return result
