"""Dataset implementations for QR restoration and recognition tasks."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from PIL import Image


ImageTransform = Callable[[Image.Image], Any]
DEFAULT_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


class BaseDataset:
    """Base image dataset with deterministic file discovery.

    Args:
        root: Root directory containing image files.
        transform: Optional transform applied to each input image.
        extensions: Lowercase image extensions to include.
    """

    def __init__(
        self,
        root: str | Path,
        transform: ImageTransform | None = None,
        extensions: frozenset[str] = DEFAULT_EXTENSIONS,
    ) -> None:
        """Discover image files under ``root``."""
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {self.root}")
        self.transform = transform
        self.extensions = frozenset(extension.lower() for extension in extensions)
        self.samples = tuple(self._discover_samples())
        if not self.samples:
            raise ValueError(f"no image files found in: {self.root}")

    def _discover_samples(self) -> Iterator[Path]:
        """Yield supported image files in deterministic order."""
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.extensions:
                yield path

    def __len__(self) -> int:
        """Return the number of discovered samples."""
        return len(self.samples)

    def _load_image(self, path: Path) -> Image.Image:
        """Load an image without retaining an open file handle."""
        with Image.open(path) as image:
            return image.convert("RGB").copy()


class QRDataset(BaseDataset):
    """QR dataset that returns an image and associated metadata.

    Args:
        root: Root directory containing source images.
        transform: Optional transform applied to source images.
        target_root: Optional root containing target images with matching relative paths.
        target_transform: Optional transform applied to target images.
    """

    def __init__(
        self,
        root: str | Path,
        transform: ImageTransform | None = None,
        *,
        target_root: str | Path | None = None,
        target_transform: ImageTransform | None = None,
    ) -> None:
        """Initialize a QR image dataset."""
        super().__init__(root, transform)
        self.target_root = Path(target_root) if target_root is not None else None
        if self.target_root is not None and not self.target_root.is_dir():
            raise FileNotFoundError(f"target directory does not exist: {self.target_root}")
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one sample and optional matched target.

        Args:
            index: Zero-based sample index.

        Returns:
            Mapping with ``image``, ``path``, and optional ``target`` entries.
        """
        path = self.samples[index]
        image = self._load_image(path)
        sample: dict[str, Any] = {"image": self.transform(image) if self.transform else image, "path": str(path)}
        if self.target_root is not None:
            relative_path = path.relative_to(self.root)
            target_path = self.target_root / relative_path
            if not target_path.is_file():
                raise FileNotFoundError(f"missing target image: {target_path}")
            target = self._load_image(target_path)
            sample["target"] = self.target_transform(target) if self.target_transform else target
        return sample


class TrainDataset(QRDataset):
    """Named dataset type for training splits."""


class ValidationDataset(QRDataset):
    """Named dataset type for validation splits."""


class TestDataset(QRDataset):
    """Named dataset type for test splits."""


def split_paths(paths: Sequence[Path], fraction: float) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Split ordered paths into train and validation portions.

    Args:
        paths: Ordered sample paths.
        fraction: Validation fraction in the half-open interval [0, 1).

    Returns:
        Train paths followed by validation paths.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    boundary = round(len(paths) * (1.0 - fraction))
    return tuple(paths[:boundary]), tuple(paths[boundary:])
