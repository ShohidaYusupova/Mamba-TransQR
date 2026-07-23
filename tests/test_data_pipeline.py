"""Tests for the Mamba-TransQR data pipeline."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from mambatransqr.data.augmentations import (
    GaussianBlur,
    GaussianNoise,
    JPEGCompression,
    MotionBlur,
    PerspectiveTransform,
    RandomOcclusion,
    RandomRain,
    RandomShadow,
)
from mambatransqr.data.collate import qr_collate
from mambatransqr.data.damage_generator import QRDamageGenerator
from mambatransqr.data.dataset import QRDataset, split_paths
from mambatransqr.data.sampler import EpochSampler


@pytest.fixture
def image() -> Image.Image:
    """Return a simple, non-uniform RGB image."""
    result = Image.new("RGB", (32, 32), "white")
    result.putpixel((0, 0), (0, 0, 0))
    return result


@pytest.mark.parametrize(
    "augmentation",
    [
        GaussianNoise(std=10.0, seed=1),
        GaussianBlur(radius=1.0),
        MotionBlur(kernel_size=3),
        JPEGCompression(quality=40),
        PerspectiveTransform(distortion_scale=0.1, seed=1),
        RandomShadow(opacity=0.4, seed=1),
        RandomRain(density=2.0, seed=1),
        RandomOcclusion(max_fraction=0.2, seed=1),
    ],
)
def test_augmentation_preserves_image_shape(
    image: Image.Image,
    augmentation: Callable[[Image.Image], Image.Image],
) -> None:
    """Every augmentation returns an RGB image with the input dimensions."""
    result = augmentation(image)
    assert result.mode == "RGB"
    assert result.size == image.size


def test_damage_generator_supports_all_named_operations(image: Image.Image) -> None:
    """Every configured damage operation can be applied in isolation."""
    generator = QRDamageGenerator(count=0, seed=1)
    for name in generator.available_damages():
        result = generator.apply(name, image)
        assert result.size == image.size


def test_dataset_discovers_images_and_matches_targets(tmp_path: Path) -> None:
    """The QR dataset returns source, target, and source-path metadata."""
    source = tmp_path / "source" / "nested"
    target = tmp_path / "target" / "nested"
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(source / "qr.png")
    Image.new("RGB", (8, 8), "black").save(target / "qr.png")

    dataset = QRDataset(tmp_path / "source", target_root=tmp_path / "target")
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["image"].size == (8, 8)
    assert sample["target"].getpixel((0, 0)) == (0, 0, 0)
    assert sample["path"].endswith("qr.png")


def test_collate_keeps_optional_values() -> None:
    """Collation handles samples with different optional keys."""
    batch = qr_collate([{"image": "one"}, {"image": "two", "target": "clean"}])
    assert batch == {"image": ["one", "two"], "target": [None, "clean"]}


def test_epoch_sampler_is_deterministic_and_changes_per_epoch() -> None:
    """Epoch-derived ordering is deterministic and changes across epochs."""
    sampler = EpochSampler(10, seed=9)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    sampler.set_epoch(1)
    assert first != second
    assert second == list(sampler)


def test_split_paths_rejects_invalid_fraction() -> None:
    """Splitting rejects invalid validation fractions."""
    with pytest.raises(ValueError, match="fraction"):
        split_paths([], 1.0)
