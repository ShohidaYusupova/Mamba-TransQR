"""Tests for the Mamba-TransQR data pipeline."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image, ImageChops

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
from mambatransqr.data.dataset_builder import (
    DatasetGenerationConfig,
    SyntheticQRDatasetBuilder,
    validate_dataset,
)
from mambatransqr.data.qr_degradation import QRDegradationEngine
from mambatransqr.data.qr_generator import QRGenerationSpec, QRGenerator
from mambatransqr.data.sampler import EpochSampler
from mambatransqr.data.splits import split_payloads


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


def test_qr_degradation_engine_preserves_grid_and_records_operations() -> None:
    """QR-specific degradations are module-aware and expose sampled provenance."""
    image = Image.new("RGB", (116, 116), "white")
    engine = QRDegradationEngine(
        module_count=21,
        quiet_zone_modules=4,
        enabled=("module_dropout", "finder_damage"),
        count=2,
        seed=7,
    )
    result = engine.degrade(image)
    assert result.image.size == image.size
    assert set(result.operations) == {"finder_damage", "module_dropout"}
    assert ImageChops.difference(result.image, image).getbbox() is not None


@pytest.mark.parametrize("name", QRDegradationEngine.available_degradations())
def test_qr_degradation_engine_supports_named_operations(name: str) -> None:
    """Every QR-specific operation preserves the input raster dimensions."""
    image = Image.new("RGB", (116, 116), "white")
    engine = QRDegradationEngine(count=0, seed=1)
    assert engine.apply(name, image).size == image.size


def test_qr_degradation_rejects_non_square_images() -> None:
    """Module-grid corruption rejects images that cannot represent square QR codes."""
    with pytest.raises(ValueError, match="square"):
        QRDegradationEngine().degrade(Image.new("RGB", (32, 24), "white"))


def test_payload_splits_are_deterministic_and_do_not_leak() -> None:
    """All variants of one payload stay in one reproducible split."""
    payloads = ["same", "same", "other", "third"]
    first = split_payloads(payloads, seed=8)
    assert first == split_payloads(payloads, seed=8)
    assert first["same"] in {"train", "validation", "test"}


@pytest.mark.parametrize("level", ("L", "M", "Q", "H"))
def test_qr_generator_supports_all_error_correction_levels(level: str) -> None:
    """Renderer accepts all standard correction levels when qrcode is installed."""
    pytest.importorskip("qrcode")
    image = QRGenerator.render(QRGenerationSpec(1, level, "numeric", "123", 64))
    assert image.size == (64, 64)


@pytest.mark.parametrize("version", (1, 10, 40))
def test_qr_generator_supports_multiple_versions(version: int) -> None:
    """Renderer accepts QR versions across the standard 1--40 range."""
    pytest.importorskip("qrcode")
    image = QRGenerator.render(QRGenerationSpec(version, "L", "numeric", "123", 64))
    assert image.size == (64, 64)


def test_qr_generator_normalizes_payload_overflow() -> None:
    """Package-specific QR overflow errors are exposed as the public ValueError."""
    pytest.importorskip("qrcode")
    with pytest.raises(ValueError, match="does not fit"):
        QRGenerator.render(QRGenerationSpec(1, "H", "byte", "x" * 200, 64))


def test_synthetic_builder_creates_pairs_metadata_and_validates(tmp_path: Path) -> None:
    """Builder emits paired PNGs, complete metadata, and a valid manifest."""
    pytest.importorskip("qrcode")
    config = DatasetGenerationConfig(
        samples=3,
        seed=3,
        versions=(1,),
        error_correction_levels=("L",),
        payload_types=("numeric",),
        payload_length=3,
        image_size=64,
        severities=("mild",),
    )
    records = SyntheticQRDatasetBuilder(config).build(tmp_path)
    assert len(records) == 3
    assert (tmp_path / "dataset_manifest.csv").is_file()
    assert validate_dataset(tmp_path)["records"] == 3
    record = records[0]
    assert record.sample_id and record.damaged_image_path is not None
    assert (tmp_path / record.clean_image_path).is_file()
    assert (tmp_path / record.damaged_image_path).is_file()


def test_synthetic_builder_is_deterministic(tmp_path: Path) -> None:
    """A fixed seed produces the same payloads, QR settings, and split mapping."""
    pytest.importorskip("qrcode")
    config = DatasetGenerationConfig(
        samples=2,
        seed=13,
        versions=(1,),
        error_correction_levels=("L",),
        payload_types=("numeric",),
        payload_length=3,
        image_size=64,
        severities=("mild",),
    )
    first = SyntheticQRDatasetBuilder(config).build(tmp_path / "first")
    second = SyntheticQRDatasetBuilder(config).build(tmp_path / "second")
    assert [record.to_dict() for record in first] == [
        record.to_dict() for record in second
    ]


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
