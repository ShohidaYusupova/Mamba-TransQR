"""Data loading, augmentation, and preprocessing utilities."""

from mambatransqr.data.collate import qr_collate
from mambatransqr.data.damage_generator import QRDamageGenerator
from mambatransqr.data.dataset_builder import (
    DatasetGenerationConfig,
    DatasetIntegrityError,
    SyntheticQRDatasetBuilder,
    load_generation_config,
    validate_dataset,
)
from mambatransqr.data.metadata import QRMetadata
from mambatransqr.data.qr_degradation import QRDegradationEngine, QRDegradationResult
from mambatransqr.data.qr_generator import (
    OptionalQRDependencyError,
    QRGenerationSpec,
    QRGenerator,
    decode_readability,
)
from mambatransqr.data.splits import split_payloads
from mambatransqr.data.datamodule import QRDataModule
from mambatransqr.data.dataset import (
    BaseDataset,
    QRDataset,
    TestDataset,
    TrainDataset,
    ValidationDataset,
)
from mambatransqr.data.loader import (
    create_dataloader,
    create_evaluation_dataloader,
    create_train_dataloader,
)
from mambatransqr.data.sampler import EpochSampler

__all__ = [
    "BaseDataset",
    "DatasetGenerationConfig",
    "DatasetIntegrityError",
    "EpochSampler",
    "QRDamageGenerator",
    "QRDegradationEngine",
    "QRDegradationResult",
    "QRGenerationSpec",
    "QRGenerator",
    "QRMetadata",
    "QRDataModule",
    "QRDataset",
    "TestDataset",
    "TrainDataset",
    "ValidationDataset",
    "create_dataloader",
    "create_evaluation_dataloader",
    "create_train_dataloader",
    "decode_readability",
    "load_generation_config",
    "OptionalQRDependencyError",
    "qr_collate",
    "split_payloads",
    "SyntheticQRDatasetBuilder",
    "validate_dataset",
]
