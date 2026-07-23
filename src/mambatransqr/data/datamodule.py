"""Data module coordinating train, validation, and test datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mambatransqr.data.dataset import TestDataset, TrainDataset, ValidationDataset
from mambatransqr.data.loader import (
    create_evaluation_dataloader,
    create_train_dataloader,
)
from mambatransqr.data.transforms import ImageTransform


@dataclass(slots=True)
class QRDataModule:
    """Build datasets and loaders for standard train/validation/test splits.

    Args:
        train_dir: Optional directory of training images.
        validation_dir: Optional directory of validation images.
        test_dir: Optional directory of test images.
        batch_size: Samples per loader batch.
        num_workers: DataLoader worker count.
        train_transform: Transform applied to training images.
        evaluation_transform: Transform applied to validation and test images.
    """

    train_dir: str | Path | None = None
    validation_dir: str | Path | None = None
    test_dir: str | Path | None = None
    batch_size: int = 32
    num_workers: int = 0
    train_transform: ImageTransform | None = None
    evaluation_transform: ImageTransform | None = None
    train_dataset: TrainDataset | None = field(default=None, init=False)
    validation_dataset: ValidationDataset | None = field(default=None, init=False)
    test_dataset: TestDataset | None = field(default=None, init=False)

    def setup(self, stage: str | None = None) -> None:
        """Create datasets required for a requested stage.

        Args:
            stage: ``fit``, ``validate``, ``test``, or ``None`` for all stages.

        Raises:
            ValueError: If the stage is unsupported.
        """
        valid_stages = {None, "fit", "validate", "test"}
        if stage not in valid_stages:
            raise ValueError(f"unsupported stage: {stage}")
        if stage in (None, "fit") and self.train_dir is not None:
            self.train_dataset = TrainDataset(self.train_dir, self.train_transform)
        if stage in (None, "fit", "validate") and self.validation_dir is not None:
            self.validation_dataset = ValidationDataset(
                self.validation_dir, self.evaluation_transform
            )
        if stage in (None, "test") and self.test_dir is not None:
            self.test_dataset = TestDataset(self.test_dir, self.evaluation_transform)

    def train_dataloader(self) -> Any:
        """Return the training DataLoader.

        Raises:
            RuntimeError: If the training dataset has not been set up.
        """
        if self.train_dataset is None:
            raise RuntimeError(
                "call setup('fit') before requesting the training loader"
            )
        return create_train_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def validation_dataloader(self) -> Any:
        """Return the validation DataLoader."""
        if self.validation_dataset is None:
            raise RuntimeError("call setup('validate') before requesting this loader")
        return create_evaluation_dataloader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> Any:
        """Return the test DataLoader."""
        if self.test_dataset is None:
            raise RuntimeError("call setup('test') before requesting the test loader")
        return create_evaluation_dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
