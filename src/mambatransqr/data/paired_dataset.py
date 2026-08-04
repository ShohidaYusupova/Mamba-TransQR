"""Read-only paired QR dataset used by restoration experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PairedQRDataset(Dataset[dict[str, Any]]):
    """Load existing paired images without generating or modifying a dataset."""

    def __init__(self, root: Path, split: str) -> None:
        self.root = root
        self.records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / split / "metadata").glob("*.json"))
        ]
        if not self.records:
            raise ValueError(f"no paired QR records found for split: {split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "image": _load_tensor(self.root / record["damaged_image_path"]),
            "target": _load_tensor(self.root / record["clean_image_path"]),
            "metadata": record,
        }


def _load_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1) / 255.0
