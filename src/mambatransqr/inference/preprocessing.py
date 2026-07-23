"""Input image preprocessing for model inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def load_image(path: str | Path) -> Image.Image:
    """Load an RGB image without keeping the source file open."""
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def preprocess_image(image: Image.Image, image_size: int | tuple[int, int]) -> Tensor:
    """Resize a PIL image and convert it to a normalized CHW batch tensor.

    Args:
        image: Input RGB image.
        image_size: Output size as an integer or ``(height, width)``.

    Returns:
        Float tensor shaped ``(1, 3, height, width)`` in [0, 1].
    """
    size = (image_size, image_size) if isinstance(image_size, int) else image_size
    resized = image.convert("RGB").resize((size[1], size[0]), Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
