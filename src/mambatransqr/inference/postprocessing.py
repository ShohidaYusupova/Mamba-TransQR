"""Output image postprocessing and persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch import Tensor


def tensor_to_image(tensor: Tensor) -> Image.Image:
    """Convert a CHW or single-item BCHW output tensor to a PIL RGB image."""
    image = tensor.detach().cpu().float()
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError("output tensor must have shape (C, H, W) or (1, C, H, W)")
    array = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).round().astype(np.uint8), mode="RGB")


def save_image(image: Image.Image, path: str | Path) -> Path:
    """Save an image, creating its parent directory as needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination
