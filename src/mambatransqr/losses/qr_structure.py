"""QR module-structure aware restoration objective."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from mambatransqr.losses.edge import EdgeLoss


def qr_structural_masks(
    images: Tensor, metadata: list[dict[str, Any]] | None = None
) -> dict[str, Tensor]:
    """Build finder/alignment/timing/quiet masks from metadata or image edges."""
    batch, _, height, width = images.shape
    device, dtype = images.device, images.dtype
    masks = {
        name: torch.zeros((batch, 1, height, width), device=device, dtype=dtype)
        for name in ("finder", "alignment", "timing", "quiet", "boundary")
    }
    for index in range(batch):
        item = metadata[index] if metadata and index < len(metadata) else {}
        version = int(item.get("qr_version", 1)) if isinstance(item, dict) else 1
        border = (
            int(
                item.get(
                    "border",
                    item.get("degradation_parameters", {}).get("quiet_zone_modules", 4),
                )
            )
            if isinstance(item, dict)
            else 4
        )
        modules, total = 17 + 4 * version, 17 + 4 * version + 2 * border

        def box(
            row: int,
            col: int,
            size: int,
            border: int = border,
            total: int = total,
        ) -> tuple[slice, slice]:
            return slice(
                round((row + border) * height / total),
                round((row + border + size) * height / total),
            ), slice(
                round((col + border) * width / total),
                round((col + border + size) * width / total),
            )

        for row, col in ((0, 0), (0, modules - 7), (modules - 7, 0)):
            y, x = box(row, col, 7)
            masks["finder"][index, :, y, x] = 1
        y, x = box(6, 8, max(1, modules - 16))
        masks["timing"][index, :, y, x] = 1
        y, x = box(8, 6, max(1, modules - 16))
        masks["timing"][index, :, y, x] = 1
        if version > 1:
            y, x = box(modules - 9, modules - 9, 5)
            masks["alignment"][index, :, y, x] = 1
        quiet = max(1, round(border * min(height, width) / total))
        masks["quiet"][index, :, :quiet] = 1
        masks["quiet"][index, :, -quiet:] = 1
        masks["quiet"][index, :, :, :quiet] = 1
        masks["quiet"][index, :, :, -quiet:] = 1
    gray = images.mean(dim=1, keepdim=True)
    gx = torch.nn.functional.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1))
    gy = torch.nn.functional.pad(
        (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1)
    )
    masks["boundary"] = (gx + gy > (gx + gy).mean(dim=(2, 3), keepdim=True)).to(dtype)
    return masks


class QRStructureLoss(nn.Module):
    """Weighted L1 and edge loss over QR structural regions."""

    def __init__(
        self,
        finder: float = 2.0,
        alignment: float = 1.5,
        timing: float = 1.5,
        quiet_zone: float = 1.0,
        boundaries: float = 1.5,
    ) -> None:
        super().__init__()
        self.weights = {
            "finder": finder,
            "alignment": alignment,
            "timing": timing,
            "quiet": quiet_zone,
            "boundary": boundaries,
        }
        self.edge = EdgeLoss()

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        metadata: list[dict[str, Any]] | None = None,
    ) -> Tensor:
        masks = qr_structural_masks(targets, metadata)
        weight = sum(
            (self.weights[name] * mask for name, mask in masks.items()),
            torch.ones_like(masks["finder"]),
        )
        pixel = ((predictions - targets).abs() * weight).sum() / (
            weight.sum() * predictions.shape[1]
        ).clamp_min(1)
        return pixel + self.weights["boundary"] * self.edge(
            predictions * masks["boundary"], targets * masks["boundary"]
        )
