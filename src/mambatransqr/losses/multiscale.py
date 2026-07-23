"""Composable multi-scale QR restoration objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from mambatransqr.losses.charbonnier import CharbonnierLoss
from mambatransqr.losses.decode_consistency import DecodeConsistencyLoss
from mambatransqr.losses.edge import EdgeLoss
from mambatransqr.losses.qr_structure import QRStructureLoss
from mambatransqr.losses.reconstruction import ReconstructionLoss
from mambatransqr.losses.ssim import SSIMLoss


@dataclass(frozen=True, slots=True)
class QRLossWeights:
    reconstruction_weight: float = 1.0
    ssim_weight: float = 0.2
    edge_weight: float = 0.2
    structure_weight: float = 1.0
    decode_consistency_weight: float = 0.2
    perceptual_weight: float = 0.0


class MultiScaleRestorationLoss(nn.Module):
    """Combine reconstruction, SSIM, edges, QR structure, and decode surrogate."""

    def __init__(
        self,
        weights: QRLossWeights | None = None,
        reconstruction: str = "charbonnier",
        perceptual: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or QRLossWeights()
        self.reconstruction = (
            CharbonnierLoss()
            if reconstruction == "charbonnier"
            else ReconstructionLoss("l1")
        )
        self.ssim, self.edge, self.structure = SSIMLoss(), EdgeLoss(), QRStructureLoss()
        self.decode, self.perceptual = DecodeConsistencyLoss(), perceptual
        self.last_components: dict[str, Tensor] = {}

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        metadata: list[dict[str, Any]] | None = None,
    ) -> Tensor:
        components = {
            "reconstruction": self.reconstruction(predictions, targets),
            "ssim": self.ssim(predictions, targets),
            "edge": self.edge(predictions, targets),
            "structure": self.structure(predictions, targets, metadata),
            "decode_consistency": self.decode(predictions, targets),
        }
        if self.perceptual is not None and self.weights.perceptual_weight:
            components["perceptual"] = self.perceptual(predictions, targets)
        total = sum(
            getattr(self.weights, f"{name}_weight") * value
            for name, value in components.items()
        )
        self.last_components = components
        return total
