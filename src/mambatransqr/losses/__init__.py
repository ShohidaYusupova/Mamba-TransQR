"""Modular losses for reconstruction, segmentation, and classification."""

from mambatransqr.losses.charbonnier import CharbonnierLoss
from mambatransqr.losses.combined import CombinedLoss
from mambatransqr.losses.decode_consistency import DecodeConsistencyLoss
from mambatransqr.losses.dice import DiceLoss
from mambatransqr.losses.edge import EdgeLoss
from mambatransqr.losses.focal import FocalLoss
from mambatransqr.losses.multiscale import MultiScaleRestorationLoss, QRLossWeights
from mambatransqr.losses.perceptual import PerceptualLoss
from mambatransqr.losses.qr_structure import QRStructureLoss, qr_structural_masks
from mambatransqr.losses.reconstruction import ReconstructionLoss
from mambatransqr.losses.ssim import SSIMLoss

__all__ = [
    "CharbonnierLoss",
    "CombinedLoss",
    "DecodeConsistencyLoss",
    "DiceLoss",
    "EdgeLoss",
    "FocalLoss",
    "PerceptualLoss",
    "MultiScaleRestorationLoss",
    "QRLossWeights",
    "QRStructureLoss",
    "ReconstructionLoss",
    "SSIMLoss",
    "qr_structural_masks",
]
