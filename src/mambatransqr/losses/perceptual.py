"""Optional feature-space perceptual loss."""

from __future__ import annotations

from torch import Tensor, nn


class PerceptualLoss(nn.Module):
    """Compare images in a frozen VGG feature space.

    Args:
        layers: Number of initial VGG16 feature layers to use.
    """

    def __init__(self, layers: int = 8) -> None:
        """Initialize a frozen, no-download VGG feature extractor."""
        super().__init__()
        try:
            from torchvision.models import vgg16
        except ImportError as error:
            raise ImportError("PerceptualLoss requires torchvision.") from error
        if layers < 1:
            raise ValueError("layers must be positive")
        self.features = vgg16(weights=None).features[:layers].eval()
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Return L1 feature-space distance."""
        return nn.functional.l1_loss(self.features(predictions), self.features(targets))
