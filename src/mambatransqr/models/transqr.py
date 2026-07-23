"""Top-level Mamba-Transformer QR reconstruction model."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from mambatransqr.models.decoder import Decoder
from mambatransqr.models.encoder import Encoder
from mambatransqr.models.layers import initialize_weights


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for :class:`MambaTransQR`.

    Attributes:
        image_size: Square input size or ``(height, width)``.
        patch_size: Square patch size or ``(height, width)``.
        in_channels: Number of input image channels.
        out_channels: Number of reconstructed image channels.
        embed_dim: Patch-token embedding width.
        depth: Number of hybrid fusion blocks.
        num_heads: Attention-head count.
        mlp_ratio: Feed-forward expansion ratio.
        dropout: Dropout probability.
        drop_path_rate: Maximum stochastic-depth probability.
        positional_encoding: ``learnable`` or ``sinusoidal``.
        activation: Activation function name.
    """

    image_size: int | tuple[int, int] = 256
    patch_size: int | tuple[int, int] = 16
    in_channels: int = 3
    out_channels: int = 3
    embed_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    drop_path_rate: float = 0.1
    positional_encoding: str = "learnable"
    activation: str = "gelu"


class MambaTransQR(nn.Module):
    """Hybrid Mamba-Transformer image-to-image model for damaged QR codes.

    Args:
        config: Model hyperparameter configuration.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize the encoder-decoder architecture."""
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = Encoder(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.config.embed_dim,
            depth=self.config.depth,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
            dropout=self.config.dropout,
            drop_path_rate=self.config.drop_path_rate,
            positional_encoding=self.config.positional_encoding,
            activation=self.config.activation,
        )
        self.decoder = Decoder(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            embed_dim=self.config.embed_dim,
            out_channels=self.config.out_channels,
            dropout=self.config.dropout,
        )
        self.apply(initialize_weights)

    def forward(self, images: Tensor) -> Tensor:
        """Reconstruct clean QR images from input images.

        Args:
            images: Tensor shaped ``(batch, channels, height, width)``.

        Returns:
            Reconstructed image tensor with configured output channels.
        """
        return self.decoder(self.encoder(images))
