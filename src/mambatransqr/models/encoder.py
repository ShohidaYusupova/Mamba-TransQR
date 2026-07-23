"""Hybrid Mamba-Transformer encoder."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from mambatransqr.models.fusion_block import HybridFusionBlock
from mambatransqr.models.patch_embed import PatchEmbedding
from mambatransqr.models.positional_encoding import (
    LearnablePositionalEncoding,
    SinusoidalPositionalEncoding,
)


class Encoder(nn.Module):
    """Encode an image into hybrid patch-token representations.

    Args:
        image_size: Input image height and width.
        patch_size: Patch height and width.
        in_channels: Number of image channels.
        embed_dim: Patch token dimension.
        depth: Number of hybrid fusion blocks.
        num_heads: Number of Transformer attention heads.
        mlp_ratio: Feed-forward expansion ratio.
        dropout: Dropout probability.
        drop_path_rate: Maximum linearly scaled stochastic-depth probability.
        positional_encoding: ``learnable`` or ``sinusoidal``.
        activation: Activation function name.
    """

    def __init__(
        self,
        image_size: int | Sequence[int],
        patch_size: int | Sequence[int],
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path_rate: float,
        positional_encoding: str,
        activation: str,
    ) -> None:
        """Initialize patch embedding and hybrid encoder stages."""
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        position_layers: dict[str, type[nn.Module]] = {
            "learnable": LearnablePositionalEncoding,
            "sinusoidal": SinusoidalPositionalEncoding,
        }
        try:
            position_layer = position_layers[positional_encoding.lower()]
        except KeyError as error:
            raise ValueError(
                "positional_encoding must be 'learnable' or 'sinusoidal'"
            ) from error
        self.position = position_layer(
            self.patch_embedding.num_patches, embed_dim, dropout
        )
        rates = [drop_path_rate * index / max(depth - 1, 1) for index in range(depth)]
        self.blocks = nn.ModuleList(
            HybridFusionBlock(
                embed_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=rate,
                activation=activation,
            )
            for rate in rates
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: Tensor) -> Tensor:
        """Encode images into normalized patch tokens."""
        tokens = self.position(self.patch_embedding(images))
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)
