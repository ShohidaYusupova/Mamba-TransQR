"""Image decoder that reconstructs pixels from patch tokens."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from mambatransqr.models.patch_embed import _pair


class Decoder(nn.Module):
    """Decode patch tokens to an image using learned patch reconstruction.

    Args:
        image_size: Reconstructed image height and width.
        patch_size: Patch height and width.
        embed_dim: Token embedding dimension.
        out_channels: Number of output image channels.
        dropout: Dropout probability before pixel prediction.
    """

    def __init__(
        self,
        image_size: int | Sequence[int],
        patch_size: int | Sequence[int],
        embed_dim: int,
        out_channels: int = 3,
        dropout: float = 0.0,
    ) -> None:
        """Initialize patch reconstruction layers."""
        super().__init__()
        self.image_size = _pair(image_size)
        self.patch_size = _pair(patch_size)
        if any(
            image % patch
            for image, patch in zip(self.image_size, self.patch_size, strict=True)
        ):
            raise ValueError("image_size must be divisible by patch_size")
        if embed_dim < 1 or out_channels < 1:
            raise ValueError("embed_dim and out_channels must be positive")
        self.grid_size = tuple(
            image // patch
            for image, patch in zip(self.image_size, self.patch_size, strict=True)
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        patch_pixels = self.patch_size[0] * self.patch_size[1] * out_channels
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(embed_dim, patch_pixels)
        )
        self.out_channels = out_channels

    def forward(self, tokens: Tensor) -> Tensor:
        """Reconstruct an image tensor from patch tokens.

        Args:
            tokens: Tensor shaped ``(batch, num_patches, embed_dim)``.

        Returns:
            Image tensor shaped ``(batch, channels, height, width)``.
        """
        if tokens.ndim != 3 or tokens.shape[1] != self.num_patches:
            raise ValueError(f"expected {self.num_patches} patch tokens")
        batch_size = tokens.shape[0]
        patches = self.head(self.norm(tokens))
        grid_height, grid_width = self.grid_size
        patch_height, patch_width = self.patch_size
        patches = patches.view(
            batch_size,
            grid_height,
            grid_width,
            self.out_channels,
            patch_height,
            patch_width,
        )
        logits = patches.permute(0, 3, 1, 4, 2, 5).reshape(
            batch_size,
            self.out_channels,
            self.image_size[0],
            self.image_size[1],
        )
        return logits.sigmoid()
