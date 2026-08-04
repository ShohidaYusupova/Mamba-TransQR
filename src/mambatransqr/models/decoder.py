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
        refinement_channels: int = 0,
        residual_learning: bool = False,
        residual_scale: float = 1.0,
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
        if refinement_channels < 0:
            raise ValueError("refinement_channels must be non-negative")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
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
        self.residual_learning = residual_learning
        self.residual_scale = residual_scale
        self.refinement = (
            nn.Sequential(
                nn.Conv2d(out_channels, refinement_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(refinement_channels, refinement_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(refinement_channels, out_channels, 3, padding=1),
            )
            if refinement_channels > 0
            else nn.Identity()
        )

    def forward(self, tokens: Tensor, skip_image: Tensor | None = None) -> Tensor:
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
        refined = self.refinement(logits)
        if not self.residual_learning:
            return refined.sigmoid()
        if skip_image is None or skip_image.shape != refined.shape:
            raise ValueError("residual learning requires a matching skip image")
        residual = self.residual_scale * refined.tanh()
        return (skip_image + residual).clamp(0.0, 1.0)

    def initialize_residual_identity(self) -> None:
        """Zero the final refinement layer so residual mode starts as identity."""
        if not self.residual_learning or not isinstance(self.refinement, nn.Sequential):
            return
        final = self.refinement[-1]
        if isinstance(final, nn.Conv2d):
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)
