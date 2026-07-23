"""Patch embedding layers for image inputs."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    """Convert an integer or two values into a validated height-width pair."""
    if isinstance(value, int):
        result = (value, value)
    else:
        if len(value) != 2:
            raise ValueError("image and patch sizes must contain exactly two values")
        result = (int(value[0]), int(value[1]))
    if any(item <= 0 for item in result):
        raise ValueError("image and patch sizes must be positive")
    return result


class PatchEmbedding(nn.Module):
    """Convert images into a sequence of learned patch embeddings.

    Args:
        image_size: Input image height and width.
        patch_size: Patch height and width.
        in_channels: Number of input image channels.
        embed_dim: Embedding dimension for each patch.
    """

    def __init__(
        self,
        image_size: int | Sequence[int] = 256,
        patch_size: int | Sequence[int] = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
    ) -> None:
        """Initialize patch projection."""
        super().__init__()
        self.image_size = _pair(image_size)
        self.patch_size = _pair(patch_size)
        if any(
            image % patch
            for image, patch in zip(self.image_size, self.patch_size, strict=True)
        ):
            raise ValueError("image_size must be divisible by patch_size")
        if in_channels < 1 or embed_dim < 1:
            raise ValueError("in_channels and embed_dim must be positive")
        self.grid_size = tuple(
            image // patch
            for image, patch in zip(self.image_size, self.patch_size, strict=True)
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Embed a batch of images.

        Args:
            inputs: Image tensor shaped ``(batch, channels, height, width)``.

        Returns:
            Patch tokens shaped ``(batch, num_patches, embed_dim)``.
        """
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape (batch, channels, height, width)")
        if tuple(inputs.shape[-2:]) != self.image_size:
            raise ValueError(
                f"expected image size {self.image_size}, got {tuple(inputs.shape[-2:])}"
            )
        return self.projection(inputs).flatten(2).transpose(1, 2)
