"""Top-level Mamba-Transformer QR reconstruction model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn

from mambatransqr.models.decoder import Decoder
from mambatransqr.models.encoder import Encoder
from mambatransqr.models.layers import initialize_weights
from mambatransqr.models.mamba_block import mamba_ssm_version


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
        mamba_backend: ``mamba_ssm`` (official) or ``lightweight`` compatibility backend.
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
    mamba_backend: Literal["mamba_ssm", "lightweight"] = "mamba_ssm"
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    use_mamba: bool = True
    decoder_refinement_channels: int = 0
    residual_learning: bool = False
    residual_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate Mamba backend settings before model construction."""
        if self.mamba_backend not in {"mamba_ssm", "lightweight"}:
            raise ValueError("mamba_backend must be 'mamba_ssm' or 'lightweight'")
        if min(self.mamba_d_state, self.mamba_d_conv, self.mamba_expand) < 1:
            raise ValueError("Mamba dimensions must be positive")
        if self.decoder_refinement_channels < 0:
            raise ValueError("decoder_refinement_channels must be non-negative")
        if self.residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")

    def architecture_identity(self) -> dict[str, str | None]:
        """Return backend metadata suitable for reports and checkpoints."""
        return {
            "mamba_backend": self.mamba_backend,
            "mamba_implementation": (
                "official_mamba_ssm"
                if self.mamba_backend == "mamba_ssm"
                else "lightweight_state_space"
            ),
            "mamba_ssm_version": (
                mamba_ssm_version() if self.mamba_backend == "mamba_ssm" else None
            ),
        }


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
            mamba_backend=self.config.mamba_backend,
            mamba_d_state=self.config.mamba_d_state,
            mamba_d_conv=self.config.mamba_d_conv,
            mamba_expand=self.config.mamba_expand,
            use_mamba=self.config.use_mamba,
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
            refinement_channels=self.config.decoder_refinement_channels,
            residual_learning=self.config.residual_learning,
            residual_scale=self.config.residual_scale,
        )
        self._initialize_weights()
        self.decoder.initialize_residual_identity()

    def _initialize_weights(self) -> None:
        """Initialize project layers without overwriting official Mamba defaults."""
        official_mamba_modules = {
            id(child)
            for block in self.encoder.blocks
            if getattr(block.mamba, "backend", None) == "mamba_ssm"
            for child in block.mamba.implementation.modules()  # type: ignore[union-attr]
        }
        for module in self.modules():
            if id(module) not in official_mamba_modules:
                initialize_weights(module)

    def architecture_identity(self) -> dict[str, str | None]:
        """Return the exact Mamba implementation identity used by this model."""
        return self.config.architecture_identity()

    def forward(self, images: Tensor) -> Tensor:
        """Reconstruct clean QR images from input images.

        Args:
            images: Tensor shaped ``(batch, channels, height, width)``.

        Returns:
            Reconstructed image tensor with configured output channels.
        """
        return self.decoder(
            self.encoder(images), images if self.config.residual_learning else None
        )
