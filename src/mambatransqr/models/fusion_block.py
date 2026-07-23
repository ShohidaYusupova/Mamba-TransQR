"""Hybrid Mamba-Transformer fusion block."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from mambatransqr.models.layers import DropPath
from mambatransqr.models.mamba_block import MambaBlock
from mambatransqr.models.transformer_block import TransformerBlock


class HybridFusionBlock(nn.Module):
    """Fuse local state-space and global attention representations.

    Args:
        embed_dim: Input and output token dimension.
        num_heads: Number of Transformer attention heads.
        mlp_ratio: Transformer MLP expansion ratio.
        dropout: Dropout probability.
        drop_path: Stochastic-depth probability for fusion output.
        activation: Activation function name.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        mamba_backend: Literal["mamba_ssm", "lightweight"] = "mamba_ssm",
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        """Initialize Mamba and Transformer branches."""
        super().__init__()
        self.mamba = MambaBlock(
            embed_dim,
            backend=mamba_backend,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=dropout,
            drop_path=drop_path,
            activation=activation,
        )
        self.transformer = TransformerBlock(
            embed_dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=drop_path,
            activation=activation,
        )
        self.fusion_norm = nn.LayerNorm(embed_dim * 2)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Dropout(dropout)
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, tokens: Tensor) -> Tensor:
        """Fuse the Mamba and Transformer residual branch outputs."""
        mamba_tokens = self.mamba(tokens)
        transformer_tokens = self.transformer(tokens)
        features = self.fusion_norm(
            torch.cat((mamba_tokens, transformer_tokens), dim=-1)
        )
        gate = torch.sigmoid(self.gate(features))
        fused = gate * mamba_tokens + (1.0 - gate) * transformer_tokens
        return tokens + self.drop_path(self.projection(fused))
