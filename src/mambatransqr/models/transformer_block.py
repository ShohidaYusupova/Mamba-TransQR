"""Transformer encoder block for global token interactions."""

from __future__ import annotations

from torch import Tensor, nn

from mambatransqr.models.layers import DropPath, get_activation


class TransformerBlock(nn.Module):
    """Pre-normalized Transformer block with residual connections.

    Args:
        embed_dim: Input and output token dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Feed-forward hidden-dimension expansion ratio.
        dropout: Attention and projection dropout probability.
        drop_path: Stochastic-depth probability for each residual branch.
        activation: Activation function name.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        """Initialize attention and feed-forward branches."""
        super().__init__()
        if embed_dim < 1 or num_heads < 1 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        hidden_dim = max(1, round(embed_dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            get_activation(activation)(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, tokens: Tensor) -> Tensor:
        """Apply attention and MLP residual branches."""
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + self.drop_path(attended)
        return tokens + self.drop_path(self.mlp(self.norm2(tokens)))
