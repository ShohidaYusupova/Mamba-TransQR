"""Positional encodings for patch-token sequences."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embeddings.

    Args:
        num_positions: Maximum supported token count.
        embed_dim: Token embedding dimension.
        dropout: Dropout probability applied after position addition.
    """

    def __init__(self, num_positions: int, embed_dim: int, dropout: float = 0.0) -> None:
        """Initialize learnable position parameters."""
        super().__init__()
        if num_positions < 1 or embed_dim < 1:
            raise ValueError("num_positions and embed_dim must be positive")
        self.position = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        """Add positions to a batch of tokens."""
        if tokens.shape[1] > self.position.shape[1]:
            raise ValueError("token count exceeds configured positional-encoding length")
        return self.dropout(tokens + self.position[:, : tokens.shape[1]])


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional embeddings.

    Args:
        num_positions: Maximum supported token count.
        embed_dim: Token embedding dimension.
        dropout: Dropout probability applied after position addition.
    """

    def __init__(self, num_positions: int, embed_dim: int, dropout: float = 0.0) -> None:
        """Create the fixed positional-encoding buffer."""
        super().__init__()
        if num_positions < 1 or embed_dim < 1:
            raise ValueError("num_positions and embed_dim must be positive")
        position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / embed_dim)
        )
        encoding = torch.zeros(num_positions, embed_dim)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        """Add fixed positions to a batch of tokens."""
        if tokens.shape[1] > self.encoding.shape[1]:
            raise ValueError("token count exceeds configured positional-encoding length")
        return self.dropout(tokens + self.encoding[:, : tokens.shape[1]].to(tokens.dtype))
