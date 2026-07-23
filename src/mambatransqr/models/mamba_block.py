"""Mamba-inspired selective state-space sequence block."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mambatransqr.models.layers import DropPath, get_activation


class MambaBlock(nn.Module):
    """Residual selective state-space block with local depthwise mixing.

    This implementation is self-contained and does not require the optional
    ``mamba-ssm`` package. It combines input-dependent gates, a depthwise local
    convolution, and a stable recurrent state update across patch tokens.

    Args:
        embed_dim: Input and output token dimension.
        expansion: Hidden-state expansion ratio.
        dropout: Dropout probability.
        drop_path: Stochastic-depth probability for the residual branch.
        activation: Activation function name.
    """

    def __init__(
        self,
        embed_dim: int,
        expansion: float = 2.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        activation: str = "silu",
    ) -> None:
        """Initialize the state-space block."""
        super().__init__()
        if embed_dim < 1 or expansion <= 0:
            raise ValueError("embed_dim and expansion must be positive")
        hidden_dim = max(1, round(embed_dim * expansion))
        self.norm = nn.LayerNorm(embed_dim)
        self.input_projection = nn.Linear(embed_dim, hidden_dim * 2)
        self.local_mixing = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.state_gate = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, embed_dim)
        self.activation = get_activation(activation)()
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, tokens: Tensor) -> Tensor:
        """Apply selective sequence mixing with a residual connection.

        Args:
            tokens: Tensor shaped ``(batch, sequence, embed_dim)``.

        Returns:
            Mixed tokens with the input residual added.
        """
        normalized = self.norm(tokens)
        content, gate = self.input_projection(normalized).chunk(2, dim=-1)
        content = self.local_mixing(content.transpose(1, 2)).transpose(1, 2)
        content = self.activation(content)
        state = torch.zeros_like(content[:, 0])
        outputs: list[Tensor] = []
        for step in content.unbind(dim=1):
            decay = torch.sigmoid(self.state_gate(step))
            state = decay * state + (1.0 - decay) * step
            outputs.append(state)
        mixed = torch.stack(outputs, dim=1) * torch.sigmoid(gate)
        output = self.dropout(self.output_projection(mixed))
        return tokens + self.drop_path(output)
