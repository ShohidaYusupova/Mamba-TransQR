"""Official and lightweight Mamba-style sequence backends."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Literal
import warnings

import torch
from torch import Tensor, nn

from mambatransqr.models.layers import DropPath, get_activation

MambaBackend = Literal["mamba_ssm", "lightweight"]


class OptionalDependencyError(ImportError):
    """Raised when an explicitly requested optional runtime dependency is absent."""


def mamba_ssm_version() -> str | None:
    """Return the installed official package version, when available."""
    try:
        return version("mamba-ssm")
    except PackageNotFoundError:
        return None


class LightweightStateSpaceBlock(nn.Module):
    """Legacy lightweight recurrent state-space approximation.

    This is a self-contained, Mamba-inspired block.  It is **not** an
    implementation of the official Mamba selective SSM.
    """

    def __init__(
        self,
        embed_dim: int,
        expansion: float = 2.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if embed_dim < 1 or expansion <= 0:
            raise ValueError("embed_dim and expansion must be positive")
        hidden_dim = max(1, round(embed_dim * expansion))
        self.norm = nn.LayerNorm(embed_dim)
        self.input_projection = nn.Linear(embed_dim, hidden_dim * 2)
        self.local_mixing = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        self.state_gate = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, embed_dim)
        self.activation = get_activation(activation)()
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, tokens: Tensor) -> Tensor:
        """Apply the legacy recurrence while preserving token shape."""
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


class MambaBlock(nn.Module):
    """Residual sequence block with an explicit official or lightweight backend.

    ``backend='mamba_ssm'`` wraps the official :mod:`mamba_ssm` ``Mamba``
    module. It never falls back to another implementation.  The ``lightweight``
    backend exists for compatibility and CPU-only smoke tests only.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        backend: MambaBackend = "mamba_ssm",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        expansion: float | None = None,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if backend not in {"mamba_ssm", "lightweight"}:
            raise ValueError("backend must be 'mamba_ssm' or 'lightweight'")
        if embed_dim < 1 or d_state < 1 or d_conv < 1 or expand < 1:
            raise ValueError("embed_dim, d_state, d_conv, and expand must be positive")
        self.backend: MambaBackend = backend
        self.d_state, self.d_conv, self.expand = d_state, d_conv, expand
        if backend == "lightweight":
            warnings.warn(
                "backend='lightweight' uses LightweightStateSpaceBlock, not the "
                "official Mamba selective SSM.",
                UserWarning,
                stacklevel=2,
            )
            self.implementation: nn.Module = LightweightStateSpaceBlock(
                embed_dim,
                expansion=expansion if expansion is not None else float(expand),
                dropout=dropout,
                drop_path=drop_path,
                activation=activation,
            )
            return
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
        except ImportError as error:
            raise OptionalDependencyError(
                "backend='mamba_ssm' requires the optional mamba-ssm package. "
                "Install it with `pip install -e \".[mamba]\"` on a supported "
                "Linux/NVIDIA CUDA environment. No lightweight fallback is used."
            ) from error
        self.norm = nn.LayerNorm(embed_dim)
        self.implementation = Mamba(
            d_model=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, tokens: Tensor) -> Tensor:
        """Mix ``(batch, sequence, embed_dim)`` tokens with a residual path."""
        if self.backend == "lightweight":
            return self.implementation(tokens)
        output = self.dropout(self.implementation(self.norm(tokens)))
        return tokens + self.drop_path(output)
