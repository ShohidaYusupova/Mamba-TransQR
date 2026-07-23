"""Unit tests for the hybrid Mamba-Transformer model architecture."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mambatransqr.models import (  # noqa: E402
    DropPath,
    LearnablePositionalEncoding,
    MambaBlock,
    ModelConfig,
    PatchEmbedding,
    SinusoidalPositionalEncoding,
    TransformerBlock,
    build_model,
    summarize_model,
)


@pytest.fixture
def config() -> ModelConfig:
    """Return a compact configuration for fast model tests."""
    return ModelConfig(
        image_size=32,
        patch_size=8,
        embed_dim=16,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path_rate=0.1,
    )


def test_patch_embedding_returns_expected_token_shape() -> None:
    """Patch embedding maps a square image to the expected token sequence."""
    embedding = PatchEmbedding(image_size=32, patch_size=8, embed_dim=16)
    tokens = embedding(torch.randn(2, 3, 32, 32))
    assert embedding.num_patches == 16
    assert tokens.shape == (2, 16, 16)


@pytest.mark.parametrize(
    "encoding_type",
    [LearnablePositionalEncoding, SinusoidalPositionalEncoding],
)
def test_positional_encoding_preserves_token_shape(
    encoding_type: type[LearnablePositionalEncoding]
    | type[SinusoidalPositionalEncoding],
) -> None:
    """Both positional encodings preserve the token tensor shape."""
    encoding = encoding_type(16, 12, 0.0)
    tokens = torch.zeros(2, 16, 12)
    assert encoding(tokens).shape == tokens.shape


def test_mamba_block_has_residual_shape() -> None:
    """Mamba block preserves batch, sequence, and embedding dimensions."""
    block = MambaBlock(embed_dim=16, expansion=2.0, dropout=0.0)
    assert block(torch.randn(2, 9, 16)).shape == (2, 9, 16)


def test_transformer_block_has_residual_shape() -> None:
    """Transformer block preserves batch, sequence, and embedding dimensions."""
    block = TransformerBlock(embed_dim=16, num_heads=4, dropout=0.0)
    assert block(torch.randn(2, 9, 16)).shape == (2, 9, 16)


def test_model_reconstructs_configured_image_shape(config: ModelConfig) -> None:
    """Full model reconstruction matches its configured image dimensions."""
    model = build_model(config)
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 3, 32, 32)


def test_model_summary_counts_trainable_parameters(config: ModelConfig) -> None:
    """Model summary reports nonzero and internally consistent parameter counts."""
    summary = summarize_model(build_model(config))
    assert summary.total_parameters > 0
    assert summary.trainable_parameters == summary.total_parameters


def test_drop_path_is_identity_during_evaluation() -> None:
    """Stochastic depth is disabled in evaluation mode."""
    layer = DropPath(0.5).eval()
    inputs = torch.ones(2, 3, 4)
    assert torch.equal(layer(inputs), inputs)


def test_invalid_patch_configuration_is_rejected() -> None:
    """Patch embedding rejects images not divisible by their patch size."""
    with pytest.raises(ValueError, match="divisible"):
        PatchEmbedding(image_size=30, patch_size=8, embed_dim=16)
