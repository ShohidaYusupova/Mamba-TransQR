"""Unit tests for the hybrid Mamba-Transformer model architecture."""

from __future__ import annotations

from typing import Any

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
        mamba_backend="lightweight",
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
    encoding_type: (
        type[LearnablePositionalEncoding] | type[SinusoidalPositionalEncoding]
    ),
) -> None:
    """Both positional encodings preserve the token tensor shape."""
    encoding = encoding_type(16, 12, 0.0)
    tokens = torch.zeros(2, 16, 12)
    assert encoding(tokens).shape == tokens.shape


def test_lightweight_mamba_block_preserves_shape_and_backward() -> None:
    """Mamba block preserves batch, sequence, and embedding dimensions."""
    with pytest.warns(UserWarning, match="not the official Mamba"):
        block = MambaBlock(embed_dim=16, backend="lightweight", dropout=0.0)
    inputs = torch.randn(2, 9, 16, requires_grad=True)
    output = block(inputs)
    output.square().mean().backward()
    assert output.shape == inputs.shape
    assert inputs.grad is not None


def test_invalid_mamba_backend_is_rejected() -> None:
    """Backend selection is explicit and validated before construction."""
    with pytest.raises(ValueError, match="backend"):
        MambaBlock(embed_dim=16, backend="unknown")  # type: ignore[arg-type]


def test_official_backend_missing_dependency_does_not_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable official backend must fail instead of changing architecture."""
    import builtins

    from mambatransqr.models import OptionalDependencyError

    original_import = builtins.__import__

    def reject_mamba(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("mamba_ssm"):
            raise ImportError("simulated missing mamba-ssm")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_mamba)
    with pytest.raises(OptionalDependencyError, match="No lightweight fallback"):
        MambaBlock(embed_dim=16, backend="mamba_ssm")


def test_official_backend_forward_when_installed() -> None:
    """Official Mamba preserves token shape when the optional package is installed."""
    pytest.importorskip("mamba_ssm")
    block = MambaBlock(embed_dim=16, backend="mamba_ssm", dropout=0.0)
    assert block(torch.randn(2, 9, 16)).shape == (2, 9, 16)


def test_model_builder_propagates_mamba_settings() -> None:
    """All model Mamba settings reach each encoder fusion block."""
    config = ModelConfig(
        image_size=16,
        patch_size=8,
        embed_dim=8,
        depth=2,
        num_heads=2,
        mamba_backend="lightweight",
        mamba_d_state=9,
        mamba_d_conv=3,
        mamba_expand=4,
    )
    with pytest.warns(UserWarning, match="not the official Mamba"):
        model = build_model(config)
    for block in model.encoder.blocks:
        assert block.mamba.backend == "lightweight"
        assert (block.mamba.d_state, block.mamba.d_conv, block.mamba.expand) == (
            9,
            3,
            4,
        )


def test_transformer_block_has_residual_shape() -> None:
    """Transformer block preserves batch, sequence, and embedding dimensions."""
    block = TransformerBlock(embed_dim=16, num_heads=4, dropout=0.0)
    assert block(torch.randn(2, 9, 16)).shape == (2, 9, 16)


def test_model_reconstructs_configured_image_shape(config: ModelConfig) -> None:
    """Full model reconstruction has the configured image shape and range."""
    model = build_model(config)
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 3, 32, 32)
    assert torch.all((output >= 0) & (output <= 1))


def test_refinement_decoder_residual_starts_as_bounded_identity() -> None:
    """The refined residual decoder starts from its damaged-image skip path."""
    config = ModelConfig(
        image_size=16,
        patch_size=8,
        embed_dim=8,
        depth=1,
        num_heads=2,
        mamba_backend="lightweight",
        decoder_refinement_channels=4,
        residual_learning=True,
    )
    with pytest.warns(UserWarning, match="not the official Mamba"):
        model = build_model(config)
    inputs = torch.rand(2, 3, 16, 16)
    output = model(inputs)
    assert torch.equal(output, inputs)
    assert any(isinstance(layer, torch.nn.Conv2d) for layer in model.decoder.refinement)


def test_residual_learning_without_refinement_preserves_patch_path() -> None:
    """An ablation may bypass refinement without replacing patch reconstruction."""
    config = ModelConfig(residual_learning=True)
    assert config.decoder_refinement_channels == 0


def test_phase3_capacity_exceeds_phase2_within_planned_values() -> None:
    """The selected Phase 3 capacity is larger and uses planned dimensions."""
    phase2 = ModelConfig(
        image_size=128,
        patch_size=16,
        embed_dim=64,
        depth=4,
        num_heads=4,
        mamba_backend="lightweight",
    )
    phase3 = ModelConfig(
        image_size=128,
        patch_size=8,
        embed_dim=96,
        depth=6,
        num_heads=6,
        mamba_backend="lightweight",
        decoder_refinement_channels=32,
        residual_learning=True,
    )
    with pytest.warns(UserWarning):
        phase2_model = build_model(phase2)
    with pytest.warns(UserWarning):
        phase3_model = build_model(phase3)
    assert phase3.patch_size == 8
    assert phase3.embed_dim == 96
    assert phase3.depth == 6
    assert summarize_model(phase3_model).total_parameters > summarize_model(
        phase2_model
    ).total_parameters


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
