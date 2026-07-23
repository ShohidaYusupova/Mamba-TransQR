# Architecture

Mamba-TransQR restores QR images with a patch-token encoder and learned patch
decoder.

```text
Image → PatchEmbedding → PositionalEncoding → HybridFusionBlock × depth → Decoder → Restored image
                                             ├─ MambaBlock: official mamba-ssm selective SSM
                                             └─ TransformerBlock: global multi-head attention
```

`MambaBlock` uses `mamba_backend="mamba_ssm"` by default. This wraps the
official `mamba_ssm.modules.mamba_simple.Mamba` module with pre-normalization,
dropout, DropPath, and a residual connection, mapping `embed_dim` to `d_model`.
`mamba_d_state`, `mamba_d_conv`, and `mamba_expand` configure the official
layer.

`mamba_backend="lightweight"` uses `LightweightStateSpaceBlock`, the legacy
self-contained recurrence. It is not the official Mamba selective SSM and
emits a warning; use it only where an explicit CPU-compatible smoke backend is
needed. Every checkpoint and report records `mamba_backend`,
`mamba_implementation`, and `mamba_ssm_version` for traceability.

Synthetic QR training data is generated from an explicit QR module grid. The
dataset builder applies module-aware degradations at mild, moderate, or severe
levels and preserves paired clean targets plus per-sample provenance.

The QR training objective weights finder, alignment, timing, quiet-zone, and
module-boundary regions. Its decode-consistency term is a differentiable
binarization/contrast surrogate and never claims differentiation through ZBar
or ZXing.
