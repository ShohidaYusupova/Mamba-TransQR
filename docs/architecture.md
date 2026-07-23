# Architecture

Mamba-TransQR restores QR images with a patch-token encoder and learned patch
decoder.

```text
Image → PatchEmbedding → PositionalEncoding → HybridFusionBlock × depth → Decoder → Restored image
                                             ├─ MambaBlock: local convolution + selective state
                                             └─ TransformerBlock: global multi-head attention
```

Each hybrid block uses LayerNorm, residual paths, configurable activations,
dropout, and stochastic depth. `ModelConfig` controls image/patch sizes,
embedding width, depth, heads, MLP ratio, and regularization.
