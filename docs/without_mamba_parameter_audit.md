# Without-Mamba parameter-count audit

The parameter count reported for the controlled 30-epoch `without_mamba`
ablation is correct.

| Configuration | Trainable parameters |
| --- | ---: |
| Full Phase 3 | 1,476,259 |
| Without Mamba | 1,476,259 |
| Difference | 0 |

Both models were reconstructed through `build_model(ModelConfig(**recipe["model"]))`,
the model-construction path used by `scripts/run_phase4_ablation.py`. The recipes
were produced with that script's recursive merge of `configs/phase3_pilot.yaml`
and the corresponding entry in `configs/phase4_ablation.yaml`. Counts were then
computed directly as:

```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

The full recipe leaves `ModelConfig.use_mamba` at its default value of `true`.
The `without_mamba` override sets it to `false`; the saved resolved recipe at
`results/ablation/without_mamba/resolved_config.yaml` records the same setting.
This flag does not remove or freeze any parameters. `MambaTransQR` passes the
flag through `Encoder`, and every `HybridFusionBlock` still constructs its
`MambaBlock`, Transformer branch, fusion layers, and projection. In
`HybridFusionBlock.forward`, `use_mamba == false` returns through the
Transformer/projection path before executing the Mamba branch. Consequently,
all six Mamba modules remain registered and trainable but are bypassed during
the forward pass. They contain 562,752 of the parameters in each reconstructed
model.

The independently saved 30-epoch artifacts agree with the reconstruction:
`results/phase3_pilot/benchmark_summary.csv`,
`results/ablation/without_mamba/benchmark_summary.csv`, and
`results/ablation/ablation_summary.json` each report 1,476,259 parameters for
the applicable model. The full and without-Mamba checkpoints also have the
same 183 model-state entries, consistent with an execution bypass rather than
a physically smaller architecture.

Relevant implementation and configuration evidence:

- `configs/phase3_pilot.yaml`: final Phase 3 architecture and 30-epoch policy.
- `configs/phase4_ablation.yaml`: base recipe, completed 30-epoch variants, and
  the `model.use_mamba: false` override.
- `scripts/run_phase4_ablation.py`: recursive recipe merge, model construction,
  evaluation, and parameter reporting path.
- `src/mambatransqr/models/transqr.py` and
  `src/mambatransqr/models/encoder.py`: configuration propagation and block
  construction.
- `src/mambatransqr/models/fusion_block.py`: unconditional Mamba-module
  construction and conditional forward-pass bypass.

Conclusion: Table 2 does not require correction. Its 1,476,259 value describes
registered trainable parameters, not only parameters executed by a particular
forward path. The without-Mamba ablation disables Mamba computation but does
not physically remove or freeze the branch.
