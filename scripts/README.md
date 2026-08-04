# Scripts

Developer and operational entry points:

- `generate_qr_dataset.py`: create the paired QR dataset;
- `train_qr_restoration.py`: run the baseline/phase training workflow;
- `run_phase4_ablation.py`: run or resume controlled ablations;
- `finalize_ablation_study.py`: collect completed ablation outputs only;
- `run_final_benchmark.py`: evaluate the completed Phase 3 checkpoint and
  available compatible baselines;
- `prepare_final_analysis.py`: collect completed outputs and generate audited
  publication tables, figures, provenance, and discussion.

The collection scripts do not train models. `prepare_final_analysis.py` reads
only completed result artifacts; it must be run after those ignored artifacts
have been restored to their documented paths.

```powershell
python scripts/prepare_final_analysis.py
```

Use `--resume auto` with the ablation runner to resume from a compatible
`latest.pt`. Resume identity validation covers the variant, configuration and
dataset hashes, seed, backend, and architecture settings.
