# Configuration

Store version-controlled, reproducible experiment configurations here.

Completed research recipes:

- `first_real_qr_restoration.yaml`: first real baseline;
- `phase1_pilot.yaml`: bounded output and loss rebalance;
- `phase2_pilot.yaml`: warmup/cosine schedule, EMA selection, and early stopping;
- `phase3_pilot.yaml`: Phase 3 architecture using the lightweight backend;
- `phase4_ablation.yaml`: controlled one-factor-at-a-time ablations and resume;
- `final_benchmark.yaml`: evaluation-only final benchmark recipe.

Production defaults remain separate from these experiment recipes. In
particular, a recipe declaring `mamba_backend: lightweight` uses
`LightweightStateSpaceBlock`, not the official `mamba-ssm` selective SSM.
