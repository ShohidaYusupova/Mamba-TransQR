# Experiment reproducibility

This guide describes how to verify and reconstruct the completed research
package without confusing generated artifacts with version-controlled source.

## Scope

The recorded sequence is:

1. first real QR restoration baseline;
2. Phase 1 output/loss optimization;
3. Phase 2 training-policy optimization;
4. Phase 3 architecture optimization;
5. six-configuration, 30-epoch ablation study;
6. final fixed-protocol benchmark;
7. publication-ready aggregation and audit.

Datasets, checkpoints, TensorBoard logs, predictions, and generated results are
ignored by Git. Reproducing a report therefore requires both the source commit
and the corresponding external/generated artifacts at their original paths.

## Recorded source lineage

| Stage | Commit |
| --- | --- |
| First real baseline | `b5e9d56` |
| Phase 1 | `1fa4adb` |
| Phase 2 | `c96f2a7` |
| Phase 3 | `b8fee1f` |
| Phase 4 ablation runner | `86d1a12` |
| Automatic ablation resume | `ca3c7b0` |
| Finalized ablation analysis | `a5200ed` |
| Final benchmark | `f38381f` |
| Publication package | `a4b4e63` |
| Publication audit corrections | `2ba3805` |

Use the full commit object from `git rev-parse <short-hash>` when recording a
new reproduction manifest.

## Environment

Create a Python 3.11+ environment and install the relevant extras:

```bash
python -m pip install -e ".[dev,benchmark,qr-training,qr-generation]"
```

Install `.[qr-decode]` only when payload recovery metrics are required. ZBar may
also require a platform library. If no decoder backend is installed, payload
decode rate is unavailable and must remain `N/A`.

The completed Phase 3, ablation, and final-analysis results used
`mamba_backend: lightweight`. This selects `LightweightStateSpaceBlock`, a
compatibility implementation for CPU-capable experiments. It is not the
official Mamba-SSM implementation. Production defaults and experiments using
`mamba_backend: mamba_ssm` require the separate `.[mamba]` extra and must be
reported independently.

## Artifact identity

Before using a saved result, verify:

- Git commit and resolved configuration;
- dataset manifest SHA-256 and fixed split;
- checkpoint SHA-256;
- random seed;
- backend identity and architecture settings;
- epoch budget and checkpoint-selection policy;
- latency batch size, warmups, iterations, device, and runtime context.

Phase 4 checkpoints additionally persist optimizer, scheduler, EMA when
enabled, GradScaler when present, early-stopping state, epoch/best metric, and
random-number-generator states. Automatic resume rejects incompatible variant,
configuration, dataset, seed, backend, or architecture identity.

## Canonical paths

```text
datasets/first_real_qr_10k/
checkpoints/phase3_pilot/best.pt
results/first_real_qr_restoration/
results/phase1_pilot/
results/phase2_pilot/
results/phase3_pilot/
results/ablation/final/
results/final_benchmark/
results/final_analysis/paper_ready/
```

The publication audit records cell/figure provenance in
`results/final_analysis/paper_ready/source_provenance.csv`. Its
`audit_report.md` documents verification status and `corrections_log.md`
records presentation-only corrections. CSV sources preserve measured
precision; publication tables round metrics to six decimals and latency to
three decimals.

## Rebuilding analysis without experiments

With all completed source artifacts present, rebuild only derived tables,
figures, and prose:

```bash
python scripts/finalize_ablation_study.py
python scripts/prepare_final_analysis.py
```

These commands collect saved outputs; they do not retrain models. Do not run
training or benchmark scripts when the goal is only to reproduce the paper
package.

## Interpretation constraints

- Training results use one random seed; confidence intervals and statistical
  significance are unavailable.
- Phase 1 used 10 epochs, whereas the baseline, Phase 2, Phase 3, and completed
  ablations used 30; loss values are also affected by objective changes.
- The ablation design is one-factor-at-a-time and cannot isolate interactions.
- The Mamba bypass retains inert parameters and represents branch-compute
  removal, not a parameter-matched redesign.
- The saved "Decode Rate" in the final benchmark is a threshold proxy, not
  payload recovery.
- SRCNN, ESPCN, VDSR, EDSR, and SwinIR lacked compatible local artifacts and
  remain `N/A`; no state-of-the-art ranking is supported for those rows.
- Latencies are batch-one CPU measurements. The saved artifacts do not identify
  the CPU model, core allocation, or power state, so results should not be
  generalized across hardware.

## Verification

Run the available static and test checks after restoring the environment:

```bash
ruff check src tests scripts
pytest -q
```

The LaTeX publication tables use `booktabs` commands. Compile them inside a
document that loads `\\usepackage{booktabs}`; when no TeX engine is installed,
the analysis audit performs balanced-environment, command, row-terminator, and
column-count validation instead.
