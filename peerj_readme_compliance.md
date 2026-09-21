# PeerJ README compliance report for v1.0.1

## Compliance matrix

| PeerJ requirement | README section | Status | Evidence |
|---|---|---|---|
| Title | Project heading | PASS | Names Mamba-TransQR. |
| Description | 1. Description | PASS | States the damaged-QR objective, Mamba-Transformer approach, contents, backend, and reproducibility purpose. |
| Dataset Information | 3; 7.1; 8 | PASS | Documents 10,000 pairs, grouped 7,000/1,500/1,500 split, size, seed, degradation, manifests, generator command, archive boundary, and byte-identity limitation. |
| Code Information | 2; 4 | PASS | Maps canonical model, data/degradation, losses, training, inference, evaluation, experiment, configuration, test, result, and documentation paths. |
| Usage Instructions | 7; 8 | PASS | Gives supported dataset, training, evaluation, inference, ablation-report, test, and reviewer commands with inputs/outputs and limitations. |
| Requirements | 5; 6 | PASS | Records Python/PyTorch/OS evidence, lock file, optional dependencies, CPU/GPU limitations, and installation workflow. |
| Methodology | 9 | PASS | Covers generation, grouped split, degradation, preprocessing, model, objectives, training, checkpoint selection, restoration/evaluation, latency, and ablations. |
| Citations/References | 12 | PASS | Links CITATION.cff, preserves the confirmed four-person creator list, cites the unchanged v1.0.0 scientific archive, and does not invent a v1.0.1 or article DOI. |
| License & Contribution Guidelines | 13; 14 | PASS | States MIT for source and associated documentation, identifies dataset/checkpoint licenses as not separately documented, distinguishes historical Zenodo record metadata, and links LICENSE and CONTRIBUTING.md. |

## Validation

- Release README SHA-256 (Git blob/archive bytes):
  `42713160089c7c634ac39d078e9db813cfa69adb8f1d549967ea71fd56be5b80`.
- Previous compliant working-tree SHA-256:
  `e965d22d9803d10294ce16932eba4f304ab447dc505777e0ac0def858820bdfb`.
- Hash change reason: corrected two Gratifikus repository URLs, added v1.0.1
  documentation-only semantics, aligned commands to the frozen v1.0.0 code,
  disclosed the historical ablation-orchestration limitation, and clarified
  v1.0.0 scientific-archive citation.
- Internal Markdown targets: 20 checked, 0 missing.
- Automated tests: 89 passed, 0 failed, 1 skipped. The skip is the optional
  official mamba_ssm backend, which did not produce the reported results.
- Ruff: PASS across src, tests, and scripts.
- CLI parser checks: dataset generation, training, Phase 4 report runner, and
  generic predict help all exited successfully.
- Phase 3 model construction: PASS; 1,476,259 trainable parameters.
- The final benchmark script has no help-only mode; a dry invocation stopped
  before evaluation because the isolated source worktree intentionally does not
  contain the external dataset. No scientific output was written.

## Release scope

This v1.0.1 candidate changes documentation and version/release metadata only.
No dataset, checkpoint, model architecture, experimental result, figure, table,
M3D, mobile, or physical-capture file is included.

## Status

PEERJ README COMPLIANCE: PASS
