# Mamba-TransQR v1.0.1 release notes

Mamba-TransQR v1.0.1 is a documentation-only reproducibility update for the
PeerJ Computer Science AI Application submission. The trained model, dataset
lineage, checkpoints, quantitative results, figures, and experimental
conclusions are unchanged from v1.0.0.

## Changed

- Expanded README to satisfy PeerJ AI Application reproducibility requirements.
- Documented the synthetic dataset, grouped split, manifests, and archival
  boundary.
- Documented installation and the validated dependency environment.
- Documented exact usage and reproduction commands supported by the frozen
  source.
- Documented methodology, outputs, limitations, and reviewer workflow.
- Documented citation, license, and contribution information.
- Aligned citation and package metadata with the confirmed four-person creator
  list already established by Zenodo v1.0.0.
- Clarified MIT coverage for source and associated documentation, while
  retaining dataset and checkpoint/model-weight licenses as not separately
  documented.
- Corrected the authoritative GitHub repository URL to
  <https://github.com/ShohidaYusupova/Mamba-TransQR>.
- Clarified that historical generator dependency versions prevent a claim of
  byte-identical dataset regeneration.
- Clarified the limitation of the historical ablation orchestration rather
  than claiming a nonexistent clean six-run command.
- Updated package and citation version metadata to 1.0.1.

## Unchanged

- Model architecture and implementation.
- Trained Phase 3 checkpoint and all checkpoint bytes.
- Training dataset lineage and authoritative dataset manifest.
- Reported PSNR and SSIM.
- Latency results and protocol.
- Controlled ablation results.
- Figures, tables, and manuscript-supporting quantitative data.
- Scientific conclusions.

## Scientific provenance

The scientific release remains v1.0.0 at commit
`781bbbd39fbea517f6481e78f837a5e6989d3232`, archived as
[Zenodo DOI 10.5281/zenodo.21913175](https://doi.org/10.5281/zenodo.21913175).
Version 1.0.1 points back to those unchanged assets and does not replace,
retag, or reinterpret them.
