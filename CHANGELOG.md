# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.2] - 2026-09-21

### Changed

- Finalized post-publication README wording for the PeerJ Computer Science AI
  Application resubmission.
- Synchronized package, citation, and reproducibility version metadata.
- Documented the published v1.0.1 Zenodo archival record, DOI
  `10.5281/zenodo.22874149`.
- Removed obsolete pre-publication wording about adding the v1.0.1 DOI later.
- Stabilized the README so it does not depend on insertion of a future v1.0.2
  Zenodo DOI.

### Unchanged

- Scientific experiments, model implementation, dataset lineage, checkpoints,
  quantitative results, figures, tables, and scientific conclusions.

## [Unreleased]

### Added

- Data, model, training, evaluation, inference, loss, metric, visualization,
  and experiment-management frameworks.
- Official `mamba-ssm` Mamba backend with configurable state, convolution, and
  expansion settings, plus backend identity in checkpoints and reports.
- Module-aware QR degradation engine for dropout, occlusion, finder-pattern,
  ink-spread, and print/scan corruption with reproducible provenance.
- Synthetic QR paired-dataset generation with payload-safe data splits,
  metadata, integrity validation, and dataset manifests.
- QR-specific multi-scale restoration objective and reproducibility recipe.
- Reproducible benchmark registry, measured-result reporting, tables, and figures.
- CPU-only full-pipeline smoke validation.

### Changed

- Production documentation, package exports, CLI configuration, and release
  quality checks.
- Tensor batch collation now integrates with the training engine; a CPU
  end-to-end restoration smoke test covers training, checkpointing, inference,
  and evaluation.
- The former Mamba-inspired recurrence is now clearly named
  `LightweightStateSpaceBlock` and is only selected explicitly for compatibility
  and CPU smoke tests.

## [0.1.0] - 2026-07-23

### Added

- Initial production-quality project scaffold.
