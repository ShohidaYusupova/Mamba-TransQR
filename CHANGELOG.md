# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Data, model, training, evaluation, inference, loss, metric, visualization,
  and experiment-management frameworks.
- Official `mamba-ssm` Mamba backend with configurable state, convolution, and
  expansion settings, plus backend identity in checkpoints and reports.

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
