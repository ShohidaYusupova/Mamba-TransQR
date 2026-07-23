# Repository Audit

Audit date: 2026-07-23  
Scope: repository source, packaging metadata, tests, public entry points, and
the available local runtime. This report is observational; no application code
was changed.

## Executive summary

The repository is a substantial scaffold with implemented-looking data, model,
training, evaluation, inference, loss, metric, visualization, and experiment
modules. It builds as a wheel and its Python source compiles. It is **not yet
verified as an executable end-to-end QR restoration project**: the available
runtime lacks PyTorch and all quality/test tools, and static review identifies
integration and production-readiness gaps.

Estimated completeness: **55%** for a public, reproducible QR-restoration
research/release repository; roughly **75%** for API surface scaffolding.

## Resolution update (2026-07-23)

The highest-priority data/training integration finding has been resolved:

- `qr_collate` now stacks homogeneous tensor fields while preserving metadata
  lists, making paired `image` and `target` batches compatible with
  `TrainingEngine`.
- `tests/test_end_to_end_smoke.py` adds a CPU-only workflow covering synthetic
  paired data creation, loaders, model forward/backward/update, checkpoint
  save/restore, predictor inference, and evaluator metrics.
- Optional dependencies are grouped in `pyproject.toml` and missing ONNX,
  torchvision, and psutil capabilities now raise clear errors or warnings.

Remaining gaps include DDP/distributed support,
real QR fixtures and benchmark datasets, and execution of the full quality
suite in a Python environment with PyTorch and development dependencies.

The full CPU quality suite was subsequently executed in a dedicated local
environment: 47 tests passed, Ruff passed after formatting/import cleanup, and
MyPy completed with explicit configuration for PyTorch and optional-package
untyped boundaries. The initial `pip install -e ".[dev]"` resolver timed out
after installing the core test stack, but an editable package wheel had already
been built successfully; the remaining environment limitation is installation
latency/permissions, not package metadata.

## Evidence collected

| Check | Result | Evidence / limitation |
|---|---|---|
| Python compilation | Pass | `python -m compileall -q src tests` completed successfully. |
| Wheel build | Pass | `pip wheel . --no-deps` succeeded when directed to a workspace-local pip cache. |
| `pip install -e .` | Inconclusive / local failure | Editable wheel creation succeeded, but installation failed with `WinError 5` when the bundled runtime tried to write to `AppData\\Roaming\\Python`. This is a runtime-permission issue, not evidence of invalid package metadata. Verify in a normal virtual environment. |
| Imports | Not executable here | The runtime has no `torch`; importing model/training/inference/evaluation packages therefore cannot be exercised. |
| Pytest | Not executable here | `pytest` and PyTorch are absent. |
| Ruff / Black / MyPy | Not executable here | `ruff`, `black`, and `mypy` are absent. |
| TODO/FIXME scan | No source findings | Matches exist only in Git's sample hook under `.git/hooks`; no application-source TODO/FIXME was found. |
| Syntax-level unimplemented method | One intentional base stub | `losses/base.py:BaseLoss.forward` raises `NotImplementedError`; this is an abstract base-style extension point. |

## Import and packaging assessment

`pyproject.toml` uses a conventional Hatchling src-layout setup and exposes the
`mambatransqr` CLI. Required runtime dependencies include NumPy, Pillow,
PyTorch, and torchvision. The editable wheel and regular wheel were built
successfully.

Import testing cannot be completed in this runtime because `torch`,
`torchvision`, and the development tools are absent. Package initializers for
models, training, evaluation, and inference import PyTorch-backed modules, so
they correctly require the declared runtime dependencies. A clean Python 3.11
virtual environment should run:

```bash
python -m pip install -e ".[dev,tensorboard]"
python -c "import mambatransqr.data, mambatransqr.models, mambatransqr.training"
pytest
ruff check .
black --check .
mypy src tests
```

The CLI entry point is declared as `mambatransqr.inference.cli:main`. It cannot
be executed here without PyTorch. Static review finds the command names present:
`predict`, `predict-folder`, `predict-video`, and `export`.

## Test assessment

Seven test files exist and cover package, data, model, training, evaluation,
inference, and core-module paths. They are useful smoke/unit tests but do not
constitute comprehensive integration coverage. They were not runnable in the
available environment.

Missing test categories include:

- installed-package and CLI subprocess tests;
- end-to-end data-loader → trainer → checkpoint → predictor tests;
- real QR encode/damage/restore/decode fixtures;
- CPU and CUDA device matrix coverage;
- ONNX Runtime validation and dynamic-batch export tests;
- distributed or multi-GPU tests;
- malformed input, corrupted checkpoint, and report-schema failure tests.

## Placeholder, stub, and duplicate assessment

The `BaseLoss.forward` method is the only direct `NotImplementedError` in
application source and is an intentional base-class contract. There are no
application TODO/FIXME markers.

Several directories remain placeholders or minimally populated: `utils`,
`docs` before the added guides, `examples`, `scripts`, `notebooks`, and
`benchmarks`. Their README files describe intended future contents rather than
providing runnable artifacts.

There is duplicated functionality:

- `evaluation.metrics.MetricsManager` and `metrics.manager.MetricsManager`;
- image/binary metric formulas are duplicated across `evaluation` and `metrics`;
- evaluation QR decoding and inference QR decoding are coupled through optional
  decoder helpers rather than a stable shared interface;
- TensorBoard/logging concepts appear in both training and visualization.

This duplication creates drift risk and should be consolidated around one core
metrics package and one reporting/logger abstraction.

## Mamba model assessment

`MambaBlock` now has explicit `mamba_ssm` and `lightweight` backends. The
production default wraps the official `mamba-ssm` Mamba module and fails clearly
when that optional dependency is unavailable; it never silently changes the
architecture. The former custom recurrence is retained as
`LightweightStateSpaceBlock`, warns on use, and is not presented as official
Mamba. CPU smoke tests select it explicitly. Backend/version identity is saved
in checkpoints and report artifacts.

## Pipeline assessment

### Data and QR restoration

The data package discovers generic image files, supplies PIL/NumPy
augmentations, and contains a synthetic damage generator for scratches, missing
blocks, blur, illumination, noise, and JPEG artifacts. It does not include QR
generation, QR content/metadata labels, dataset download/validation,
reproducible split manifests, or verified target-pair management beyond relative
path matching.

Most importantly, the default `qr_collate` returns lists for all sample fields,
whereas `TrainingEngine._unpack` requires tensor-valued `image` and `target`
entries. Thus the default data loader and training engine are not directly
compatible. A tensor-stacking collate path or a training-specific collator is
required for executable end-to-end restoration training.

Conclusion: the QR restoration pipeline is **partial**, not complete.

### Training

The trainer includes recognizable components: optimizer/scheduler factories,
AMP, gradient accumulation/clipping, EMA, callbacks, CSV/TensorBoard logging,
checkpointing, resume, and a DataParallel branch. Static review identifies
limitations:

- no verified runnable path because PyTorch tests were unavailable;
- DataParallel is not distributed multi-GPU training (no DDP/process setup,
  distributed samplers, rank-aware checkpointing, or metric reduction);
- scheduler/checkpoint protocol typing and callback integrations need actual
  MyPy/PyTorch validation;
- best-checkpoint selection is absent when the monitored metric is not emitted;
- training is blocked by the default data-collation mismatch above.

Conclusion: components exist, but the training pipeline is **not proven
executable end-to-end**.

### Inference

`Predictor`, batch/folder/video helpers, reporting, and export helpers exist.
CPU/CUDA selection, AMP, checkpoint loading, TorchScript, and ONNX APIs are
present. Video, LPIPS, QR decoding, ONNX validation, OpenCV, ZBar, and ZXing
depend on optional packages that are not declared as dedicated extras.

Static issues include creating a `Predictor` even for the export command before
building the export model, and flat output naming in folder inference, which can
overwrite outputs from nested folders with equal basenames. No real checkpoint,
CLI, TorchScript, or ONNX invocation could be run here.

Conclusion: inference is **partially implemented and unverified**.

### Evaluation

Evaluator, benchmark, latency, reports, ROC, confusion matrix, decoder, and
visualizer modules exist. SSIM is a global approximation rather than common
windowed SSIM; LPIPS constructs a model on each call; CPU memory measurement is
best-effort. Decoder and visualization paths rely on optional packages. Tests
were not runnable.

Conclusion: evaluation is **partially implemented and unverified**.

## Modules still incomplete or needing production work

1. **Model performance** — add benchmark coverage for official Mamba SSM on
   supported accelerator environments.
2. **Data-to-training integration** — return stacked tensors from a training
   collator, add paired-data contracts, and test a full training step.
3. **Distributed training** — add DDP, distributed sampling, rank-safe logging,
   checkpointing, and metric aggregation if multi-GPU support is a release
   claim.
4. **Optional dependencies** — publish clear extras for TensorBoard,
   visualization, QR decoders, LPIPS, ONNX, ONNX Runtime, OpenCV, and psutil.
5. **Configuration system** — YAML files are examples; there is no loader,
   schema validation, override mechanism, or reproducibility snapshot wired to
   model/data/training construction.
6. **Examples/benchmarks** — provide executable training, inference, export,
   evaluation, and benchmark examples with small fixture data.
7. **Quality gate** — run and fix Ruff, Black, MyPy, and Pytest in CI; current
   local evidence is limited to syntax compilation and wheel construction.
8. **API consolidation** — eliminate duplicate metric managers and duplicate
   metric implementations; formalize public imports and deprecation policy.
9. **Release metadata** — replace placeholder GitHub URLs and contributor
   identity, establish supported-version/security contact policy, and publish
   release automation.

## Completeness estimate rationale

The repository has broad API coverage and documentation, which supports the
75% scaffolding estimate. The lower 55% production/research-release estimate
accounts for unexecuted tests, absent runtime dependencies, unverified package
installation, missing end-to-end QR fixtures, collator/trainer incompatibility,
non-official Mamba implementation, incomplete distributed support, and optional
dependency/configuration gaps.
