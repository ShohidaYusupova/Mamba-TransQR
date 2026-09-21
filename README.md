# Mamba-TransQR

Mamba-TransQR is a lightweight QR-image restoration system that combines a
Transformer encoder, a lightweight state-space sequence mixer, and a refinement
decoder. This repository contains the source code, configurations, compact
results, and provenance documentation supporting the PeerJ Computer Science AI
Application submission.

This README accompanies **v1.0.2**, the final documentation-only
reproducibility synchronization patch prepared for the PeerJ Computer Science
AI Application resubmission. The corresponding public source release is
[GitHub v1.0.2](https://github.com/ShohidaYusupova/Mamba-TransQR/releases/tag/v1.0.2).

The trained model, dataset lineage, checkpoints, quantitative results, figures,
and experimental conclusions are unchanged from the frozen scientific release
**v1.0.0** at commit
`781bbbd39fbea517f6481e78f837a5e6989d3232`. The unchanged scientific assets are
[Zenodo DOI 10.5281/zenodo.21913175](https://doi.org/10.5281/zenodo.21913175).
The preceding documentation-only reproducibility release, **v1.0.1**, is
archived at [Zenodo DOI 10.5281/zenodo.22874149](https://zenodo.org/records/22874149).
Neither v1.0.1 nor v1.0.2 reran the experiments or changed their results. See
[the v1.0.2 release notes](RELEASE_NOTES_v1.0.2.md) for the exact scope.

## 1. Description

The project restores synthetically degraded QR images while preserving the
structural features needed for decoding. Its Mamba-Transformer approach couples
a Transformer representation with the validated lightweight state-space mixer
and a refinement decoder. The reported workflow generates paired
clean/degraded QR images, trains restoration models in staged phases, evaluates
the selected Phase 3 checkpoint on a fixed test split, and runs controlled
ablations and latency measurements.

The manuscript experiments used the repository's
`lightweight_state_space` backend. The optional official `mamba-ssm` package is
an alternative implementation and was **not** used to produce the reported
results.

## 2. Repository Structure

```text
Mamba-TransQR/
├── src/mambatransqr/       Python package and canonical model implementation
├── configs/                Dataset, training, benchmark, and ablation configs
├── scripts/                Dataset, training, evaluation, and audit entry points
├── tests/                  Automated unit and integration tests
├── docs/                   Reproducibility and archival provenance documents
├── results/                Compact manuscript-supporting results and audits
├── release/                Archival bundle manifests and checksums
├── datasets/               Local generated datasets (large content is ignored)
├── checkpoints/            Local trained weights (large content is ignored)
├── pyproject.toml           Package metadata and dependency groups
└── requirements-peerj-lock.txt  Validated PeerJ CPU environment lock
```

Physical smartphone/deployment data developed after v1.0.0 are not included in
this publication release and are not evidence for its reported training or test
results.

## 3. Dataset Information

The manuscript dataset contains **10,000 paired samples**: 7,000 training,
1,500 validation, and 1,500 test pairs. Each pair consists of a clean 128×128 QR
image and one synthetically degraded image. Generation uses seed 42 and samples
QR versions 1, 5, 10, and 20; error-correction levels L, M, Q, and H; and
alphanumeric, numeric, byte, and URL payloads. The degradation engine applies
the configured moderate-severity blur, noise, compression, illumination,
perspective, occlusion, and related transformations.

Payloads are assigned to exactly one split before image generation. The tracked
the grouped split implementation assigns each generated payload to exactly one
split before degradation. The split and manifest provenance are documented in
the archival manifest.
The authoritative manifest SHA-256 is
`2c47950acb452137b3634bb3527cd48f3554b18f3416cd9c8cc7a1422df45c2f`.

The complete image dataset is intentionally excluded from Git history. It must
be obtained from the v1.0.0 archival assets or generated locally. Historical
versions of `qrcode` and PyYAML used for the original generation run were not
recorded, so current regeneration is structurally deterministic but is not
claimed to be byte-identical to the archived dataset. Reviewers should verify
the downloaded archive against [the archival manifest](docs/peerj_archival_manifest.md)
and its supplied checksums.

## 4. Code Information

The canonical model is `MambaTransQR` in
[`src/mambatransqr/models/transqr.py`](src/mambatransqr/models/transqr.py), with
architecture parameters represented by `ModelConfig`. Supporting package code
is organized as follows:

- `src/mambatransqr/data/`: dataset readers, paired transforms, and degradation;
- `src/mambatransqr/models/`: Transformer/state-space model components;
- `src/mambatransqr/losses/`: reconstruction and QR-aware objectives;
- `src/mambatransqr/training/`: training loop, scheduling, and checkpoints;
- `src/mambatransqr/evaluation/`: image-quality metrics and evaluation helpers;
- `src/mambatransqr/inference/`: image, folder, and video inference CLI;
- `src/mambatransqr/benchmarks/`: benchmark utilities;
- `src/mambatransqr/experiments/`: experiment orchestration and provenance.

Dataset generation is implemented by
[`scripts/generate_qr_dataset.py`](scripts/generate_qr_dataset.py) and
[`src/mambatransqr/data/dataset_builder.py`](src/mambatransqr/data/dataset_builder.py).
Training uses [`scripts/train_qr_restoration.py`](scripts/train_qr_restoration.py),
and final evaluation uses
[`scripts/run_final_benchmark.py`](scripts/run_final_benchmark.py).

## 5. Requirements

- Python 3.11 or newer (the validated archival environment used Python 3.12.13)
- PyTorch 2.13.0+cpu for the validated Windows CPU reproduction environment
- Windows 10 build 19045 for the recorded validation run
- Sufficient storage for the external dataset, checkpoints, and generated output

The deterministic package list is
[`requirements-peerj-lock.txt`](requirements-peerj-lock.txt). Core and optional
dependency groups are declared in [`pyproject.toml`](pyproject.toml). The
official `mamba-ssm` package is optional, is absent from the validated lock, and
must not be installed when reproducing the reported lightweight-backend results.

The exact CPU model, RAM capacity, core/thread count, power mode, and GPU model
were not recorded. CUDA was unavailable in the validated experiment metadata;
the reported run was CPU-only. See
[`docs/reproducibility.md`](docs/reproducibility.md) for the complete provenance
and limitations.

## 6. Installation

For the frozen archival version on Windows PowerShell:

```powershell
git clone https://github.com/ShohidaYusupova/Mamba-TransQR.git
cd Mamba-TransQR
git checkout v1.0.1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-peerj-lock.txt
python -m pip install --no-deps -e .
python -c "import mambatransqr; print(mambatransqr.__version__)"
mambatransqr --help
```

The `--no-deps` editable install preserves the exact versions installed from
the lock file. For development rather than archival reproduction, install the
relevant optional groups, for example:

```powershell
python -m pip install -e ".[dev,benchmark,qr-training,qr-generation]"
```

Do not install `.[mamba]` for the reported experiments.

## 7. Usage Instructions

Run all commands from the repository root after activating the environment.

### 7.1 Generate/Reproduce the Dataset

```powershell
python scripts/generate_qr_dataset.py --config configs/first_real_qr_10k_dataset.yaml --output datasets/first_real_qr_10k
```

Input: [`configs/first_real_qr_10k_dataset.yaml`](configs/first_real_qr_10k_dataset.yaml).
Output: `datasets/first_real_qr_10k/`, containing
`train/`, `validation/`, and `test/` directories with `clean/`,
`damaged/moderate/`, and `metadata/`, plus `dataset_manifest.csv`,
`dataset_summary.json`, and `split_statistics.json`.

Verify the authoritative manifest after restoring the archived dataset:

```powershell
(Get-FileHash -Algorithm SHA256 datasets/first_real_qr_10k/dataset_manifest.csv).Hash.ToLower()
```

The expected value is
`2c47950acb452137b3634bb3527cd48f3554b18f3416cd9c8cc7a1422df45c2f`.
The generator also writes `dataset_summary.json` and `split_statistics.json`
for checking the 7,000/1,500/1,500 split counts.

### 7.2 Train the Model

The staged configs are
[`configs/first_real_qr_restoration.yaml`](configs/first_real_qr_restoration.yaml),
[`configs/phase1_pilot.yaml`](configs/phase1_pilot.yaml),
[`configs/phase2_pilot.yaml`](configs/phase2_pilot.yaml), and
[`configs/phase3_pilot.yaml`](configs/phase3_pilot.yaml). The final Phase 3 run is:

```powershell
python scripts/train_qr_restoration.py --config configs/phase3_pilot.yaml --resume never
```

Input: the generated dataset and the Phase 3 YAML. Output:
`checkpoints/phase3_pilot/` (including `best.pt`) and
`results/phase3_pilot/` (history, metrics, and run metadata). Phase 3 uses seed
42, batch size 16, AdamW, warm-up plus cosine decay, EMA, and at most 30 epochs.
Training is computationally expensive and is not required merely to validate
the archived result artifacts.

### 7.3 Evaluate a Checkpoint

Place the archived Phase 3 checkpoint at
`checkpoints/phase3_pilot/best.pt`, then run:

```powershell
python scripts/run_final_benchmark.py
```

The command reads
[`configs/final_benchmark.yaml`](configs/final_benchmark.yaml), evaluates the
fixed test split, and writes metrics and latency outputs beneath
`results/final_benchmark/`. Its latency protocol uses batch size 1, 10 warm-up
iterations, and 100 timed iterations. External comparison models are not
included; unavailable external baselines remain marked N/A.

The generic `mambatransqr benchmark` command displays a benchmark recipe; it is
not the measured Phase 3 evaluation entry point.

### 7.4 Run Inference

Single-image inference with the publication configuration uses the existing
Python API because the v1.0.0 command-line wrapper does not reconstruct a model
from a YAML recipe:

```powershell
python -c "from pathlib import Path; import yaml; from mambatransqr.inference import Predictor; from mambatransqr.models import ModelConfig; r=yaml.safe_load(Path('configs/phase3_pilot.yaml').read_text()); p=Predictor(config=ModelConfig(**r['model']), checkpoint='checkpoints/phase3_pilot/best.pt', device='cpu', mixed_precision=False); image,_=p.predict_path('input.png'); Path('outputs').mkdir(exist_ok=True); image.save('outputs/restored.png')"
```

Input is `input.png`; output is `outputs/restored.png`. The generic
`mambatransqr predict` CLI remains available for the package's default
architecture, but it must not be used with the Phase 3 checkpoint because that
checkpoint requires the exact YAML model mapping above.

### 7.5 Run Ablation Experiments

The frozen source can rebuild the tracked ablation report without retraining:

```powershell
python scripts/run_phase4_ablation.py --mode report
python scripts/finalize_ablation_study.py
python scripts/prepare_final_analysis.py
```

These commands consolidate the retained controlled results and publication
analysis under `results/ablation/` and `results/final_analysis/`. The frozen
`configs/phase4_ablation.yaml` records a mixed historical 30-epoch and
15-epoch pilot/resume workflow; it is not a clean one-command reproduction of
all six final 30-epoch runs. The authoritative six-row results, configuration
hashes, checkpoint hashes, and this limitation are retained in the archival
manifest. Full retraining is computationally expensive and requires resolving
one complete Phase 3-derived YAML per variant from that provenance.

### 7.6 Run Tests

```powershell
python -m pytest -q
python -m ruff check src tests scripts
python -m black --check src tests scripts
python -m mypy src tests
```

The first command runs the automated suite. The remaining commands check lint,
formatting, and static typing; they do not retrain models.

## 8. Reproducing the Paper Results

For artifact validation without retraining:

1. Check out tag `v1.0.1` and install the locked environment as in Section 6.
2. Obtain the dataset and checkpoint archival assets associated with the Zenodo
   record; verify them against
   [`docs/peerj_archival_manifest.md`](docs/peerj_archival_manifest.md) and
   `release/peerj_v1.0.0/SHA256SUMS.txt`.
3. Run the dataset verifier and split audit from Section 7.1.
4. Run `python scripts/run_final_benchmark.py` to evaluate the archived Phase 3
   checkpoint on the fixed test split.
5. Compare the generated output with the compact tracked evidence in
   `results/final_benchmark/`, `results/final_analysis/`, and
   `results/ablation_peerj_30e/`.

For full reproduction, first generate or retrieve the dataset, run each staged
training configuration, run the controlled ablations, and then execute the final
benchmark and analysis commands in Section 7. Dataset generation is reproducible
at the documented structural/configuration level; byte identity with the
historical dataset is not asserted because two generator dependency versions
were not recorded.

The selected Phase 3 model has 1,476,259 trainable parameters. The authoritative
reported test metrics, ablation table, latency measurements, checkpoint hashes,
and Figure 4/Figure 5 data are enumerated in the
[archival manifest](docs/peerj_archival_manifest.md), which should take precedence
over rounded narrative summaries.

## 9. Methodology

The workflow is:

1. Generate QR payloads and clean symbols using the fixed dataset configuration.
2. Assign payload groups to train, validation, or test with seed 42 so identical
   QR content cannot cross splits.
3. Apply one deterministic, metadata-recorded synthetic degradation to each
   clean symbol to form paired restoration samples.
4. Load and transform the paired images into normalized model tensors.
5. Train the baseline and staged Phase 1–3 models using the same dataset lineage.
6. Optimize reconstruction, SSIM, edge, QR-structure, and decode-consistency
   objectives with AdamW and the configured schedule.
7. Select the Phase 3 checkpoint by validation PSNR, preserving checkpoint,
   dataset-manifest, configuration, seed, backend, and Git provenance.
8. Restore the fixed test images, calculate image-quality metrics, assess QR
   decoding where supported, and measure CPU latency under the documented
   warm-up/timing protocol.
9. Run controlled single-component ablations from the same inherited Phase 3
   configuration and prepare the tracked manuscript tables and figures.

## 10. Output Files and Results

| Output | Location | Purpose |
|---|---|---|
| Dataset manifest and split metadata | `datasets/first_real_qr_10k/` | Pair paths, parameters, hashes, and split counts |
| Phase 3 checkpoints | `checkpoints/phase3_pilot/` | Best/latest weights and checkpoint provenance |
| Training history and logs | `results/phase3_pilot/` | CSV histories used by Figure 4 and `tensorboard/` event logs |
| Final benchmark | `results/final_benchmark/` | Fixed-test metrics and latency protocol/results |
| Controlled ablations | `results/ablation_peerj_30e/` | Six-variant compact result summaries |
| Final analysis | `results/final_analysis/` | Manuscript tables and Figure 5 quantitative data |
| Figure support | `docs/figures/` | Publication exports and qualitative sample evidence |
| Release checksums | `release/peerj_v1.0.0/SHA256SUMS.txt` | Integrity verification for staged artifacts |

Large datasets and checkpoints are distributed through archival release assets,
not Git history. Small publication-critical JSON, CSV, Markdown, and figure
supporting files remain tracked for inspection.

## 11. Reproducibility Notes

- Frozen publication software: tag `v1.0.0`, commit
  `781bbbd39fbea517f6481e78f837a5e6989d3232`.
- Validated backend: `lightweight_state_space`; official `mamba-ssm` results are
  not claimed.
- Main seed: 42. The dataset manifest and configuration hashes are recorded in
  the archival manifest and checkpoint metadata.
- Validated environment: Windows 10 build 19045, Python 3.12.13, PyTorch
  2.13.0+cpu, CUDA unavailable.
- Hardware fields absent from experiment evidence are explicitly marked
  **NOT RECORDED** in the reproducibility documentation rather than inferred.
- Large assets must be checksum-verified after download. Git alone does not
  contain the complete dataset or trained checkpoints.
- The historical dataset cannot be regenerated byte-for-byte with a verified
  claim because its exact `qrcode` and PyYAML versions were not recorded.
- Physical smartphone/deployment data added after v1.0.0 are outside the frozen
  manuscript experiment lineage.

See [`docs/reproducibility.md`](docs/reproducibility.md),
[`docs/peerj_archival_manifest.md`](docs/peerj_archival_manifest.md), and
[`docs/peerj_release_readiness_final_v2.md`](docs/peerj_release_readiness_final_v2.md)
for detailed provenance and known limitations.

## 12. Citation and References

Machine-readable citation metadata are provided in
[`CITATION.cff`](CITATION.cff). For the unchanged scientific
software/archive, cite:

> Shohida Yusupova, Otabek Khujaev, Sardorbek Davletboyev, and
> Fakhriddin Bekchanov. *Mamba-TransQR*, version v1.0.0. Zenodo.
> https://doi.org/10.5281/zenodo.21913175

Version v1.0.1 changed documentation and reproducibility metadata only and is
archived at Zenodo DOI `10.5281/zenodo.22874149`. Version v1.0.2 is the final
documentation/reproducibility synchronization patch for the PeerJ resubmission.
Neither release changes the v1.0.0 scientific evidence. Machine-readable
release metadata are provided in `CITATION.cff`. Add the final PeerJ article
citation only after verified article bibliographic metadata and DOI are issued.

## 13. License

The source code and associated repository documentation are released under the
MIT License. See [`LICENSE`](LICENSE). The synthetic dataset and
checkpoints/model weights have no separately documented artifact license; this
release does not invent or assign one. Zenodo v1.0.0's historical CC-BY-4.0
record-level metadata does not replace the MIT software license and is not
presented as an artifact-specific license for the dataset or checkpoints.

## 14. Contributing

Contribution instructions are in [`CONTRIBUTING.md`](CONTRIBUTING.md). Please
open an issue before substantial work, keep changes focused, add tests where
appropriate, and run the quality checks in Section 7.6. Do not commit private
data, generated datasets, large checkpoints, or secrets.

## 15. Contact

- Maintainer: Shohida Yusupova
- Email: gratifikus@gmail.com
- Issues: <https://github.com/ShohidaYusupova/Mamba-TransQR/issues>

For reproducibility questions, include the Git commit, configuration path,
dataset-manifest hash, checkpoint hash, backend, and command used.
