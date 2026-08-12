# PeerJ archival release readiness audit

Audit date: 2026-08-12 (Asia/Tashkent)
Scope: pre-release audit only; no tag, release, deposit, DOI, experimental
result, license, or source implementation was changed.

## Executive decision

**NEEDS FIX — the repository is not yet ready for a GitHub archival release or
Zenodo deposit.** Commit `6781bd2899c74c5caaa2179c79cec39c54327bfd` is the
best current source-code/manuscript candidate, but it is not a self-contained
reproducible manuscript release. The numerical sources, histories, dataset
manifest, and checkpoints used by the manuscript are present only in ignored
local directories. A fresh clone cannot trace the reported results to those
artifacts.

There is therefore no existing commit that should be designated as the final
reproducible release. The release commit should be a new descendant of
`6781bd2899c74c5caaa2179c79cec39c54327bfd`, created only after the blocking
items below are resolved and the exact archival artifact manifest is added.

## Repository state

| Finding | Classification | Evidence / action |
| --- | --- | --- |
| Current branch and HEAD | READY | `main` at `6781bd2899c74c5caaa2179c79cec39c54327bfd`. |
| Upstream relation | NEEDS FIX | `main` is four commits ahead of `origin/main` (`2ae5bc5`). Push the reviewed release lineage before release. |
| Tracked modifications | READY | None at audit start. |
| Untracked files | NEEDS FIX | `docs/figures/figure5_experimental_results.{pdf,png,svg}`. Review, validate, document sources, and deliberately add or remove from the release worktree. Do not leave release content untracked. |
| Ignored local state | READY | Virtual environments and tool caches are ignored: `.venv/`, `.venv-validation/`, `.pip-cache/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `.runtime_tmp/`, `.pytest-temp/`, `.pytest_tmp/`, and Python bytecode caches. They must not be archived. |
| Ignored research outputs | NEEDS FIX | `datasets/`, `checkpoints/`, and `results/` are ignored. These contain the evidence required for manuscript reproducibility and need a curated archival strategy, checksums, and provenance rather than wholesale omission. |
| Local temporary/private material | NEEDS FIX | `results/ablation_scratch_orchestrator.ps1` contains the absolute local repository path `C:\Users\UBTUIT\Documents\Mamba-TransQR`; `results/ablation_scratch_orchestrator.log`, `.matplotlib-cache/`, TensorBoard event files containing the host name, scratch/pre-rerun outputs, and redundant epoch checkpoints should not be included in an archive without curation. |
| Large local files | OPTIONAL IMPROVEMENT | Local environments contain duplicate 305 MB Torch DLLs and must remain excluded. Checkpoints contain many roughly 29.9 MB epoch files; archive only the documented canonical checkpoints needed to verify/reuse the work, with SHA-256 values, rather than all redundant epochs. |

The working tree after adding this audit report and its documentation-index
link contains those two intended documentation changes plus the three
pre-existing untracked Figure 5 exports. No other file was changed by the
audit.

## Required manuscript materials

| Material | Classification | Repository evidence |
| --- | --- | --- |
| Model implementation | READY | `src/mambatransqr/models/`, including model, encoder, decoder, Transformer, Mamba wrapper, fusion, patch embedding, and positional encoding. |
| Final Phase 3 configuration | READY | `configs/phase3_pilot.yaml` records the 30-epoch, seed-42, CPU, lightweight-backend recipe. |
| Synthetic QR generation code | READY | `scripts/generate_qr_dataset.py`, `src/mambatransqr/data/qr_generator.py`, `dataset_builder.py`, and generation configurations. |
| Degradation engine | READY | `src/mambatransqr/data/qr_degradation.py`, `damage_generator.py`, and augmentation configuration. |
| Training scripts | READY | `scripts/train_qr_restoration.py`, experiment runners, and `src/mambatransqr/training/`. |
| Validation/test evaluation | READY | Training runner test evaluation plus `src/mambatransqr/evaluation/`, benchmark scripts, and evaluation configuration. |
| Controlled benchmark scripts | READY | `scripts/run_final_benchmark.py`, `finalize_controlled_benchmark.py`, and `configs/final_benchmark.yaml`. |
| Controlled benchmark results | NEEDS FIX | Present locally under ignored `results/final_benchmark/`; absent from the Git tree. Curate and archive the raw CSV/JSON, report, protocol, and provenance. The unavailable comparison baselines correctly remain `N/A`. |
| Ablation configurations | READY | `configs/phase4_ablation.yaml` and `scripts/run_phase4_ablation.py`. |
| Ablation results | NEEDS FIX | Present locally under ignored `results/ablation/`; absent from the Git tree. Archive the controlled 30-epoch table, per-variant resolved configurations/histories/metadata, and canonical checkpoint identities. Exclude or clearly segregate pilot, pre-rerun, and scratch material. |
| Figure 4 histories | NEEDS FIX | Tracked Figure 4 exports cite ignored `results/ablation/full_model/{training_history,validation_history}.csv` (also described as byte-identical to Phase 3 histories). The source CSVs must be archived and checksummed. |
| Figure 5 quantitative data | NEEDS FIX | The local source-provenance table points to ignored Phase 1–3 and ablation CSVs. The three Figure 5 exports are untracked, while tracked `docs/figures/README.md` still says Figure 5 is pending and describes a different planned qualitative panel. Reconcile the figure, caption/index, and archived source data. |
| Reproducibility documentation | NEEDS FIX | `docs/reproducibility.md` is useful and records source lineage/protocols, but explicitly depends on external artifacts at original paths and does not give an archive URL/identifier, artifact inventory, checksums, or one-command verification procedure for a fresh clone plus archive. |
| Environment/dependencies | NEEDS FIX | `pyproject.toml` specifies ranges and extras, but there is no locked environment or exported environment with exact versions used for the reported experiments. Add an archival environment lock/specification and recorded platform/runtime versions. |
| Checkpoint/provenance documentation | NEEDS FIX | Documentation describes required identities and local `source_provenance.csv` exists, but the provenance file and checkpoints are ignored. Archive canonical best checkpoints (or explicitly justify non-distribution), resolved configs, commit, dataset/checkpoint hashes, backend/version, and selection policy. CPU model/core allocation/power state is missing for latency, as already disclosed. |

## Traceability of principal manuscript values

The following values are **READY locally but NEEDS FIX for the repository
release**, because their primary evidence is ignored and unavailable in a
clone.

| Reported item | Local primary evidence | Verification |
| --- | --- | --- |
| Full Phase 3 trainable parameters: 1,476,259 | `results/phase3_pilot/benchmark_summary.csv`; `results/ablation/final/ablation_table.csv` | Exact value `1476259`; also displayed in tracked Figure 2 and documented in `docs/without_mamba_parameter_audit.md`. |
| Test PSNR: 16.452353 dB | `results/phase3_pilot/benchmark_summary.csv`; `results/final_benchmark/benchmark_table.csv` | Source precision `16.4523534166052`, correctly rounded to six decimals. |
| Test SSIM: 0.950374 | Same files | Source precision `0.9503737294927557`, correctly rounded to six decimals. |
| Controlled ablations | `results/ablation/final/ablation_table.csv` | Full 16.452353/0.950374; without Mamba 16.885128/0.954737; without QR structure loss 17.209027/0.958015; without decode-consistency loss 17.335719/0.959144; without EMA 17.112800/0.957167; without refinement decoder 19.220918/0.973401. The design is single-seed, one-factor-at-a-time and does not establish general component harm. |
| Latency | `results/final_benchmark/benchmark_table.csv`; ablation table | Standardized Mamba-TransQR CPU batch-one latency `82.002256... ms` (10 warmups, 100 iterations); ablation full-model run `79.497241... ms`; without-Mamba `12.073368... ms`. The Phase 3 summary separately records `83.927585... ms`. These are different recorded runs and must not be conflated. Hardware identity is incomplete. |
| Without-Mamba parameter audit | `docs/without_mamba_parameter_audit.md`, implementation, configs, and local summaries/checkpoints | Both full and bypass models have 1,476,259 trainable registered parameters; the bypass retains 562,752 inert Mamba-branch parameters and is a compute-branch ablation, not a parameter-matched removal. The tracked audit is internally consistent. |
| Provenance mapping | `results/final_analysis/paper_ready/source_provenance.csv` | Maps publication cells/figures to source fields and transformations, but is ignored and therefore must be included in the archival artifact package. Its current labels reference Figures 5–8 and should be reconciled with the manuscript's final numbering. |

## Absolute and machine-specific paths

| Finding | Classification | Required correction |
| --- | --- | --- |
| Tracked documentation/source | READY | No tracked `C:\Users\...` path was found. Documentation uses repository-relative paths. |
| Ignored orchestration script | NEEDS FIX | `results/ablation_scratch_orchestrator.ps1:2` contains `C:\Users\UBTUIT\Documents\Mamba-TransQR`. Exclude this scratch script from the archive, or replace the value with a repository-relative/runtime-resolved path if the script is intentionally preserved. |
| Machine identity | OPTIONAL IMPROVEMENT | TensorBoard filenames include the local host name. They are not necessary when canonical CSV histories are preserved; exclude them unless there is a documented reason to archive them. |

## README audit

| Requirement | Classification | Finding |
| --- | --- | --- |
| Project description | READY | Clear scope and feature description. |
| Installation | READY | Editable installation, extras, optional Mamba build note, and verification commands are present. |
| Dataset generation | READY | Command, configuration, outputs, supported QR variants, and leakage-safe split behavior are described. |
| Training | READY | General and QR-specific training commands, checkpoint behavior, and guide link are present. |
| Evaluation | READY | Metrics, benchmark command, latency/memory behavior, and guide link are present. |
| Ablation reproduction | NEEDS FIX | Completed artifacts are listed, but the README does not provide the exact controlled ablation command, config, selection of final 30-epoch cohort, expected files, or analysis-only rebuild command. |
| Expected outputs | NEEDS FIX | Dataset and checkpoints have partial output descriptions, and artifact directories are listed, but a reproducible command-to-output inventory and success checks are missing. |
| Reproducibility instructions | NEEDS FIX | README links to the guide but lacks a concise fresh-clone + archived-artifact workflow and verification/checksum command. |
| Lightweight backend disclosure | READY | It explicitly states that completed Phase 3/ablation work used the non-official `lightweight` backend. |
| Distinction from official `mamba-ssm` | READY | The compatibility backend is distinctly named and the official optional backend/extra is described. One training example defaults explicitly to `mamba_ssm`, while the completed-experiment section correctly distinguishes the actual lightweight runs. |

## Citation and license

| Finding | Classification | Required correction |
| --- | --- | --- |
| `CITATION.cff` exists and declares CFF 1.2.0 | READY | File is syntactically small and structurally plausible. Validate again with a CFF validator after edits. |
| Citation author metadata | NEEDS FIX | `Mamba-TransQR Contributors` is a placeholder collective name and is insufficient for the manuscript software citation unless that is intentionally the legal author. Replace it only with confirmed author metadata; do not invent ORCIDs. |
| Citation repository URL | NEEDS FIX | `https://github.com/your-org/Mamba-TransQR` is a placeholder. The configured remote is `https://github.com/ShohidaYusupova/Mamba-TransQR.git`; confirm the canonical public URL before editing. The same placeholder occurs in `pyproject.toml` Homepage/Repository/Issues. |
| Citation version/date | NEEDS FIX | `version: 0.1.0` and `date-released: 2026-07-23` predate the archival release. Update them only when the release version/date are approved. Add manuscript identifiers only when known. |
| Software license | READY | `LICENSE` contains an explicit MIT License; `pyproject.toml` and `CITATION.cff` declare MIT. Do not change without author approval. Confirm the copyright holder wording before archival. |

## Automated tests

Command: `.venv\Scripts\python.exe -m pytest -q`

| Result | Count | Notes |
| --- | ---: | --- |
| Passed | 79 | Core suite passed on the audit machine. |
| Failed | 0 | None. |
| Skipped | 11 | Ten skips because optional `qrcode` is not installed; one because optional `mamba_ssm` is not installed. |

**OPTIONAL IMPROVEMENT:** before archival, rerun the full suite in the locked
release environment with `.[dev,qr-generation]`; separately test the official
backend only if it is claimed as supported for the release. Preserve the CI
run and exact dependency versions. The observed warnings concern explicit
lightweight-backend disclosure, missing optional RSS metrics, and upstream
TorchScript deprecations; no test failed.

## Blocking issues before archival

1. Curate and make available the primary numerical artifacts, Figure 4/5 source
   data, provenance mapping, resolved configurations, dataset manifest, and
   canonical checkpoint identities. Decide whether these belong in Git (small
   CSV/JSON/YAML/Markdown files are suitable) or a versioned external archive;
   add SHA-256 checksums and stable relative paths either way.
2. Reconcile and deliberately track the final Figure 5 exports and update
   `docs/figures/README.md`; verify its panel content, source rows, caption, and
   manuscript numbering.
3. Replace confirmed placeholder citation/package URLs and confirmed author
   metadata; synchronize the approved release version/date. Do not invent
   authors or ORCIDs.
4. Add an exact environment specification and sufficient hardware/runtime
   provenance for latency interpretation.
5. Add a release artifact manifest linking the source commit, dataset manifest
   hash, checkpoint hashes, result files, backend identity, and verification
   commands. Remove/exclude scratch, cache, hostname-bearing, redundant, and
   absolute-path material from the curated archive.
6. Push the four local source commits and the eventual remediation/release
   commit, then audit a fresh clone plus the staged archive before tagging.

## Exact files requiring modification

Existing tracked files that require changes before release:

- `CITATION.cff`
- `pyproject.toml`
- `README.md`
- `docs/reproducibility.md`
- `docs/figures/README.md`
- `.gitignore` (only if selected small canonical result/provenance files will
  be tracked; use narrow negation rules rather than unignoring all results)

Files/artifacts that must be deliberately added or replaced by an archival
manifest pointing to a stable external package:

- `docs/figures/figure5_experimental_results.svg`
- `docs/figures/figure5_experimental_results.pdf`
- `docs/figures/figure5_experimental_results.png`
- canonical Phase 1–3 and full-model training/validation histories used by the
  figures;
- `results/ablation/final/ablation_table.csv` plus controlled-run provenance;
- `results/final_benchmark/benchmark_table.csv` and raw/protocol metadata;
- `results/final_analysis/paper_ready/source_provenance.csv`;
- dataset manifest/checksum and canonical checkpoint/checksum manifest;
- an exact release environment lock/specification.

Do not add `results/ablation_scratch_orchestrator.ps1`, caches, virtual
environments, scratch/pre-rerun directories, redundant epoch checkpoints, or
machine-named TensorBoard files to the archival package.

## Release recommendation

- **Current HEAD:** `6781bd2899c74c5caaa2179c79cec39c54327bfd`
- **Working-tree status at audit start:** no tracked changes; three untracked
  Figure 5 exports; branch four commits ahead of `origin/main`
- **Recommended release commit:** none of the existing commits. Use a new,
  reviewed descendant of `6781bd2899c74c5caaa2179c79cec39c54327bfd`
  containing the remediation and archival manifest; record its full hash only
  after those changes are complete.
- **Recommended semantic version:** `1.0.0` for the first stable,
  manuscript-associated archival software release (subject to maintainer
  approval); keep `0.1.0` if the maintainers intentionally classify the
  manuscript software as pre-stable.
- **GitHub release readiness:** **NO — NEEDS FIX**
- **Zenodo archival readiness:** **NO — NEEDS FIX**

After remediation, repeat this audit from a clean clone, restore only the
curated artifact bundle, verify checksums, regenerate derived tables/figures,
and rerun the locked test environment before creating any tag or deposit.
