# PeerJ release readiness after remediation

Assessment date: 2026-08-12. Status: **NEEDS FIX**.

The repository-side release package is substantially prepared: real repository
metadata and confirmed author identity replace placeholders; version metadata
is synchronized at 1.0.0; Figure 5 is tracked and its quantitative panel is
verified; compact source summaries are tracked; the environment is pinned; and
the external artifact strategy, hashes, hardware limitations, backend identity,
and reproduction commands are documented.

## Remaining blockers

1. Three current local ablation `best.pt` hashes do not equal the hashes stored
   in `results/ablation/reproducibility.json` and
   `results/ablation/ablation_summary.json`: without DecodeConsistencyLoss,
   without EMA, and without refinement decoder. Recover the recorded
   checkpoints or establish the correct serialization before deposit.
2. Standalone restored PNG files used in Figure 5 are not present. Recover or
   deterministically regenerate them from the verified Phase 3 checkpoint and
   fixed test inputs, then hash and include them in the results archive.
3. Final Zenodo archive files, archive-level sizes/hashes, DOI, and URL do not
   yet exist by instruction. Stage and verify the curated bundles before
   release; update the manifest after staging and before tagging/deposit.
4. The release lineage is local and must be pushed/reviewed. A clean-clone plus
   restored-bundle verification remains required.
5. Original dataset-generator versions of `qrcode` and PyYAML were not
   recorded. The verified generated dataset must be deposited; byte-identical
   regeneration must not be claimed.

## Verification results

Final command results are recorded immediately before the release-preparation
commit:

- Pytest: **89 passed, 0 failed, 1 skipped**. The sole skip is the optional
  official `mamba_ssm` backend, which is intentionally outside the validated
  lightweight experiment environment.
- Ruff: **passed** (`ruff check .`).
- Package build: **passed**; built
  `mambatransqr-1.0.0-py3-none-any.whl` (114,162 bytes; build SHA-256
  `f076d43014fc5ebf196dcd4111ee714421c42245bd37b5fe1700a39163da7816`).
  The wheel is a disposable validation artifact and is not committed.
- Parameter count: **passed**, 1,476,259 trainable parameters reconstructed
  from `configs/phase3_pilot.yaml`.
- Figure 5 consistency: **passed for the controlled quantitative panel**; all
  six source rows match the requested PSNR/SSIM values at six decimals and the
  PNG was visually inspected for matching labels and rendering defects. PDF
  text/page structure and all three export hashes were also checked. The
  missing standalone restored inputs remain a deposit blocker as stated above.

## Release decision

- Current HEAD before this release-preparation commit:
  `62e2e5b8a90c73aeee65a5358d66d9053e4858a4`.
- Final release-preparation commit: populated by Git after this report is
  committed; the commit itself is the authoritative identity.
- Recommended release version: `1.0.0`.
- Recommended tag: `v1.0.0` (do not create until blockers are resolved).
- GitHub release can be created now: **NO**.
- Zenodo archival can be created now: **NO**.

## Files added or modified

- Metadata/version: `CITATION.cff`, `pyproject.toml`,
  `src/mambatransqr/_version.py`.
- Reproduction: `README.md`, `requirements-peerj-lock.txt`,
  `docs/reproducibility.md`, `docs/peerj_archival_manifest.md`.
- Figures/docs: `docs/figures/figure5_experimental_results.{svg,pdf,png}`,
  `docs/figures/README.md`, `docs/README.md`.
- Compact results: Phase 3 benchmark/histories, controlled ablation summaries,
  final benchmark tables/protocol, and publication source provenance under
  their existing `results/` paths.
- This final readiness report.
