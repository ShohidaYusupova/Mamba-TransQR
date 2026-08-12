# PeerJ v1.0.0 archival readiness - final reconciliation

Assessment date: 2026-08-12. Status: **READY FOR RELEASE CREATION, SUBJECT TO
PUSH/REVIEW**.

No model was retrained, no manuscript numerical result changed, and no tag,
GitHub release, Zenodo deposit, or DOI was created.

## Release candidate

The release candidate is the commit containing this report and the reconciled
manifest (`HEAD`; its immutable full hash is reported in the final handoff).
Recommended version/tag: `1.0.0` / `v1.0.0`.

## Checkpoint reconciliation

The recorded hashes were located exactly in
`checkpoints/ablation_pre_rerun/<variant>/best.pt`. The three recorded/current
pairs have byte-identical logical model and raw-model tensor streams and
identical epoch-30 state, architecture, backend, and validation metrics. The
current files are enriched re-serializations with resume/RNG/early-stopping
fields. The original recorded-hash files are selected for archival because
they are the exact serializations cited by the final provenance.

| Variant | Recorded/final SHA-256 | Current enriched SHA-256 | Outcome |
| --- | --- | --- | --- |
| Without DecodeConsistencyLoss | `a9f53d7d466e71b35932fa4fec96c2a497eea019bdf8e283abb4b1d6e1169500` | `9bd0d1166ffccbe723d309c7558f4463656d63e9290cc27d8d5db0e8db9e9dcd` | C: same trained artifact, different serialization; archive recorded original |
| Without EMA | `bf3f32fa8de2b30c3ae19da5540f4b0954bb32520f5fe3303e0368aac39ff606` | `bff08d0f6883d167ab5ca34e6267e014d5001d4da435494332d59e279c0585a6` | C; archive recorded original |
| Without Refinement Decoder | `e83def556d5f45da753ca2f569092f7fe7dde59686c08958289a5cfe0043e752` | `f5dce71cf3321ace5b4819521bc16141ea537849bd88a3d0592d12ec9a50e679` | C; archive recorded original |

Full field-level evidence and selection reasons are in
`docs/peerj_archival_manifest.md`.

## Figure 5 provenance

All four clean/degraded/restored triplets are tracked under
`docs/figures/figure5_samples/`. Clean/degraded inputs are byte-identical fixed
test-split files. Restored outputs were generated with deterministic CPU,
non-AMP inference from the verified Phase 3 EMA checkpoint. `manifest.csv`
records every hash, degradation, checkpoint, configuration, and seed.
Reverse-raster comparison with the images embedded in the Figure 5 SVG yields
98.284912%-99.945068% exact RGB pixels and mean absolute error 0.008097-3.133586;
visual inspection confirms correspondence. Differences are panel-resampling
boundaries; no image was edited.

## Local Zenodo staging

Five deterministic bundles are staged under `release/peerj_v1.0.0/`:

- source and reproducibility;
- verified dataset;
- six canonical recorded checkpoints;
- compact manuscript results/provenance;
- Figure 5 supporting data.

Their exact byte sizes and SHA-256 values are recorded in
`release/peerj_v1.0.0/SHA256SUMS.txt`. ZIP payloads remain local/ignored and
have not been uploaded.

## Clean-clone verification

A fresh local clone of the committed release candidate was verified using only
the clone, staged bundles, the pinned requirements, and documented paths. It
passed package import/version, Phase 3 construction and 1,476,259 trainable
parameters, compact-result access, Figure 5 sample access and hashes,
checkpoint reconciliation, checksum validation, Ruff, the relevant tests, and
wheel build. No undocumented original-workspace result/dataset/checkpoint path
was used after bundle extraction.

## Validation summary

- Full tests: **89 passed, 0 failed, 1 skipped** (`mamba_ssm`, intentionally
  optional and outside the lightweight reproduction claim).
- Ruff: **passed**.
- Wheel build: **passed** for `mambatransqr-1.0.0-py3-none-any.whl`.
- Phase 3 parameter count: **1,476,259**, passed.
- Figure 5 quantitative values: **6/6 matched**, passed.
- Checkpoint reconciliation: **6/6 canonical hashes verified**, passed.
- `SHA256SUMS.txt`: **all entries verified**, passed.
- Clean-clone plus bundles: **passed**.

## Remaining release administration

There is no remaining evidence/provenance blocker. The candidate and local
archive assets still require maintainer review, push, and final archive upload.
After upload, record the Zenodo URL/DOI in metadata before or as part of the
approved release workflow.

- GitHub release may now be created: **YES, after push/review and checksum
  confirmation of the uploaded assets**.
- Zenodo archival may now be created: **YES, after push/review; upload exactly
  the staged bundles and preserve their recorded hashes**.
