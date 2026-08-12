# PeerJ archival manifest

This manifest defines the source and external artifact package supporting the
PeerJ Computer Science manuscript. Hashes are SHA-256. Paths are relative to
the repository root. Large datasets and checkpoints are not stored in Git.

## Source identity

| Purpose | Commit |
| --- | --- |
| Phase 3 experiment source recorded by the ablation package | `b8fee1fcbc2da8fa39a78ce4b2a183e0c8bdfdbc` |
| Controlled ablation runner | `86d1a12` |
| Automatic ablation resume | `ca3c7b0` |
| Final benchmark | `f38381f` |
| Controlled benchmark finalization | `2f06fa15b2049351de8bc393f1fc3303299ccae7` |
| Without-Mamba audit and pre-release source candidate | `6781bd2899c74c5caaa2179c79cec39c54327bfd` |
| Pre-release readiness audit | `62e2e5b8a90c73aeee65a5358d66d9053e4858a4` |
| Final archival release candidate | `HEAD` containing `docs/peerj_release_readiness_final_v2.md`; record the immutable full commit in the release metadata and final handoff |

## Backend and protocol identity

- Experiment backend: `mamba_backend: lightweight`.
- Implementation: `lightweight_state_space` / `LightweightStateSpaceBlock`.
- Official `mamba-ssm` version: `null`; it was not used and is not covered by
  the reproduction claim.
- Dataset seed: 42; split: 7,000 train, 1,500 validation, 1,500 test; one
  moderate degradation per unique payload.
- Model evaluation batch size: 16.
- Latency: CPU, batch size 1, 10 warm-ups, 100 timed iterations.
- Runtime: Python 3.12.13, PyTorch 2.13.0+cpu, Windows 10 build 19045.
- CPU model, RAM, core allocation, thread count, power state, and installed GPU:
  **NOT RECORDED**. Recipes explicitly used `device: cpu`.

## Dataset identity

| File / identity | Bytes | SHA-256 |
| --- | ---: | --- |
| `datasets/first_real_qr_10k/dataset_manifest.csv` | 2,977,028 | `2c47950acb452137b3634bb3527cd48f3554b18f3416cd9c8cc7a1422df45c2f` |
| `datasets/first_real_qr_10k/split_statistics.json` (split hash) | 181 | `fadb5c61b22afa65d82f6c66a99b5b8972ac9646212a237ce238927cc5adeca7` |
| `datasets/first_real_qr_10k/dataset_summary.json` | 80 | `ddc67d4052304e4e0a2c6729c94b8e0b8a9d9de91e9b971a3390d281ced41e92` |
| Complete `datasets/first_real_qr_10k/` currently present | 30,003 files | 163,411,356 bytes (directory size; not a content hash) |

The manifest hash is the experiment's recorded dataset identity. A canonical
compressed dataset archive and its archive-level SHA-256 must be generated and
added here immediately before deposit. Exact `qrcode` and PyYAML versions used
for original dataset generation were **NOT RECORDED**, so byte-identical
regeneration must not be claimed; deposit the verified dataset.

## Checkpoints

| Variant | Required archive path | Bytes | Current SHA-256 | Recorded SHA-256 | Status |
| --- | --- | ---: | --- | --- | --- |
| Full Phase 3 / full model | `checkpoints/phase3_pilot/best.pt` | 29,799,773 | `a0e1abf2c3241a360dc7f5a66936aa838ad5d170a2c4a2cc44db1b98ee7099d6` | same | Verified; byte-identical to `checkpoints/ablation/full_model/best.pt` |
| Without Mamba | `checkpoints/ablation/without_mamba/best.pt` | 24,334,055 | `2e172078ae467debafdbe7a7b4d1ab41fe91086d4fb7c55f6237ed70539ff9f6` | same | Verified |
| Without QRStructureLoss | `checkpoints/ablation/without_qr_structure_loss/best.pt` | 29,799,901 | `f2f45dd5707c52270cf4c79c19167bac943b30a18fc4f5788b9cec25c05bcf99` | same | Verified |
| Without DecodeConsistencyLoss | `checkpoints/ablation_pre_rerun/without_decode_consistency_loss/best.pt` | 29,799,901 | `a9f53d7d466e71b35932fa4fec96c2a497eea019bdf8e283abb4b1d6e1169500` | same | Reconciled; original evaluated serialization |
| Without EMA | `checkpoints/ablation_pre_rerun/without_ema/best.pt` | 17,897,377 | `bf3f32fa8de2b30c3ae19da5540f4b0954bb32520f5fe3303e0368aac39ff606` | same | Reconciled; original evaluated serialization |
| Without Refinement Decoder | `checkpoints/ablation_pre_rerun/without_refinement_decoder/best.pt` | 29,570,733 | `e83def556d5f45da753ca2f569092f7fe7dde59686c08958289a5cfe0043e752` | same | Reconciled; original evaluated serialization |

### Checkpoint reconciliation

All 369 local `.pt` files were hashed. The three recorded hashes were found
under `checkpoints/ablation_pre_rerun/`. For each variant, the recorded file
and current `checkpoints/ablation/.../best.pt` have identical logical hashes
for every deployable `model` tensor and every `raw_model` tensor, identical
epoch-30 state/global step/best epoch/validation metrics, and identical model
architecture and backend. The later current files add `grad_scaler`,
`early_stopping`, `rng_state`, `resume_identity`, and `resume_events`, and use a
repository-relative reproduction recipe; serializing that enriched mapping
changes the whole-file SHA-256 without changing trained weights.

| Variant | Recorded hash source | Current hash | Config hash | Dataset hash / seed | Resumed | Final artifact and reason |
| --- | --- | --- | --- | --- | --- | --- |
| Without DecodeConsistencyLoss | final `ablation_raw_results.csv`, `ablation_summary.json`, and `reproducibility.json`; original file found locally | `9bd0d1166ffccbe723d309c7558f4463656d63e9290cc27d8d5db0e8db9e9dcd` | `4fc78dc1acd76309eb4543e77f4c34d03440968e8e4ddb0d337601bfc3321f4c` | `2c47950...45c2f` / 42 | false | Recorded-hash pre-rerun file: it is the exact serialization named by final provenance and has the same trained state as the enriched copy. |
| Without EMA | same sources; original file found locally | `bff08d0f6883d167ab5ca34e6267e014d5001d4da435494332d59e279c0585a6` | `08b3f1fb1846e80d1ac9a976a80341390f895c63ecc1cb0918bf344d47903933` | `2c47950...45c2f` / 42 | false | Recorded-hash pre-rerun file for the same reason. |
| Without Refinement Decoder | same sources; original file found locally | `f5dce71cf3321ace5b4819521bc16141ea537849bd88a3d0592d12ec9a50e679` | `7abaf99c86256c29e604ab7ebde051013d5f3a7e95c746eb1825aadfcdbe2ad3` | `2c47950...45c2f` / 42 | false | Recorded-hash pre-rerun file for the same reason. |

All three use the lightweight backend, base experiment commit
`b8fee1fcbc2da8fa39a78ce4b2a183e0c8bdfdbc`, 30 epochs, 13,140 global
steps, and the variant architecture recorded in the resolved configuration.
The test metrics remain those in the final controlled results; checkpoints do
not contain test metrics and none were inserted or altered. Outcome: category
**C** at the file-serialization level, with identical trained tensors; the
recorded original serialization is the unambiguous archival choice.

## Publication-critical tracked summaries

| Artifact | Bytes | SHA-256 | Use |
| --- | ---: | --- | --- |
| `results/phase3_pilot/benchmark_summary.csv` | 187 | `56956305140352ce50b392b7994efa366488a1021351df5891e0c5019104d31d` | Phase 3 test metrics, parameters, historical latency |
| `results/phase3_pilot/training_history.csv` | 3,830 | `f0329e61c08b8dc84292d144db6665075034b10b84f405a2a58fe911d95f2d8d` | Figure 4 training loss |
| `results/phase3_pilot/validation_history.csv` | 3,158 | `b82937a0aa4f3325dbe38f8f1c6bb1c6882906ec43a21880ddd5f3f764d980c6` | Figure 4 PSNR/SSIM |
| `results/ablation/final/ablation_table.csv` | 782 | `eebcd3f75235056e19f6f49f022025b7c0ad72752c220eb1d69300d1b4bf11eb` | Figure 5 quantitative panel and manuscript ablation table |
| `results/final_benchmark/benchmark_raw_results.json` | 1,130 | `5d8826d56bb624cba85157babe5cb594aa902886ba19a3750614d2eb28d3d327` | Exact benchmark rows |
| `results/final_benchmark/benchmark_table.csv` | 297 | `ccbda56937ce21248c89caca15da19a990a7327a5192ba1d217619eb22af2681` | Manuscript benchmark table |
| `results/final_analysis/paper_ready/source_provenance.csv` | 4,010 | `d0b641a2c923e450fb34f402843edb4113707cb0e86df100b400911e724da4f1` | Publication cell/figure mapping |

The tracked ablation package also includes `results/ablation/ablation_summary.json`
and `results/ablation/reproducibility.json`; the tracked benchmark package
includes its report and reproducibility protocol. These are primary compact
provenance records, not reconstructed values.

## Principal values

| Configuration | Parameters | Test PSNR (dB) | Test SSIM | Ablation latency (ms) |
| --- | ---: | ---: | ---: | ---: |
| Full Phase 3 | 1,476,259 | 16.4523534166052 | 0.9503737294927557 | 79.49724099598825 |
| Without Mamba | 1,476,259 | 16.885127767603446 | 0.9547370697589631 | 12.073367995908484 |
| Without QRStructureLoss | 1,476,259 | 17.209026529433878 | 0.9580150531961563 | 74.12841899669729 |
| Without DecodeConsistencyLoss | 1,476,259 | 17.33571861145344 | 0.9591436956791167 | 75.27521498501301 in the finalized table; 74.70554600120522 in the earlier summary |
| Without EMA | 1,476,259 | 17.1128003952351 | 0.957166960898866 | 74.29265097947791 in the finalized table; 75.9409920021426 in the earlier summary |
| Without Refinement Decoder | 1,465,248 | 19.220917661139307 | 0.9734007925429242 | 70.82787598017603 in the finalized table; 71.75717299804091 in the earlier summary |

The standardized final benchmark records Mamba-TransQR latency
`82.00225600623526 ms`, batch size 1, 10 warm-ups, 100 iterations. Phase 3's
training summary separately records `83.92758500413038 ms`. These are distinct
runs and must not be conflated. Loss/EMA latency differences are measurement
variation because their inference graphs are unchanged.

## Figure 5 identity

| Export | Bytes | SHA-256 |
| --- | ---: | --- |
| `docs/figures/figure5_experimental_results.pdf` | 185,897 | `0dfa49da3a078cd15e57bc784a7373b6eb786580aa320ccd58914486c2d5e873` |
| `docs/figures/figure5_experimental_results.png` | 659,484 | `a0ae01b41aaa81ea5bed388cb9bca9c4f8cbb50b437d121b04d7cec9ad8db56a` |
| `docs/figures/figure5_experimental_results.svg` | 245,898 | `ad2c01621c747624a3136741104d8aaff462e8ba16459431e157ff7697660b4a` |

The displayed controlled-ablation PSNR/SSIM values match the table after
rounding. The four clean/degraded/restored triplets and their hashes are now
tracked under `docs/figures/figure5_samples/`. Restored files were regenerated
by deterministic CPU inference from the verified Phase 3 EMA checkpoint; the
sample manifest records inference settings and quantitative correspondence to
the raster images embedded in the SVG. No image was manually edited.

## Future Zenodo/supplementary deposit

The following deterministic ZIPs are staged locally under
`release/peerj_v1.0.0/` but have not been uploaded. Archive and critical-file
hashes are in `release/peerj_v1.0.0/SHA256SUMS.txt`; keeping the source ZIP's
own hash outside the source ZIP avoids a circular self-hash.

| Staged bundle | Bytes | SHA-256 |
| --- | ---: | --- |
| `source_and_reproducibility/mambatransqr-peerj-source-and-reproducibility.zip` | See `SHA256SUMS.txt` | See `SHA256SUMS.txt` |
| `dataset_or_dataset_manifest/mambatransqr-peerj-dataset.zip` | 159,957,360 | `87390c58af5aa71dbb128ac7a236b82f1c56ad3f837712fa5b933833df4fecfe` |
| `checkpoints/mambatransqr-peerj-checkpoints.zip` | 147,608,332 | `fd153768cc3609b18df531ca8ed2067d25666c9e541793aa4ed92effc15ad04d` |
| `manuscript_results/mambatransqr-peerj-results.zip` | 46,965 | `893afd78146ed4a058e20e52d4de79065ed00d7a5f9667de8ef0ef47f2963295` |
| `figure_supporting_data/mambatransqr-peerj-figure5-data.zip` | 1,028,150 | `5880c5a79d7f1f8dda99f387931af24c8f726cf1e3d25ca437348a71c3be847c` |

Exclude virtual environments, caches, scratch orchestration files, pre-rerun
and pilot outputs, redundant epoch/latest checkpoints, machine-named
TensorBoard files, and absolute local paths. Record archive byte sizes and
SHA-256 hashes in this manifest after creating the final packages.
