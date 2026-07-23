# Optimization plan: raise QR-restoration PSNR

## Scope and evidence

This is a planning document only. It makes no model, data, or training changes.
It is based on the completed first experiment:

| Observation | Value |
| --- | ---: |
| Initial train loss | 2.473154 |
| Final train loss | 1.299901 |
| Peak validation PSNR | 10.019730 dB at epoch 9 |
| Final validation PSNR | 9.970476 dB at epoch 30 |
| Peak validation SSIM | 0.755690 at epoch 30 |
| Test PSNR / SSIM | 9.932789 dB / 0.753203 |

The target of 25–30 dB should be treated as a staged empirical goal, not a
guarantee. It may be infeasible for individual 128 px version-20 codes after
the current print/scan degradation: their modules are approximately one pixel
wide before blur, resampling, noise, and JPEG compression. The next run must
report results by QR version and degradation operator before claiming an
aggregate target is reachable.

## Diagnosis

### 1. Loss function: pixel fidelity is underweighted

`MultiScaleRestorationLoss` combines Charbonnier, global SSIM, Sobel edge, QR
structure, and decode-surrogate losses. In the final train epoch its recorded
component values were approximately:

| Component | Raw value | Configured weight | Approx. contribution |
| --- | ---: | ---: | ---: |
| Reconstruction | 0.2153 | 1.0 | 0.2153 |
| SSIM loss | 0.2456 | 0.2 | 0.0491 |
| Edge | 0.8505 | 0.2 | 0.1701 |
| Structure | 0.8127 | 1.0 | 0.8127 |
| Decode consistency | 0.2633 | 0.2 | 0.0527 |

The structural term therefore dominates the optimized objective by almost four
times the reconstruction term. It can reward a structurally plausible but
pixel-inaccurate result, which is consistent with PSNR flattening around 10 dB.
The SSIM implementation is global over the whole image rather than windowed,
so it does not directly enforce local QR-module geometry.

### 2. Normalization and output range are inconsistent

Inputs and targets are converted to RGB tensors in `[0, 1]`, which is correct.
The decoder, however, ends in an unconstrained linear projection: there is no
sigmoid, clamp, or range-preserving residual output. Losses and PSNR are then
computed as if outputs were in `[0, 1]`, and the SSIM loss is configured with
`data_range=1.0`. Out-of-range predictions can degrade both the metric and the
gradients, especially on the black/white QR distribution.

### 3. Optimization has no learning-rate schedule

The experiment used AdamW at a fixed `2e-4`; the runner constructs no scheduler.
PSNR reached its maximum at epoch 9, then fell by 0.0493 dB by epoch 30, while
SSIM continued rising. This is evidence that the constant rate keeps moving the
model after its best PSNR region. There is also no early stopping or restore of
the best monitored state.

### 4. EMA is evaluated but is not the default inference state

Validation and test temporarily apply the EMA parameters. The checkpoint saves
the raw model under `model` and the EMA shadow separately. Standard predictor
loading restores `model`; it does not automatically apply `ema.shadow`. Thus a
normal inference run from `best.pt` can differ from the model whose validation
PSNR selected the checkpoint. The analysis grid applied the stored EMA manually.

### 5. Data diversity and difficulty are not controlled

The 10,000-pair dataset uses one static `moderate` degradation per clean QR,
chosen from two operators. There are no training-time augmentations. The
`print_scan` operator can combine down/up sampling, contrast reduction, blur,
noise, and JPEG quality 20–65; module occlusion and dropout can remove content
irreversibly. Neither severity/operator strata nor damaged-image baselines were
logged, so the recoverable and unrecoverable portions of the score are unknown.

### 6. Model capacity and patch resolution limit reconstruction

The run uses `embed_dim=64`, `depth=4`, and `patch_size=16`: a 128×128 QR is
represented by only an 8×8 token grid. Each output patch is independently
produced by one linear layer from one token. There is no decoder-side spatial
mixing, image-space refinement, skip connection, residual-to-input path, or
bounded output activation. This favors smooth/patchwise reconstructions over
sharp module boundaries. The selected `lightweight` backend is explicitly
described in the repository as a compatibility/CPU backend, not official Mamba.

## Measurement gate: run before tuning

Do these diagnostics first, keeping the current model fixed. They prevent a
misleading aggregate PSNR gain.

1. Evaluate the damaged input itself against clean targets for PSNR, SSIM, and
   QR decode rate; report restoration gain over that baseline.
2. Evaluate the EMA-applied model and raw `model` state separately on the same
   validation/test set. Record their PSNR difference.
3. Report mean and median per-image PSNR/SSIM by QR version, error-correction
   level, degradation operator combination, and payload type.
4. Log prediction minimum, maximum, fraction below 0, and fraction above 1 at
   every validation epoch. Save a fixed qualitative panel selected before
   training.
5. Add a true QR decoder evaluation where the optional decoder is available;
   report exact payload recovery separately from the current threshold proxy.
6. Validate the metric implementation against a trusted windowed-SSIM reference
   on a small fixed batch. Do not compare new and old SSIM values without noting
   the definition.

**Gate:** do not target 25–30 dB until the damaged baseline and per-stratum
scores identify a recoverable subset with a meaningful headroom to that range.

## Prioritized optimization plan

### Phase A — Correct the training signal (highest priority, architecture preserved)

1. **Bound the output to the target range.** Use a sigmoid final activation, or
   predict a residual added to the damaged input and clamp only at evaluation.
   Test both, with the same seed and data. Verify gradients do not saturate.
   This corrects the metric/loss range contract without changing the encoder or
   decoder topology.
2. **Retune and normalize loss terms for PSNR.** Start with a pixel-first
   ablation: Charbonnier (or L1) only, then add a *windowed* SSIM term at a
   small coefficient, then edge/QR terms one at a time. Normalize each term by
   an observed running scale or select weights so no auxiliary term contributes
   more than the reconstruction term initially. A recommended initial sweep is
   reconstruction weight 1.0; windowed-SSIM 0.05/0.10; edge 0/0.05; structure
   0/0.10; decode 0/0.05. These are candidates, not established values.
3. **Measure PSNR on unclamped and clamped outputs during transition.** The
   final reported metric should be on a documented, valid `[0,1]` image range;
   retain the other value for debugging only.
4. **Use QR-aware terms as late or conditional refinements.** Enable structure
   and decode losses after pixel reconstruction becomes stable, or ramp them
   over early epochs. Their purpose is module fidelity/decode success, not to
   dominate pixel PSNR.

**Success criterion:** on the same split, exceed the current epoch-9 validation
PSNR by at least 1 dB without reducing true decode rate. If not, inspect the
output-range and damaged-baseline logs before moving to Phase B.

### Phase B — Make optimization select the desired model

1. Add a warmup followed by cosine decay or `ReduceLROnPlateau` on validation
   PSNR. Candidate grid: warmup 1–3 epochs; peak LR `1e-4`, `2e-4`, `3e-4`;
   minimum LR `1e-6` to `1e-5`. Keep AdamW weight decay in a separate sweep
   (`0`, `1e-5`, `1e-4`) because QR images have a highly discrete target.
2. Use early stopping on validation PSNR and restore the best state. Since the
   current maximum is epoch 9, a 5–8 epoch patience is a reasonable initial
   guardrail.
3. Save a deployable best checkpoint whose `model` weights are exactly the
   EMA weights used for selection, or make the predictor explicitly apply the
   stored EMA. Preserve the raw model in a distinct field for resuming.
4. Sweep EMA decay (`0.99`, `0.995`, `0.999`, disabled) and compare EMA versus
   raw state, rather than assuming the current value helps. At only 30 epochs,
   `0.999` can lag substantially.
5. Fix random seeds, log effective updates, and run at least three seeds for
   candidates that beat baseline. Use the same held-out test set once per final
   selection only.

**Success criterion:** repeatable validation gains across three seeds and a
test score within a predeclared tolerance of validation performance.

### Phase C — Improve data curriculum without overstating the result

1. Replace the single static moderate pair with deterministic, stratified
   severity/operator coverage. Keep train, validation, and test payload groups
   disjoint as now.
2. Train with a curriculum: clean/mild and recoverable moderate examples first;
   introduce stronger print/scan, occlusion, and finder damage gradually. Keep
   a fixed moderate-only validation set for comparability and a separate mixed
   stress test for robustness.
3. Add *paired-consistent* geometry augmentations (same crop/flip/rotation to
   damaged and clean) only when QR orientation and quiet zones remain valid.
   Apply appearance noise, blur, JPEG, scale, and contrast to the damaged image
   only within calibrated real-world ranges. Do not randomly augment clean
   targets independently.
4. Calibrate synthetic degradation against representative real scans/prints.
   The current `print_scan` ranges should be treated as hypotheses, not ground
   truth. Maintain per-operator validation subsets so a gain is not caused by
   simplifying the data.
5. Increase image size or limit high-version codes when the module raster is
   below a verified minimum sampling width. Report separate scores for
   versions 1, 5, 10, and 20.

**Success criterion:** an improvement on the original fixed moderate test set,
not merely on an easier or rebalanced dataset.

### Phase D — Capacity/configuration sweep (architecture preserved)

The current 16-pixel patch size is the most likely representation bottleneck.
Before changing topology, sweep supported configuration values on the same
protocol:

| Variable | Candidates | Rationale |
| --- | --- | --- |
| Patch size | 8, then 4 if compute allows | Retains spatial/module detail; 16 creates only 8×8 tokens. |
| Embedding width | 96, 128, 192 | Adds token capacity after the signal is corrected. |
| Depth | 6, 8 | Adds fusion capacity; compare parameter-matched variants. |
| Heads | 4, 6, 8 (dividing embedding width) | Tests attention granularity. |
| Mamba backend | lightweight for comparability; official backend on supported CUDA environment | Separates a CPU compatibility baseline from the intended implementation. |

Smaller patches sharply increase token count, and the lightweight recurrence is
slow on CPU. Run this phase on a supported GPU or use a controlled subset for
screening, then retrain selected candidates on the full protocol. Capacity
alone should not be expected to fix an unbounded output or a misweighted loss.

### Phase E — Decoder assessment (defer unless phases A–D plateau)

The current decoder is a normalized token followed by one linear projection to
a non-overlapping 16×16 pixel patch. This is a plausible source of block
artifacts and missing sharp module boundaries.

Because the present request prohibits an architecture change, do not modify it
in the next experiment. If the architecture constraint is lifted after the
earlier phases, compare a patch-overlap or convolutional refinement decoder,
input skip/residual reconstruction, and a bounded final head. Evaluate each
against an equal-parameter linear-head baseline and retain only changes that
improve both PSNR and decode rate on the fixed test set.

## Experiment sequence and decision rules

1. Establish measurement gate and damaged baseline.
2. Run Phase-A loss/range ablations with current capacity, same 10k split, and
   early stopping; choose by validation PSNR plus decode rate.
3. Add Phase-B schedule/EMA correction to the selected Phase-A recipe.
4. Run Phase-C curriculum only with a fixed legacy test set retained.
5. Run the Phase-D patch-size sweep; select using validation only, then execute
   one final held-out test evaluation.
6. Consider Phase E only if the best architecture-preserving configuration
   remains materially below the recoverable-stratum target.

For every run, record config hash, commit, raw/EMA checkpoint identity, split
manifest, output range statistics, train/validation curves, PSNR, windowed
SSIM, and decoder success. Do not describe a 25–30 dB target as achieved until
it is reproduced on the fixed test split with those artifacts.
