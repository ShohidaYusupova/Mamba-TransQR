# Final state-of-the-art benchmark report

## Scope and availability

The completed Phase 3 checkpoint was evaluated without retraining. Degraded Input is the identity baseline. SRCNN, ESPCN, VDSR, EDSR, and SwinIR are listed as unavailable because the repository contains neither implementations nor checkpoints for them; the baseline registry explicitly prohibits substituting another model. Their metrics remain `N/A` rather than fabricated.

The reported decode rate is the Phase 3 threshold-based decode proxy (`mean(|output - 0.5| > 0.25)`), not payload recovery by ZBar/ZXing. No decoder backend is installed, so a true payload decode rate is unavailable for every model.

## Strengths of Mamba-TransQR

Mamba-TransQR improves PSNR over Degraded Input by **+5.835291 dB** and SSIM by **+0.246855** on the fixed 1,500-image test split. It uses the verified EMA-selected Phase 3 checkpoint and preserves the project's normalized image contract.

## Weaknesses

The model has **1,476,259 parameters** and measured batch-one CPU latency of **82.002 ms**, compared with **0.002 ms** for the identity input. Its PSNR remains modest, and the absence of an installed QR decoder prevents reporting actual payload recovery.

## Comparison with CNN baselines

No defensible comparison with SRCNN, ESPCN, VDSR, or EDSR can be made from this workspace. Training new variants would introduce architecture/training choices not specified by the experiment, while importing generic super-resolution checkpoints would violate the same-dataset requirement. No ranking claim is made.

## Comparison with the Transformer baseline

SwinIR is likewise unavailable locally. Without a checkpoint trained under the same dataset and preprocessing, a numeric Transformer comparison would be misleading; its table cells remain `N/A`.

## Computational trade-offs

Mamba-TransQR trades approximately 1.48 million parameters and substantial CPU latency for improved reconstruction quality over the degraded input. The missing baseline artifacts prevent determining whether this trade-off is favorable relative to CNN or Transformer alternatives.
