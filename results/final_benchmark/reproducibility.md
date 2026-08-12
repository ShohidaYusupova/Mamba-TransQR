# Reproducibility

## Evaluation protocol

- Dataset: `datasets/first_real_qr_10k`
- Split: `test` (1,500 fixed pairs)
- Dataset manifest SHA-256: `2c47950acb452137b3634bb3527cd48f3554b18f3416cd9c8cc7a1422df45c2f`
- Checkpoint: `checkpoints/phase3_pilot/best.pt`
- Checkpoint SHA-256: `a0e1abf2c3241a360dc7f5a66936aa838ad5d170a2c4a2cc44db1b98ee7099d6`
- Checkpoint training: reused completed Phase 3 best checkpoint; no retraining
- Preprocessing: RGB conversion and normalization to `[0, 1]`
- Evaluation batch size: 16
- Latency: batch size 1, 10 warmups, 100 timed iterations
- Metrics: Phase 3 batch-averaged PSNR, SSIM proxy, threshold decode proxy, and synchronized latency
- Runtime: Python 3.12.13, Torch 2.13.0+cpu, Windows-10-10.0.19045-SP0

## Unavailable baselines

- SRCNN: No local implementation or checkpoint; substitution is prohibited.
- ESPCN: No local implementation or checkpoint; substitution is prohibited.
- VDSR: No local implementation or checkpoint; substitution is prohibited.
- EDSR: No local implementation or checkpoint; substitution is prohibited.
- SwinIR: No local implementation or checkpoint; substitution is prohibited.

No model was trained, no dataset was generated, and no unavailable value was estimated. Raw measured rows are stored in `benchmark_table.csv`.
