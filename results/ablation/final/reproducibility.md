# Reproducibility

## Shared protocol

- Training budget: 30 epochs for every row
- Random seed: 42
- Dataset: `datasets/first_real_qr_10k` with the fixed train/validation/test split
- Dataset manifest SHA-256: `2c47950acb452137b3634bb3527cd48f3554b18f3416cd9c8cc7a1422df45c2f`
- Image size: 128 × 128
- Batch size: 16
- Optimizer: AdamW, learning rate 0.0002, weight decay 0.0001
- Schedule: two-epoch linear warmup followed by cosine decay to 0.000001 over 30 epochs
- Backend: lightweight state-space backend
- EMA: decay 0.995 except the named `without_ema` ablation
- Latency: batch size 1, 10 warmups, 100 timed iterations
- Base implementation commit: `b8fee1fcbc2da8fa39a78ce4b2a183e0c8bdfdbc`
- Finalization environment: Python 3.12.13, Torch 2.13.0+cpu, Windows-10-10.0.19045-SP0

## Artifact identities

| Variant | Epochs | Best epoch | Config SHA-256 | Checkpoint SHA-256 | Latency source |
| --- | ---: | ---: | --- | --- | --- |
| Full Model | 30 | 30 | `912d56d411ebc9742fba4ce7ec3fabae4072d8c51a932fa8f52fa25d52a5a470` | `a0e1abf2c3241a360dc7f5a66936aa838ad5d170a2c4a2cc44db1b98ee7099d6` | ablation_summary.json standardized evaluation (100 iterations) |
| Without Mamba | 30 | 30 | `4ccb669bb9f463bf4635ef7b6fa6e001e4e95eaa416b8b9358e29308bb2e7182` | `2e172078ae467debafdbe7a7b4d1ab41fe91086d4fb7c55f6237ed70539ff9f6` | ablation_summary.json standardized evaluation (100 iterations) |
| Without QRStructureLoss | 30 | 30 | `703dc6c53479288163baa9612ec2dadd04d2f4ce9e25a26bbbedfdb835cf6a05` | `f2f45dd5707c52270cf4c79c19167bac943b30a18fc4f5788b9cec25c05bcf99` | ablation_summary.json standardized evaluation (100 iterations) |
| Without DecodeConsistencyLoss | 30 | 30 | `4fc78dc1acd76309eb4543e77f4c34d03440968e8e4ddb0d337601bfc3321f4c` | `9bd0d1166ffccbe723d309c7558f4463656d63e9290cc27d8d5db0e8db9e9dcd` | refreshed benchmark_summary.csv (100 iterations) |
| Without EMA | 30 | 30 | `08b3f1fb1846e80d1ac9a976a80341390f895c63ecc1cb0918bf344d47903933` | `bff08d0f6883d167ab5ca34e6267e014d5001d4da435494332d59e279c0585a6` | refreshed benchmark_summary.csv (100 iterations) |
| Without Refinement Decoder | 30 | 30 | `7abaf99c86256c29e604ab7ebde051013d5f3a7e95c746eb1825aadfcdbe2ad3` | `f5dce71cf3321ace5b4819521bc16141ea537849bd88a3d0592d12ec9a50e679` | refreshed benchmark_summary.csv (100 iterations) |

All values were read from saved CSV, YAML, JSON, and checkpoint artifacts. This finalization script performs no model construction, inference, dataset generation, or training.
