# Statistical summary

## Study design

Each configuration has one completed 30-epoch run with seed 42 on the same fixed train/validation/test split. Consequently, `n = 1` per configuration. No variance across seeds, confidence interval, hypothesis test, or statistical significance claim can be computed without fabrication.

## Descriptive results

- Test PSNR across configurations: mean **17.369324 dB**, median **17.160913 dB**, range **16.452353–19.220918 dB**.
- Test SSIM: mean **0.958806**, median **0.957591**, range **0.950374–0.973401**.
- Batch-one latency: mean **64.349 ms**, median **74.211 ms**, range **12.073–79.497 ms**.

## Quality ranking

1. Without Refinement Decoder: 19.220918 dB PSNR, 0.973401 SSIM
2. Without DecodeConsistencyLoss: 17.335719 dB PSNR, 0.959144 SSIM
3. Without QRStructureLoss: 17.209027 dB PSNR, 0.958015 SSIM
4. Without EMA: 17.112800 dB PSNR, 0.957167 SSIM
5. Without Mamba: 16.885128 dB PSNR, 0.954737 SSIM
6. Full Model: 16.452353 dB PSNR, 0.950374 SSIM

The configurations are experimental conditions, not independent replicates; aggregate means and medians are descriptive summaries only.
