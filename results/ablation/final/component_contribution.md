# Component contribution

Contribution is reported as the observed effect of removing one component from the Full Model. Positive removal deltas mean the ablated model scored higher; equivalently, the component's signed contribution (`Full − ablated`) is negative under this protocol.

## Mamba branch

Removing Mamba branch changed test PSNR by **+0.432774 dB** and test SSIM by **+0.004363** relative to the Full Model. Its signed PSNR contribution is therefore **-0.432774 dB**. Measured latency changed by **-67.424 ms** from the Full Model's 79.497 ms.

## QRStructureLoss

Removing QRStructureLoss changed test PSNR by **+0.756673 dB** and test SSIM by **+0.007641** relative to the Full Model. Its signed PSNR contribution is therefore **-0.756673 dB**. Measured latency changed by **-5.369 ms** from the Full Model's 79.497 ms.

## DecodeConsistencyLoss

Removing DecodeConsistencyLoss changed test PSNR by **+0.883365 dB** and test SSIM by **+0.008770** relative to the Full Model. Its signed PSNR contribution is therefore **-0.883365 dB**. Measured latency changed by **-4.222 ms** from the Full Model's 79.497 ms.

## EMA

Removing EMA changed test PSNR by **+0.660447 dB** and test SSIM by **+0.006793** relative to the Full Model. Its signed PSNR contribution is therefore **-0.660447 dB**. Measured latency changed by **-5.205 ms** from the Full Model's 79.497 ms.

## Refinement decoder

Removing Refinement decoder changed test PSNR by **+2.768564 dB** and test SSIM by **+0.023027** relative to the Full Model. Its signed PSNR contribution is therefore **-2.768564 dB**. Measured latency changed by **-8.669 ms** from the Full Model's 79.497 ms.

## Interpretation limits

All five removals improved PSNR and SSIM in this single-seed study, so none showed a positive isolated contribution to these two metrics at the tested settings. This does not establish that the components are generally harmful: the study is one-factor-at-a-time, interactions are unmeasured, and QR decode success or robustness objectives may not be fully represented by PSNR/SSIM. Loss and EMA variants have unchanged inference graphs, so their latency differences are measurement variation rather than architectural speedups. The Mamba bypass retains inert Mamba parameters in the instantiated model and should be interpreted as a branch-compute ablation, not a parameter-matched redesign.
