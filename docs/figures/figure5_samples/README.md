# Figure 5 qualitative source data

Each sample directory contains the byte-identical clean and degraded images
from the fixed test split and a deterministic CPU inference output generated
from `checkpoints/phase3_pilot/best.pt` (SHA-256
`a0e1abf2c3241a360dc7f5a66936aa838ad5d170a2c4a2cc44db1b98ee7099d6`).
Inference used the model configuration in `configs/phase3_pilot.yaml`, RGB
conversion, resize to 128 x 128, `[0, 1]` input range, CPU, no mixed precision,
and the checkpoint's deployable `model` state (the selected EMA state).

The `manifest.csv` records every source/output hash and degradation operator.
To establish correspondence with the already-published composite, each
restored output was compared with the corresponding embedded PNG in
`figure5_experimental_results.svg`. SVG stores those rasters vertically flipped;
after applying the SVG transform and reverse nearest-neighbor scaling from 156
x 156 to 128 x 128, 98.284912% to 99.945068% of RGB pixels match exactly and
mean absolute intensity error is 0.008097 to 3.133586. The small boundary
difference is the expected non-invertibility of the 128-to-156 Matplotlib
rasterization. Visual inspection confirms the same QR modules and restoration.

No image was manually edited and no model was trained.
