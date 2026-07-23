# Core modules

`mambatransqr.losses` provides composable reconstruction, perceptual, edge,
SSIM, Charbonnier, Dice, focal, and weighted combined losses. The metrics
package supplies image, binary, classification, and decoder metrics.

Use `Experiment(name, config=...)` to create a timestamped run directory with
`config.json`, `metrics.csv`, and a JSON summary. Plot training values with
`plot_training_curves`; Matplotlib and TensorBoard are optional dependencies.
