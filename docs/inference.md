# Inference

`Predictor` restores a single PIL image or image path. `InferencePipeline`
adds folder processing, saved restored/original images, plus JSON and CSV
reports. `VideoPredictor` requires optional OpenCV support.

```bash
mambatransqr predict input.png outputs --checkpoint checkpoints/best.pt
mambatransqr predict-folder samples outputs --checkpoint checkpoints/best.pt
mambatransqr export exports/model.pt --format torchscript
```

ONNX export supports dynamic batch axes by default and validates the generated
graph when the optional `onnx` package is installed.
