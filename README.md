# Mamba-TransQR

Mamba-TransQR is a Python toolkit for developing, training, evaluating, and
deploying Mamba-Transformer models for quality-aware research workflows.

Mamba-TransQR provides an end-to-end, configurable workflow for damaged QR
image restoration: data preparation, hybrid Mamba-Transformer modeling,
training, evaluation, and deployable inference.

## Requirements

- Python 3.11 or later

## Installation

For a development install:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For a standard runtime install use `python -m pip install -e .`. Full setup
instructions are in [docs/installation.md](docs/installation.md).

## Quick start

```python
from mambatransqr.models import ModelConfig, build_model

model = build_model(ModelConfig(image_size=256, patch_size=16))
```

The model accepts normalized `BCHW` image tensors and returns restored tensors
with the configured output channels.

## Development

```bash
ruff check .
black --check .
mypy src tests
pytest
```

## Dataset usage

Place input QR images in split directories (for example,
`data/train`, `data/validation`, and `data/test`). Image files are discovered
recursively. For paired restoration data, provide a matching `target_root`;
each target must have the same relative path as its source image.

```python
from mambatransqr.data import QRDataModule, QRDamageGenerator
from mambatransqr.data.transforms import Compose, Resize, ToTensor

train_transform = Compose([Resize((256, 256)), ToTensor()])
data = QRDataModule(
    train_dir="data/train",
    validation_dir="data/validation",
    test_dir="data/test",
    batch_size=32,
    train_transform=train_transform,
    evaluation_transform=train_transform,
)
data.setup()
batch = next(iter(data.train_dataloader()))

# Synthetic degradation can be used before the tensor transform.
damaged_image = QRDamageGenerator(seed=7)(your_pil_image)
```

See [configs/dataset.yaml](configs/dataset.yaml) and
[configs/augmentation.yaml](configs/augmentation.yaml) for reproducible
configuration examples.

## Training

Build the model and use the trainer with a paired image loader that yields
`{"image": tensor, "target": tensor}` batches. Checkpoints are written after
every epoch as `latest.pt`, and the best monitored validation result is saved as
`best.pt`.

```python
import torch

from mambatransqr.models import ModelConfig, build_model
from mambatransqr.training import (
    CSVLogger,
    LossManager,
    OptimizerFactory,
    Trainer,
    TrainerConfig,
)

model = build_model(ModelConfig(image_size=256, patch_size=16))
optimizer = OptimizerFactory.create(model.parameters())
trainer = Trainer(
    model,
    optimizer,
    LossManager({"mse": torch.nn.MSELoss()}),
    TrainerConfig(epochs=100, checkpoint_dir="checkpoints"),
    loggers=[CSVLogger("runs/metrics.csv")],
)
history = trainer.fit(train_loader, validation_loader)

# Resume from a prior checkpoint.
trainer.fit(train_loader, validation_loader, resume_from="checkpoints/latest.pt")
```

Use [configs/training.yaml](configs/training.yaml) as a starting configuration.
See [docs/training.md](docs/training.md) for the complete guide.

## Evaluation

Evaluate paired predictions with PSNR, SSIM, MSE, and MAE, then write portable
reports. Optional QR decoding supports ZBar (`pyzbar`) and ZXing
(`zxing-cpp`) when installed.

```python
from mambatransqr.evaluation import Evaluator, MetricsManager, ReportGenerator

evaluator = Evaluator(model, MetricsManager(("psnr", "ssim", "mse", "mae")))
metrics = evaluator.evaluate(test_loader)
ReportGenerator.evaluation_json(metrics, "reports/evaluation.json")
```

For inference performance, use `BenchmarkRunner(model).run(example_batch)`;
it reports throughput/FPS, inference latency, and CPU/GPU memory. See
[configs/evaluation.yaml](configs/evaluation.yaml) for all report settings.
See [docs/evaluation.md](docs/evaluation.md) for reports and benchmarking.

## Inference and export

Run a checkpoint over one image, a folder, or video frames. Image results save
the restored image plus JSON and CSV reports. CUDA AMP is enabled automatically
when available.

```bash
mambatransqr predict input.png outputs/ --checkpoint checkpoints/best.pt
mambatransqr predict-folder images/ outputs/ --checkpoint checkpoints/best.pt
mambatransqr predict-video input.mp4 outputs/restored.mp4 --checkpoint checkpoints/best.pt
mambatransqr export exports/model.pt --checkpoint checkpoints/best.pt --format torchscript
mambatransqr export exports/model.onnx --checkpoint checkpoints/best.pt --format onnx
```

For Python use, instantiate `Predictor` with a `ModelConfig` and pass it to
`InferencePipeline`. TorchScript and ONNX exports validate their generated
artifacts; ONNX supports dynamic batch axes by default. See
[configs/inference.yaml](configs/inference.yaml) for deployment settings.
See [docs/inference.md](docs/inference.md) for the full inference guide.

## Architecture

The model combines Mamba-inspired selective state-space mixing with Transformer
self-attention in hybrid fusion blocks. A patch-token encoder and image decoder
make the design suitable for image-to-image QR restoration. See
[docs/architecture.md](docs/architecture.md) for details.

## Project layout

```text
src/mambatransqr/  Package source code
tests/              Automated tests
docs/               Documentation sources
configs/            Reproducible experiment configuration
examples/           Runnable examples
scripts/            Developer and operational scripts
notebooks/          Exploratory notebooks
benchmarks/         Performance benchmarks
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please also follow the
[Code of Conduct](CODE_OF_CONDUCT.md) and review our [Security Policy](SECURITY.md).

## License

Distributed under the [MIT License](LICENSE).
