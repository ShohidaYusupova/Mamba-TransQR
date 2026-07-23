# Mamba-TransQR

Mamba-TransQR is a Python toolkit for developing, training, evaluating, and
deploying Mamba-Transformer models for quality-aware research workflows.

> The implementation is in its initial scaffold stage; public APIs may change
> before the first stable release.

## Requirements

- Python 3.11 or later

## Installation

For a development install:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

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
