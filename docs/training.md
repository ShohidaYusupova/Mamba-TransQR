# Training

Create paired data loaders that yield `image` and `target` tensors, build a
model, and provide a `LossManager` to `Trainer`. The trainer selects an
available device, supports AMP, gradient accumulation/clipping, EMA, callbacks,
and checkpoint resume.

```python
import torch

from mambatransqr.models import ModelConfig, build_model
from mambatransqr.training import LossManager, OptimizerFactory, Trainer

model = build_model(ModelConfig(image_size=256, patch_size=16))
trainer = Trainer(
    model,
    OptimizerFactory.create(model.parameters()),
    LossManager({"mse": torch.nn.MSELoss()}),
)
trainer.fit(train_loader, validation_loader)
```

See `configs/training.yaml` for all training options.

`train_qr_restoration.yaml` defines the QR-specific weighted objective:
reconstruction, SSIM, edges, structural regions, and a differentiable decoding
surrogate. It does not differentiate through ZBar or ZXing. Recipe metadata
captures seeds, hashes, backend identity, dependencies, Git, and hardware.

## Synthetic QR pairs

Generate reproducible clean/damaged QR pairs with
`configs/dataset_generation.yaml`. The output stores damaged inputs under each
split's `damaged/` directory and matching clean targets under `clean/`; use the
split directories as `root` and `target_root` when constructing `QRDataset`.
Each metadata JSON records payload, QR settings, degradation provenance, and
decoder readability. Install the optional renderer with `.[qr-generation]`.
