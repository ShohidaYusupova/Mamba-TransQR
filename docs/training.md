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
