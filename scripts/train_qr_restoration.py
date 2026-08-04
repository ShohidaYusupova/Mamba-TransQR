"""Run a reproducible paired QR restoration experiment from a YAML recipe."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from mambatransqr.data.loader import (
    create_evaluation_dataloader,
    create_train_dataloader,
)
from mambatransqr.evaluation import measure_latency
from mambatransqr.losses import MultiScaleRestorationLoss, QRLossWeights
from mambatransqr.models import ModelConfig, build_model
from mambatransqr.training import (
    EarlyStopping,
    OptimizerConfig,
    OptimizerFactory,
    QRRestorationTrainer,
    SchedulerConfig,
    SchedulerFactory,
    TensorBoardLogger,
    TrainerConfig,
)


class PairedQRDataset(Dataset[dict[str, Any]]):
    """Load paired images described by the generated QR metadata files."""

    def __init__(self, root: Path, split: str) -> None:
        self.root = root
        self.records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / split / "metadata").glob("*.json"))
        ]
        if not self.records:
            raise ValueError(f"no paired QR records found for split: {split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "image": _load_tensor(self.root / record["damaged_image_path"]),
            "target": _load_tensor(self.root / record["clean_image_path"]),
            "metadata": record,
        }


def _load_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1) / 255.0


def _mean(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    for row in rows:
        count += 1
        for key, value in row.items():
            totals[key] = totals.get(key, 0.0) + value
    if not count:
        raise ValueError("training loader yielded no batches")
    return {key: value / count for key, value in totals.items()}


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: str | Path) -> dict[str, float]:
    """Train, validate, checkpoint, and benchmark a QR restoration model."""
    recipe = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    dataset_root = Path(recipe["dataset"]["root"])
    output_root = Path(recipe["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(recipe["trainer"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seed = int(recipe["trainer"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    datasets = {
        split: PairedQRDataset(dataset_root, split)
        for split in ("train", "validation", "test")
    }
    batch_size = int(recipe["data_loader"]["batch_size"])
    workers = int(recipe["data_loader"].get("num_workers", 0))
    train_loader = create_train_dataloader(
        datasets["train"], batch_size=batch_size, num_workers=workers
    )
    validation_loader = create_evaluation_dataloader(
        datasets["validation"], batch_size=batch_size, num_workers=workers
    )
    test_loader = create_evaluation_dataloader(
        datasets["test"], batch_size=batch_size, num_workers=workers
    )

    model = build_model(ModelConfig(**recipe["model"]))
    optimizer = OptimizerFactory.create(
        model.parameters(), OptimizerConfig(**recipe["optimizer"])
    )
    scheduler = None
    if "scheduler" in recipe:
        scheduler = SchedulerFactory.create(
            optimizer, SchedulerConfig(**recipe["scheduler"])
        )
    objective = MultiScaleRestorationLoss(QRLossWeights(**recipe.get("loss", {})))
    early_stopping = None
    if "early_stopping" in recipe:
        early_stopping = EarlyStopping(**recipe["early_stopping"])
    trainer = QRRestorationTrainer(
        model,
        optimizer,
        objective,
        TrainerConfig(**recipe["trainer"]),
        scheduler=scheduler,
        loggers=[TensorBoardLogger(output_root / "tensorboard")],
    )

    started = time.perf_counter()
    training_rows: list[dict[str, float]] = []
    validation_rows: list[dict[str, float]] = []
    for epoch in range(1, trainer.config.epochs + 1):
        learning_rate = float(trainer.optimizer.param_groups[0]["lr"])
        train_metrics = _mean(trainer.train_batch(batch) for batch in train_loader)
        trainer.state.global_step += len(train_loader)
        if trainer.ema is not None:
            trainer.ema.apply_to(trainer.model)
        validation_metrics = trainer.validate_qr(validation_loader)
        if trainer.ema is not None:
            trainer.ema.restore(trainer.model)
        metrics = {
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **validation_metrics,
            "learning_rate": learning_rate,
        }
        trainer.state.epoch = epoch
        trainer.state.metrics = metrics
        is_best = trainer._is_best(metrics)
        if is_best:
            trainer.state.best_metric = metrics[trainer.config.monitor]
            trainer.state.best_epoch = epoch
        trainer.checkpoints.save(
            trainer.model,
            trainer.optimizer,
            trainer.state,
            scheduler=trainer.scheduler,
            ema_state=trainer.ema.state_dict() if trainer.ema is not None else None,
            deployable_model_state=(
                trainer.ema.averaged_model_state_dict(trainer.model)
                if trainer.ema is not None
                else None
            ),
            reproducibility={"recipe": str(config_path), "model": asdict(model.config)},
            is_best=is_best,
        )
        shutil.copy2(trainer.checkpoints.latest_path, checkpoint_dir / f"epoch_{epoch:03d}.pt")
        for logger in trainer.loggers:
            logger.log_metrics(metrics, trainer.state.global_step)
        training_rows.append({"epoch": float(epoch), **train_metrics})
        validation_rows.append(
            {"epoch": float(epoch), **validation_metrics, "learning_rate": learning_rate}
        )
        print(
            f"epoch={epoch}/{trainer.config.epochs} "
            f"loss={train_metrics['loss']:.6f} "
            f"val_psnr={validation_metrics['val_psnr']:.4f} "
            f"val_ssim={validation_metrics['val_ssim']:.6f}",
            flush=True,
        )
        trainer._step_scheduler(metrics)
        if early_stopping is not None:
            early_stopping.on_epoch_end(trainer, metrics)
        if trainer.state.stopped_early:
            break
    for logger in trainer.loggers:
        logger.close()

    trainer.checkpoints.load(
        trainer.checkpoints.best_path, trainer.model, map_location=trainer.device
    )
    test_metrics = trainer.validate_qr(test_loader)
    latency_batch = next(iter(test_loader))["image"][:1].to(trainer.device)
    with torch.inference_mode():
        latency = measure_latency(
            lambda: trainer.model(latency_batch),
            warmup=int(recipe.get("latency", {}).get("warmup", 5)),
            iterations=int(recipe.get("latency", {}).get("iterations", 20)),
            device=trainer.device,
        )
    runtime_seconds = time.perf_counter() - started
    _write_csv(output_root / "training_history.csv", training_rows)
    _write_csv(output_root / "validation_history.csv", validation_rows)
    benchmark = {
        "model": "mamba_transqr_lightweight",
        "test_psnr": test_metrics["val_psnr"],
        "test_ssim": test_metrics["val_ssim"],
        "runtime_seconds": runtime_seconds,
        "parameter_count": sum(
            parameter.numel() for parameter in trainer.model.parameters()
        ),
        "inference_latency_ms": latency.mean_ms,
    }
    with (output_root / "benchmark_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(benchmark))
        writer.writeheader()
        writer.writerow(benchmark)
    return {key: float(value) for key, value in benchmark.items() if key != "model"}


def main() -> None:
    parser = argparse.ArgumentParser(prog="train_qr_restoration")
    parser.add_argument("--config", type=Path, required=True)
    print(run(parser.parse_args().config))


if __name__ == "__main__":
    main()
