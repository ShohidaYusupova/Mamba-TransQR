"""Run a reproducible paired QR restoration experiment from a YAML recipe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from mambatransqr.data.loader import (
    create_evaluation_dataloader,
    create_train_dataloader,
)
from mambatransqr.data.paired_dataset import PairedQRDataset
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
    capture_rng_state,
    resolve_resume_checkpoint,
    validate_resume_identity,
)


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


def _read_csv(path: Path, completed_epochs: int) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
            if int(float(row["epoch"])) <= completed_epochs
        ]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration_hash(recipe: dict[str, Any]) -> str:
    controlled = json.loads(json.dumps(recipe))
    controlled.pop("output_dir", None)
    controlled.pop("latency", None)
    controlled.get("trainer", {}).pop("checkpoint_dir", None)
    payload = json.dumps(controlled, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _resume_identity(
    recipe: dict[str, Any], model: torch.nn.Module, variant_name: str
) -> dict[str, Any]:
    manifest = Path(recipe["dataset"]["root"]) / "dataset_manifest.csv"
    architecture = model.architecture_identity()
    return {
        "variant_name": variant_name,
        "configuration_hash": _configuration_hash(recipe),
        "dataset_manifest_hash": _hash_file(manifest),
        "random_seed": int(recipe["trainer"]["seed"]),
        "backend_identity": architecture,
        "model_architecture": asdict(model.config),
    }


def run(
    config_path: str | Path,
    *,
    resume: str | Path = "never",
    variant_name: str | None = None,
) -> dict[str, float]:
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

    name = variant_name or str(recipe.get("variant_name", Path(config_path).stem))
    identity = _resume_identity(recipe, model, name)
    resume_path = resolve_resume_checkpoint(resume, checkpoint_dir)
    resume_events: list[dict[str, Any]] = []
    if resume_path is not None:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume_identity(payload, identity)
        trainer.state, ema_state = trainer.checkpoints.load(
            resume_path,
            trainer.model,
            trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.engine.amp,
            early_stopping=early_stopping,
            restore_rng=True,
            map_location=trainer.device,
        )
        if trainer.ema is not None:
            if ema_state is None:
                raise ValueError("EMA-enabled run checkpoint has no EMA state")
            trainer.ema.load_state_dict(ema_state)
        resume_events = list(payload.get("resume_events", []))
        resume_events.append(
            {
                "resumed": True,
                "checkpoint_path": str(resume_path),
                "resumed_epoch": trainer.state.epoch,
                "checkpoint_hash": _hash_file(resume_path),
                "resume_timestamp": datetime.now(datetime.UTC).isoformat(),
            }
        )

    metadata = {
        "variant_name": name,
        "resume": resume_events[-1] if resume_events else {"resumed": False},
        "resume_events": resume_events,
        "resume_identity": identity,
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    started = time.perf_counter()
    training_rows = _read_csv(
        output_root / "training_history.csv", trainer.state.epoch
    )
    validation_rows = _read_csv(
        output_root / "validation_history.csv", trainer.state.epoch
    )
    next_epoch = (
        trainer.config.epochs + 1
        if trainer.state.stopped_early
        else trainer.state.epoch + 1
    )
    for epoch in range(next_epoch, trainer.config.epochs + 1):
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
        _write_csv(output_root / "training_history.csv", training_rows)
        _write_csv(output_root / "validation_history.csv", validation_rows)
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
            scaler_state=trainer.engine.amp.state_dict(),
            early_stopping_state=(
                early_stopping.state_dict() if early_stopping is not None else None
            ),
            rng_state=capture_rng_state(),
            resume_identity=identity,
            resume_events=resume_events,
            reproducibility={"recipe": str(config_path), "model": asdict(model.config)},
            is_best=is_best,
        )
        shutil.copy2(
            trainer.checkpoints.latest_path, checkpoint_dir / f"epoch_{epoch:03d}.pt"
        )
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
    parser.add_argument("--resume", default="never", metavar="auto|never|PATH")
    parser.add_argument("--variant-name")
    args = parser.parse_args()
    print(run(args.config, resume=args.resume, variant_name=args.variant_name))


if __name__ == "__main__":
    main()
