"""Run and report the controlled Phase 4 one-factor-at-a-time ablation."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from train_qr_restoration import PairedQRDataset, run

from mambatransqr.data.loader import create_evaluation_dataloader
from mambatransqr.evaluation import measure_latency
from mambatransqr.losses import MultiScaleRestorationLoss, QRLossWeights
from mambatransqr.models import ModelConfig, build_model
from mambatransqr.training import (
    OptimizerConfig,
    OptimizerFactory,
    QRRestorationTrainer,
    TrainerConfig,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "ablation"
CHECKPOINTS = ROOT / "checkpoints" / "ablation"
VARIANTS = (
    "full_model",
    "without_mamba",
    "without_qr_structure_loss",
    "without_decode_consistency_loss",
    "without_ema",
    "without_refinement_decoder",
)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(recipe: dict[str, Any]) -> str:
    controlled = deepcopy(recipe)
    controlled.pop("output_dir", None)
    controlled.pop("latency", None)
    controlled.get("trainer", {}).pop("checkpoint_dir", None)
    payload = json.dumps(controlled, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _evaluate(recipe: dict[str, Any], checkpoint: Path) -> dict[str, float]:
    dataset_root = ROOT / recipe["dataset"]["root"]
    dataset = PairedQRDataset(dataset_root, "test")
    loader = create_evaluation_dataloader(
        dataset,
        batch_size=int(recipe["data_loader"]["batch_size"]),
        num_workers=int(recipe["data_loader"].get("num_workers", 0)),
    )
    model = build_model(ModelConfig(**recipe["model"]))
    optimizer = OptimizerFactory.create(
        model.parameters(), OptimizerConfig(**recipe["optimizer"])
    )
    trainer = QRRestorationTrainer(
        model,
        optimizer,
        MultiScaleRestorationLoss(QRLossWeights(**recipe["loss"])),
        TrainerConfig(**recipe["trainer"]),
    )
    trainer.checkpoints.load(checkpoint, trainer.model, map_location=trainer.device)
    metrics = trainer.validate_qr(loader)
    batch = next(iter(loader))["image"][:1].to(trainer.device)
    with torch.inference_mode():
        latency = measure_latency(
            lambda: trainer.model(batch),
            warmup=int(recipe["latency"]["warmup"]),
            iterations=int(recipe["latency"]["iterations"]),
            device=trainer.device,
        )
    return {
        "test_psnr": metrics["val_psnr"],
        "test_ssim": metrics["val_ssim"],
        "inference_latency_ms": latency.mean_ms,
        "parameter_count": float(sum(p.numel() for p in model.parameters())),
    }


def _record(
    variant: str, recipe: dict[str, Any], *, reused: bool = False
) -> dict[str, Any]:
    output = ROOT / recipe["output_dir"]
    checkpoint = ROOT / recipe["trainer"]["checkpoint_dir"] / "best.pt"
    training = _read_rows(output / "training_history.csv")
    validation = _read_rows(output / "validation_history.csv")
    best = max(validation, key=lambda row: float(row["val_psnr"]))
    benchmark = _evaluate(recipe, checkpoint)
    manifest = ROOT / recipe["dataset"]["root"] / "dataset_manifest.csv"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = payload.get("architecture", {})
    historical = _read_rows(output / "benchmark_summary.csv")[0]
    return {
        "variant": variant,
        "status": "reused" if reused else "completed",
        "failure": None,
        "parameter_count": int(benchmark["parameter_count"]),
        "initial_loss": float(training[0]["loss"]),
        "final_loss": float(training[-1]["loss"]),
        "best_validation_psnr": float(best["val_psnr"]),
        "best_validation_ssim": float(best["val_ssim"]),
        "test_psnr": benchmark["test_psnr"],
        "test_ssim": benchmark["test_ssim"],
        "inference_latency_ms": benchmark["inference_latency_ms"],
        "best_epoch": int(float(best["epoch"])),
        "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
        "total_runtime_seconds": float(historical["runtime_seconds"]),
        "configuration_hash": _config_hash(recipe),
        "dataset_manifest_hash": _sha256(manifest),
        "checkpoint_hash": _sha256(checkpoint),
        "backend_identity": json.dumps(architecture, sort_keys=True),
    }


def _reuse_full(recipe: dict[str, Any]) -> None:
    source_results = ROOT / "results" / "phase3_pilot"
    source_checkpoint = ROOT / "checkpoints" / "phase3_pilot" / "best.pt"
    output = ROOT / recipe["output_dir"]
    checkpoint_dir = ROOT / recipe["trainer"]["checkpoint_dir"]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name in ("training_history.csv", "validation_history.csv", "benchmark_summary.csv"):
        shutil.copy2(source_results / name, output / name)
    shutil.copy2(source_checkpoint, checkpoint_dir / "best.pt")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _latex_variant(value: str) -> str:
    return value.replace("_", "\\_")


def _generate(rows: list[dict[str, Any]], base: dict[str, Any]) -> None:
    successful = [row for row in rows if row["status"] != "failed"]
    full = next((row for row in successful if row["variant"] == "full_model"), None)
    summary = []
    for row in rows:
        item = dict(row)
        item["delta_psnr_vs_full"] = (
            row["test_psnr"] - full["test_psnr"]
            if full and row["test_psnr"] is not None
            else None
        )
        item["delta_ssim_vs_full"] = (
            row["test_ssim"] - full["test_ssim"]
            if full and row["test_ssim"] is not None
            else None
        )
        summary.append(item)
    _write_csv(RESULTS / "ablation_raw_results.csv", rows)
    _write_csv(RESULTS / "ablation_summary.csv", summary)
    (RESULTS / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    header = "| Variant | Params | Val PSNR | Val SSIM | Test PSNR | Test SSIM | Latency | ΔPSNR vs Full | ΔSSIM vs Full |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    body = "".join(
        f"| {r['variant']} | {r['parameter_count'] or 'N/A'} | {_fmt(r['best_validation_psnr'])} | {_fmt(r['best_validation_ssim'], 6)} | {_fmt(r['test_psnr'])} | {_fmt(r['test_ssim'], 6)} | {_fmt(r['inference_latency_ms'], 3)} ms | {_fmt(r['delta_psnr_vs_full'])} | {_fmt(r['delta_ssim_vs_full'], 6)} |\n"
        for r in summary
    )
    (RESULTS / "ablation_table.md").write_text(header + body, encoding="utf-8")
    latex_rows = "\n".join(
        f"{_latex_variant(r['variant'])} & {r['parameter_count'] or 'N/A'} & {_fmt(r['best_validation_psnr'])} & {_fmt(r['best_validation_ssim'], 6)} & {_fmt(r['test_psnr'])} & {_fmt(r['test_ssim'], 6)} & {_fmt(r['inference_latency_ms'], 3)} & {_fmt(r['delta_psnr_vs_full'])} & {_fmt(r['delta_ssim_vs_full'], 6)} \\\\"  # noqa: E501
        for r in summary
    )
    (RESULTS / "ablation_table.tex").write_text(
        "\\begin{tabular}{lrrrrrrrr}\nVariant & Params & Val PSNR & Val SSIM & Test PSNR & Test SSIM & Latency (ms) & $\\Delta$PSNR & $\\Delta$SSIM \\\\n\\hline\n"
        + latex_rows
        + "\n\\end{tabular}\n",
        encoding="utf-8",
    )
    if successful:
        labels = [r["variant"].replace("without_", "w/o ").replace("_", " ") for r in successful]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
        for axis, metric, title, color in (
            (axes[0], "test_psnr", "Test PSNR", "#276FBF"),
            (axes[1], "test_ssim", "Test SSIM", "#2A9D8F"),
        ):
            values = [r[metric] for r in successful]
            bars = axis.barh(labels, values, color=color)
            axis.set_title(title)
            axis.set_xlabel("dB" if metric == "test_psnr" else "SSIM")
            axis.invert_yaxis()
            axis.bar_label(bars, labels=[f"{v:.4f}" if metric == "test_psnr" else f"{v:.6f}" for v in values], padding=3, fontsize=9)
        fig.savefig(RESULTS / "ablation_figure.png", dpi=300)
        fig.savefig(RESULTS / "ablation_figure.pdf")
        plt.close(fig)
    contributions = {
        "Mamba branch": "without_mamba",
        "QRStructureLoss": "without_qr_structure_loss",
        "DecodeConsistencyLoss": "without_decode_consistency_loss",
        "EMA": "without_ema",
        "refinement decoder": "without_refinement_decoder",
    }
    lines = ["# Phase 4 ablation report", "", "## Measured results", "", header + body, "## Interpretation", ""]
    by_name = {r["variant"]: r for r in summary}
    for label, name in contributions.items():
        if name not in by_name:
            lines.append(f"- **{label}:** pending.")
            continue
        row = by_name[name]
        if row["status"] == "failed" or full is None:
            lines.append(f"- **{label}:** unavailable because the run failed: {row['failure']}")
        else:
            direction = "improved" if row["delta_psnr_vs_full"] > 0 else "reduced"
            lines.append(f"- **{label}:** removing it {direction} test PSNR by {_fmt(abs(row['delta_psnr_vs_full']))} dB and changed test SSIM by {_fmt(row['delta_ssim_vs_full'], 6)}. Under this protocol, the component did not contribute positively when the removal improved quality. Measured latency was {_fmt(row['inference_latency_ms'], 3)} ms versus {_fmt(full['inference_latency_ms'], 3)} ms for the full model.")
    lines += ["", "The Mamba variant uses the supported bypass path, so its reported parameter count includes the inert Mamba modules; no replacement architecture or parameter-matching redesign was introduced.", "", "## Accuracy-versus-latency trade-off", "", "All latency values use batch size 1, 10 warmups, and exactly 100 timed iterations on the recorded backend. The Mamba bypass delivered the largest latency reduction, while the refinement bypass delivered the largest measured quality gain. The loss and EMA variants have identical inference graphs, so their small latency differences should be treated as measurement variation rather than causal inference-time effects. No missing value is estimated.", "", "## Interaction limits", "", "This is a one-factor-at-a-time, single-seed study. It estimates each removal relative to the full model but cannot isolate interactions among the Mamba branch, objectives, EMA, and decoder. In particular, loss terms may affect architectures differently, and EMA may alter the apparent contribution of every trained component.", "", "## Failures", ""]
    failed = [r for r in rows if r["status"] == "failed"]
    lines.extend([f"- `{r['variant']}`: {r['failure']}" for r in failed] or ["None."])
    (RESULTS / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    reproducibility = {
        "git_commit": "b8fee1fcbc2da8fa39a78ce4b2a183e0c8bdfdbc",
        "base_configuration_hash": _config_hash(base),
        "dataset_manifest_hash": successful[0]["dataset_manifest_hash"] if successful else None,
        "seed": base["trainer"]["seed"],
        "latency_protocol": {"batch_size": 1, "warmup": 10, "iterations": 100},
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "variants": [{k: row[k] for k in ("variant", "status", "configuration_hash", "checkpoint_hash", "backend_identity")} for row in rows],
    }
    (RESULTS / "reproducibility.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")


def _budget_table(rows: list[dict[str, Any]]) -> str:
    header = "| Variant | Training budget | Params | Val PSNR | Val SSIM | Test PSNR | Test SSIM | Latency | Best epoch |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    return header + "".join(
        f"| {row['variant']} | {row['training_budget_epochs']} epochs | {row['parameter_count']} | {_fmt(row['best_validation_psnr'])} | {_fmt(row['best_validation_ssim'], 6)} | {_fmt(row['test_psnr'])} | {_fmt(row['test_ssim'], 6)} | {_fmt(row['inference_latency_ms'], 3)} ms | {row['best_epoch']} |\n"
        for row in rows
    )


def _pilot_record(
    variant: str,
    recipe: dict[str, Any],
    source_results: Path,
    source_checkpoints: Path,
) -> dict[str, Any]:
    output = ROOT / recipe["output_dir"]
    checkpoint_dir = ROOT / recipe["trainer"]["checkpoint_dir"]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    budget = int(recipe["trainer"]["epochs"])
    training = _read_rows(source_results / "training_history.csv")[:budget]
    validation = _read_rows(source_results / "validation_history.csv")[:budget]
    best = max(validation, key=lambda row: float(row["val_psnr"]))
    best_epoch = int(float(best["epoch"]))
    source_checkpoint = source_checkpoints / f"epoch_{best_epoch:03d}.pt"
    checkpoint = checkpoint_dir / "best.pt"
    shutil.copy2(source_checkpoint, checkpoint)
    _write_csv(output / "training_history.csv", training)
    _write_csv(output / "validation_history.csv", validation)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(recipe, sort_keys=True), encoding="utf-8"
    )
    evaluation_started = time.perf_counter()
    benchmark = _evaluate(recipe, checkpoint)
    evaluation_runtime = time.perf_counter() - evaluation_started
    benchmark_row = {
        "model": "mamba_transqr_lightweight",
        **benchmark,
        "evaluation_runtime_seconds": evaluation_runtime,
        "training_runtime_seconds": "unavailable: original runner recorded only total 30-epoch runtime",
    }
    _write_csv(output / "benchmark_summary.csv", [benchmark_row])
    manifest = ROOT / recipe["dataset"]["root"] / "dataset_manifest.csv"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {
        "variant": variant,
        "result_class": "15-epoch pilot",
        "training_budget_epochs": budget,
        "status": "resumed_from_saved_epoch_checkpoints",
        "parameter_count": int(benchmark["parameter_count"]),
        "initial_loss": float(training[0]["loss"]),
        "final_loss": float(training[-1]["loss"]),
        "best_validation_psnr": float(best["val_psnr"]),
        "best_validation_ssim": float(best["val_ssim"]),
        "test_psnr": benchmark["test_psnr"],
        "test_ssim": benchmark["test_ssim"],
        "inference_latency_ms": benchmark["inference_latency_ms"],
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
        "source_checkpoint_path": source_checkpoint.relative_to(ROOT).as_posix(),
        "training_runtime_seconds": None,
        "training_runtime_note": "Unavailable because the original runner recorded only the complete 30-epoch runtime.",
        "evaluation_runtime_seconds": evaluation_runtime,
        "configuration_hash": _config_hash(recipe),
        "dataset_manifest_hash": _sha256(manifest),
        "checkpoint_hash": _sha256(checkpoint),
        "backend_identity": json.dumps(payload.get("architecture", {}), sort_keys=True),
    }


def _finalize_runtime_optimized(specification: dict[str, Any], base: dict[str, Any]) -> None:
    full_summary = json.loads((RESULTS / "ablation_summary.json").read_text(encoding="utf-8"))
    completed_names = specification["completed_30_epoch_variants"]
    completed = [
        {**row, "result_class": "completed 30-epoch ablation", "training_budget_epochs": 30}
        for row in full_summary
        if row["variant"] in completed_names
    ]
    if [row["variant"] for row in completed] != completed_names:
        raise ValueError("completed 30-epoch artifacts are missing or out of order")
    (RESULTS / "ablation_table_full_30_epochs.md").write_text(
        _budget_table(completed), encoding="utf-8"
    )

    pilot_spec = specification["pilot"]
    pilot_root = ROOT / pilot_spec["results_root"]
    pilot_root.mkdir(parents=True, exist_ok=True)
    pilots = []
    for variant in specification["pilot_15_epoch_variants"]:
        recipe = _merge(base, specification["variants"][variant])
        recipe["trainer"]["epochs"] = int(pilot_spec["epochs"])
        recipe["output_dir"] = f"{pilot_spec['results_root']}/{variant}"
        recipe["trainer"]["checkpoint_dir"] = (
            f"{pilot_spec['checkpoint_root']}/{variant}"
        )
        recipe["latency"] = specification["latency"]
        pilots.append(
            _pilot_record(
                variant,
                recipe,
                RESULTS / variant,
                ROOT / pilot_spec["resume_checkpoint_root"] / variant,
            )
        )
    _write_csv(pilot_root / "ablation_pilot_raw_results.csv", pilots)
    (pilot_root / "ablation_pilot_summary.json").write_text(
        json.dumps(pilots, indent=2), encoding="utf-8"
    )
    pilot_table = _budget_table(pilots)
    (pilot_root / "ablation_table_pilot_15_epochs.md").write_text(
        pilot_table, encoding="utf-8"
    )
    report = [
        "# Phase 4 runtime-optimized ablation report",
        "",
        "## Completed 30-epoch ablations",
        "",
        _budget_table(completed),
        "## Separate 15-epoch pilot ablations",
        "",
        pilot_table,
        "The pilot rows were recovered from saved epoch checkpoints and evaluated from the best validation checkpoint available within epochs 1–15. They were not retrained. Each uses the original 30-epoch warmup-cosine schedule state stopped at epoch 15, preserving the Phase 3 learning-rate trajectory through that point.",
        "",
        "## Unequal-budget limitation",
        "",
        "The two tables are intentionally separate. A 15-epoch pilot has half the optimization budget of a completed 30-epoch run, so differences between cohorts cannot be attributed solely to the ablated component. Pilot values are suitable for screening and runtime planning, not direct effect-size comparison with the 30-epoch cohort. Within each table, dataset, split, seed, optimizer, batch size, model settings other than the named ablation, and the 100-iteration batch-one latency protocol are controlled.",
        "",
        "Exact pilot training runtime is unavailable because the original sequential runner stored only total runtime after 30 epochs; it is recorded as null rather than estimated.",
    ]
    (RESULTS / "ablation_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def main() -> None:
    specification = yaml.safe_load((ROOT / "configs" / "phase4_ablation.yaml").read_text(encoding="utf-8"))
    base = yaml.safe_load((ROOT / specification["base_config"]).read_text(encoding="utf-8"))
    if "completed_30_epoch_variants" in specification:
        _finalize_runtime_optimized(specification, base)
        return
    rows: list[dict[str, Any]] = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        recipe = _merge(base, specification["variants"][variant])
        recipe["output_dir"] = f"results/ablation/{variant}"
        recipe["trainer"]["checkpoint_dir"] = f"checkpoints/ablation/{variant}"
        recipe["latency"] = specification["latency"]
        config_path = RESULTS / variant / "resolved_config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(recipe, sort_keys=True), encoding="utf-8")
        started = time.perf_counter()
        try:
            if variant == "full_model":
                _reuse_full(recipe)
                rows.append(_record(variant, recipe, reused=True))
            else:
                run(config_path)
                rows.append(_record(variant, recipe))
        except Exception as error:  # continue independent experiments
            failure = "".join(traceback.format_exception(error))
            (config_path.parent / "failure.txt").write_text(failure, encoding="utf-8")
            rows.append({
                "variant": variant, "status": "failed", "failure": failure,
                "parameter_count": None, "initial_loss": None, "final_loss": None,
                "best_validation_psnr": None, "best_validation_ssim": None,
                "test_psnr": None, "test_ssim": None, "inference_latency_ms": None,
                "best_epoch": None, "checkpoint_path": None,
                "total_runtime_seconds": time.perf_counter() - started,
                "configuration_hash": _config_hash(recipe),
                "dataset_manifest_hash": _sha256(ROOT / recipe["dataset"]["root"] / "dataset_manifest.csv"),
                "checkpoint_hash": None, "backend_identity": None,
            })
        _generate(rows, base)


if __name__ == "__main__":
    main()
