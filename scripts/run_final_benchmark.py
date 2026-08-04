"""Evaluate available final-benchmark entries without training or substitution."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib
import torch
import yaml
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mambatransqr.data.loader import create_evaluation_dataloader
from mambatransqr.data.paired_dataset import PairedQRDataset
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
UNAVAILABLE = {
    name: "No local implementation or checkpoint; substitution is prohibited."
    for name in ("SRCNN", "ESPCN", "VDSR", "EDSR", "SwinIR")
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate(
    model: nn.Module,
    loader: Any,
    trainer: QRRestorationTrainer,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    trainer.model = model.to(trainer.device)
    trainer.engine.model = trainer.model
    metrics = trainer.validate_qr(loader)
    batch = next(iter(loader))["image"][:1].to(trainer.device)
    with torch.inference_mode():
        latency = measure_latency(
            lambda: trainer.model(batch),
            warmup=warmup,
            iterations=iterations,
            device=trainer.device,
        )
    return {
        "psnr": metrics["val_psnr"],
        "ssim": metrics["val_ssim"],
        "decode_rate": metrics["val_decode_rate"],
        "latency_ms": latency.mean_ms,
    }


def _write_table(output: Path, rows: list[dict[str, Any]]) -> None:
    fields = ("model", "params", "psnr", "ssim", "decode_rate", "latency_ms")
    with (output / "benchmark_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)

    def value(row: dict[str, Any], key: str, digits: int) -> str:
        item = row.get(key)
        return "N/A" if item is None else f"{float(item):.{digits}f}"

    header = "| Model | Params | PSNR | SSIM | Decode Rate | Latency |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
    body = "".join(
        f"| {row['model']} | {row['params'] if row.get('params') is not None else 'N/A'} | {value(row, 'psnr', 6)} | {value(row, 'ssim', 6)} | {value(row, 'decode_rate', 6)} | {value(row, 'latency_ms', 3)}{' ms' if row.get('latency_ms') is not None else ''} |\n"
        for row in rows
    )
    (output / "benchmark_table.md").write_text(header + body, encoding="utf-8")
    latex_rows = "\n".join(
        f"{row['model'].replace('-', '--')} & {row['params'] if row.get('params') is not None else 'N/A'} & {value(row, 'psnr', 6)} & {value(row, 'ssim', 6)} & {value(row, 'decode_rate', 6)} & {value(row, 'latency_ms', 3)} \\\\"
        for row in rows
    )
    (output / "benchmark_table.tex").write_text(
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Model & Params & PSNR & SSIM & Decode Rate & Latency (ms) "
        + "\\\\\n"
        + "\\midrule\n"
        + latex_rows
        + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )


def _write_figure(output: Path, rows: list[dict[str, Any]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )
    measured = [row for row in rows if row.get("psnr") is not None]
    colors = ["#6B7280" if row["model"] == "Degraded Input" else "#2F6B9A" for row in measured]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for axis, key, title, unit, digits in (
        (axes[0, 0], "psnr", "PSNR comparison", "PSNR (dB)", 4),
        (axes[0, 1], "ssim", "SSIM comparison", "SSIM", 6),
        (axes[1, 0], "decode_rate", "Decode-rate comparison", "Threshold decode proxy rate", 6),
    ):
        values = [row[key] for row in measured]
        bars = axis.barh([row["model"] for row in measured], values, color=colors)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel(unit)
        axis.bar_label(bars, labels=[f"{value:.{digits}f}" for value in values], padding=3)
        axis.set_xlim(0, max(values) * 1.12)
        axis.text(
            0.99,
            0.02,
            "SRCNN, ESPCN, VDSR, EDSR, SwinIR: N/A",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#4B5563",
        )
    for row, color in zip(measured, colors, strict=True):
        axes[1, 1].scatter(row["latency_ms"], row["psnr"], s=75, color=color)
        axes[1, 1].annotate(
            row["model"],
            (row["latency_ms"], row["psnr"]),
            xytext=(6, 6),
            textcoords="offset points",
        )
    axes[1, 1].set_title("Accuracy versus latency")
    axes[1, 1].set_xlabel("Inference latency (ms), batch size 1")
    axes[1, 1].set_ylabel("PSNR (dB)")
    axes[1, 1].grid(alpha=0.2, linewidth=0.6)
    axes[1, 1].text(
        0.99,
        0.02,
        "Five baselines omitted: no measured artifacts",
        transform=axes[1, 1].transAxes,
        ha="right",
        fontsize=8,
        color="#4B5563",
    )
    fig.suptitle("Final Mamba-TransQR benchmark", fontsize=15)
    fig.savefig(output / "benchmark_figure.png")
    fig.savefig(output / "benchmark_figure.pdf")
    plt.close(fig)


def _write_report(output: Path, rows: list[dict[str, Any]]) -> None:
    degraded = next(row for row in rows if row["model"] == "Degraded Input")
    mamba = next(row for row in rows if row["model"] == "Mamba-TransQR")
    lines = [
        "# Final state-of-the-art benchmark report",
        "",
        "## Scope and availability",
        "",
        "The completed Phase 3 checkpoint was evaluated without retraining. Degraded Input is the identity baseline. SRCNN, ESPCN, VDSR, EDSR, and SwinIR are listed as unavailable because the repository contains neither implementations nor checkpoints for them; the baseline registry explicitly prohibits substituting another model. Their metrics remain `N/A` rather than fabricated.",
        "",
        "The reported decode rate is the Phase 3 threshold-based decode proxy (`mean(|output - 0.5| > 0.25)`), not payload recovery by ZBar/ZXing. No decoder backend is installed, so a true payload decode rate is unavailable for every model.",
        "",
        "## Strengths of Mamba-TransQR",
        "",
        f"Mamba-TransQR improves PSNR over Degraded Input by **{mamba['psnr'] - degraded['psnr']:+.6f} dB** and SSIM by **{mamba['ssim'] - degraded['ssim']:+.6f}** on the fixed 1,500-image test split. It uses the verified EMA-selected Phase 3 checkpoint and preserves the project's normalized image contract.",
        "",
        "## Weaknesses",
        "",
        f"The model has **{mamba['params']:,} parameters** and measured batch-one CPU latency of **{mamba['latency_ms']:.3f} ms**, compared with **{degraded['latency_ms']:.3f} ms** for the identity input. Its PSNR remains modest, and the absence of an installed QR decoder prevents reporting actual payload recovery.",
        "",
        "## Comparison with CNN baselines",
        "",
        "No defensible comparison with SRCNN, ESPCN, VDSR, or EDSR can be made from this workspace. Training new variants would introduce architecture/training choices not specified by the experiment, while importing generic super-resolution checkpoints would violate the same-dataset requirement. No ranking claim is made.",
        "",
        "## Comparison with the Transformer baseline",
        "",
        "SwinIR is likewise unavailable locally. Without a checkpoint trained under the same dataset and preprocessing, a numeric Transformer comparison would be misleading; its table cells remain `N/A`.",
        "",
        "## Computational trade-offs",
        "",
        "Mamba-TransQR trades approximately 1.48 million parameters and substantial CPU latency for improved reconstruction quality over the degraded input. The missing baseline artifacts prevent determining whether this trade-off is favorable relative to CNN or Transformer alternatives.",
    ]
    (output / "benchmark_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_reproducibility(
    output: Path, recipe: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    checkpoint = ROOT / recipe["checkpoint"]
    manifest = ROOT / recipe["dataset"]["root"] / "dataset_manifest.csv"
    missing = "\n".join(f"- {name}: {reason}" for name, reason in UNAVAILABLE.items())
    content = f"""# Reproducibility

## Evaluation protocol

- Dataset: `{recipe['dataset']['root']}`
- Split: `{recipe['dataset']['split']}` (1,500 fixed pairs)
- Dataset manifest SHA-256: `{_sha256(manifest)}`
- Checkpoint: `{recipe['checkpoint']}`
- Checkpoint SHA-256: `{_sha256(checkpoint)}`
- Checkpoint training: reused completed Phase 3 best checkpoint; no retraining
- Preprocessing: RGB conversion and normalization to `[0, 1]`
- Evaluation batch size: {recipe['data_loader']['batch_size']}
- Latency: batch size 1, {recipe['latency']['warmup']} warmups, {recipe['latency']['iterations']} timed iterations
- Metrics: Phase 3 batch-averaged PSNR, SSIM proxy, threshold decode proxy, and synchronized latency
- Runtime: Python {platform.python_version()}, Torch {torch.__version__}, {platform.platform()}

## Unavailable baselines

{missing}

No model was trained, no dataset was generated, and no unavailable value was estimated. Raw measured rows are stored in `benchmark_table.csv`.
"""
    (output / "reproducibility.md").write_text(content, encoding="utf-8")
    (output / "benchmark_raw_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


def main() -> None:
    recipe = yaml.safe_load(
        (ROOT / "configs" / "final_benchmark.yaml").read_text(encoding="utf-8")
    )
    phase3 = yaml.safe_load((ROOT / recipe["model_config"]).read_text(encoding="utf-8"))
    output = ROOT / recipe["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    dataset = PairedQRDataset(ROOT / recipe["dataset"]["root"], recipe["dataset"]["split"])
    loader = create_evaluation_dataloader(
        dataset,
        batch_size=int(recipe["data_loader"]["batch_size"]),
        num_workers=int(recipe["data_loader"]["num_workers"]),
    )
    model = build_model(ModelConfig(**phase3["model"]))
    optimizer = OptimizerFactory.create(
        model.parameters(), OptimizerConfig(**phase3["optimizer"])
    )
    trainer = QRRestorationTrainer(
        model,
        optimizer,
        MultiScaleRestorationLoss(QRLossWeights(**phase3["loss"])),
        TrainerConfig(**phase3["trainer"]),
    )
    trainer.checkpoints.load(
        ROOT / recipe["checkpoint"], trainer.model, map_location=trainer.device
    )
    latency = recipe["latency"]
    degraded_metrics = _evaluate(
        nn.Identity(),
        loader,
        trainer,
        warmup=int(latency["warmup"]),
        iterations=int(latency["iterations"]),
    )
    mamba_metrics = _evaluate(
        model,
        loader,
        trainer,
        warmup=int(latency["warmup"]),
        iterations=int(latency["iterations"]),
    )
    rows = []
    for name in recipe["models"]:
        if name == "Degraded Input":
            rows.append({"model": name, "params": 0, **degraded_metrics})
        elif name == "Mamba-TransQR":
            rows.append(
                {
                    "model": name,
                    "params": sum(parameter.numel() for parameter in model.parameters()),
                    **mamba_metrics,
                }
            )
        else:
            rows.append(
                {
                    "model": name,
                    "params": None,
                    "psnr": None,
                    "ssim": None,
                    "decode_rate": None,
                    "latency_ms": None,
                }
            )
    _write_table(output, rows)
    _write_figure(output, rows)
    _write_report(output, rows)
    _write_reproducibility(output, recipe, rows)


if __name__ == "__main__":
    main()
