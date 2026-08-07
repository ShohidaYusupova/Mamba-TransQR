"""Finalize publication artifacts from completed Phase 4 ablation results."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
from pathlib import Path
from typing import Any

import matplotlib
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "ablation"
OUTPUT = SOURCE / "final"
VARIANTS = (
    "full_model",
    "without_mamba",
    "without_qr_structure_loss",
    "without_decode_consistency_loss",
    "without_ema",
    "without_refinement_decoder",
)
LABELS = {
    "full_model": "Full Model",
    "without_mamba": "Without Mamba",
    "without_qr_structure_loss": "Without QRStructureLoss",
    "without_decode_consistency_loss": "Without DecodeConsistencyLoss",
    "without_ema": "Without EMA",
    "without_refinement_decoder": "Without Refinement Decoder",
}
COMPONENTS = {
    "without_mamba": "Mamba branch",
    "without_qr_structure_loss": "QRStructureLoss",
    "without_decode_consistency_loss": "DecodeConsistencyLoss",
    "without_ema": "EMA",
    "without_refinement_decoder": "Refinement decoder",
}
BASE_GIT_COMMIT = "b8fee1fcbc2da8fa39a78ce4b2a183e0c8bdfdbc"


def _csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return next(csv.DictReader(stream))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(recipe: dict[str, Any]) -> str:
    controlled = json.loads(json.dumps(recipe))
    controlled.pop("output_dir", None)
    controlled.pop("latency", None)
    controlled.get("trainer", {}).pop("checkpoint_dir", None)
    payload = json.dumps(controlled, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def collect() -> list[dict[str, Any]]:
    """Collect exact saved metrics without invoking training or inference."""
    standardized = {
        row["variant"]: row
        for row in json.loads(
            (SOURCE / "ablation_summary.json").read_text(encoding="utf-8")
        )
    }
    rows = []
    for variant in VARIANTS:
        directory = SOURCE / variant
        benchmark = _csv_row(directory / "benchmark_summary.csv")
        training = _csv_rows(directory / "training_history.csv")
        validation = _csv_rows(directory / "validation_history.csv")
        best = max(validation, key=lambda row: float(row["val_psnr"]))
        recipe = yaml.safe_load(
            (directory / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        latency = (
            float(benchmark["inference_latency_ms"])
            if benchmark.get("resume_evaluation_runtime_seconds")
            else float(standardized[variant]["inference_latency_ms"])
        )
        rows.append(
            {
                "variant": variant,
                "label": LABELS[variant],
                "params": int(benchmark["parameter_count"]),
                "initial_loss": float(training[0]["loss"]),
                "final_loss": float(training[-1]["loss"]),
                "best_validation_psnr": float(best["val_psnr"]),
                "best_validation_ssim": float(best["val_ssim"]),
                "best_epoch": int(float(best["epoch"])),
                "test_psnr": float(benchmark["test_psnr"]),
                "test_ssim": float(benchmark["test_ssim"]),
                "latency_ms": latency,
                "latency_source": (
                    "refreshed benchmark_summary.csv (100 iterations)"
                    if benchmark.get("resume_evaluation_runtime_seconds")
                    else "ablation_summary.json standardized evaluation (100 iterations)"
                ),
                "config_hash": _config_hash(recipe),
                "checkpoint_path": f"checkpoints/ablation/{variant}/best.pt",
                "checkpoint_hash": _sha256(
                    ROOT / "checkpoints" / "ablation" / variant / "best.pt"
                ),
                "backend": recipe["model"]["mamba_backend"],
                "seed": int(recipe["trainer"]["seed"]),
                "epochs": len(training),
            }
        )
    full = rows[0]
    for row in rows:
        row["delta_psnr"] = row["test_psnr"] - full["test_psnr"]
        row["delta_ssim"] = row["test_ssim"] - full["test_ssim"]
        row["delta_latency_ms"] = row["latency_ms"] - full["latency_ms"]
    return rows


def write_tables(rows: list[dict[str, Any]]) -> None:
    """Write the requested CSV, Markdown, and LaTeX publication table."""
    fields = (
        "variant",
        "params",
        "test_psnr",
        "test_ssim",
        "latency_ms",
        "delta_psnr",
        "delta_ssim",
    )
    with (OUTPUT / "ablation_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    header = "| Variant | Params | Test PSNR | Test SSIM | Latency | ΔPSNR | ΔSSIM |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    body = "".join(
        f"| {row['label']} | {row['params']:,} | {row['test_psnr']:.6f} | {row['test_ssim']:.6f} | {row['latency_ms']:.3f} ms | {row['delta_psnr']:+.6f} | {row['delta_ssim']:+.6f} |\n"
        for row in rows
    )
    (OUTPUT / "ablation_table.md").write_text(header + body, encoding="utf-8")
    latex = "\n".join(
        f"{row['label'].replace(' ', '~')} & {row['params']:,} & {row['test_psnr']:.6f} & {row['test_ssim']:.6f} & {row['latency_ms']:.3f} & {row['delta_psnr']:+.6f} & {row['delta_ssim']:+.6f} \\\\"
        for row in rows
    )
    (OUTPUT / "ablation_table.tex").write_text(
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Variant & Params & Test PSNR & Test SSIM & Latency (ms) & "
        "$\\Delta$PSNR & $\\Delta$SSIM "
        + "\\\\\n"
        + "\\midrule\n"
        + latex
        + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )


def write_figure(rows: list[dict[str, Any]]) -> None:
    """Create a four-panel publication figure from exact saved values."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    short = [
        "Full",
        "w/o Mamba",
        "w/o Structure",
        "w/o Decode",
        "w/o EMA",
        "w/o Refinement",
    ]
    blue, green, orange, gray = "#2F6B9A", "#2A9D8F", "#D97706", "#6B7280"
    colors = [gray] + [blue] * (len(rows) - 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    psnr = [row["test_psnr"] for row in rows]
    bars = axes[0, 0].barh(short, psnr, color=colors)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title("Test PSNR comparison")
    axes[0, 0].set_xlabel("PSNR (dB)")
    axes[0, 0].bar_label(bars, labels=[f"{value:.4f}" for value in psnr], padding=3)
    axes[0, 0].set_xlim(0, max(psnr) * 1.08)

    ssim = [row["test_ssim"] for row in rows]
    bars = axes[0, 1].barh(short, ssim, color=[gray] + [green] * 5)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_title("Test SSIM comparison")
    axes[0, 1].set_xlabel("SSIM")
    axes[0, 1].set_xlim(min(ssim) - 0.004, max(ssim) + 0.006)
    axes[0, 1].bar_label(bars, labels=[f"{value:.6f}" for value in ssim], padding=3)

    components = [COMPONENTS[row["variant"]] for row in rows[1:]]
    contribution = [-row["delta_psnr"] for row in rows[1:]]
    bars = axes[1, 0].barh(components, contribution, color=orange)
    axes[1, 0].axvline(0, color="#374151", linewidth=0.8)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title("Component contribution to test PSNR")
    axes[1, 0].set_xlabel("Full − ablated PSNR (dB)")
    axes[1, 0].bar_label(
        bars, labels=[f"{value:+.4f}" for value in contribution], padding=3
    )
    axes[1, 0].set_xlim(min(contribution) * 1.22, 0.15)

    for index, row in enumerate(rows):
        marker = "D" if index == 0 else "o"
        color = gray if index == 0 else blue
        axes[1, 1].scatter(
            row["latency_ms"], row["test_psnr"], s=65, marker=marker, color=color
        )
        offsets = {
            0: (5, -14),
            1: (5, 5),
            2: (5, 10),
            3: (5, 5),
            4: (5, -18),
            5: (5, 5),
        }
        axes[1, 1].annotate(
            short[index],
            (row["latency_ms"], row["test_psnr"]),
            xytext=offsets[index],
            textcoords="offset points",
            fontsize=9,
        )
    axes[1, 1].set_title("Accuracy versus latency")
    axes[1, 1].set_xlabel("Inference latency (ms), batch size 1")
    axes[1, 1].set_ylabel("Test PSNR (dB)")
    axes[1, 1].set_xlim(8, 95)
    axes[1, 1].grid(alpha=0.2, linewidth=0.6)

    fig.suptitle("Phase 4 Mamba-TransQR ablation study", fontsize=15)
    fig.savefig(OUTPUT / "ablation_figure.png")
    fig.savefig(OUTPUT / "ablation_figure.pdf")
    plt.close(fig)


def write_component_contribution(rows: list[dict[str, Any]]) -> None:
    """Explain measured one-factor component contributions."""
    full = rows[0]
    by_variant = {row["variant"]: row for row in rows}
    lines = [
        "# Component contribution",
        "",
        "Contribution is reported as the observed effect of removing one component from the Full Model. Positive removal deltas mean the ablated model scored higher; equivalently, the component's signed contribution (`Full − ablated`) is negative under this protocol.",
        "",
    ]
    for variant, component in COMPONENTS.items():
        row = by_variant[variant]
        lines += [
            f"## {component}",
            "",
            f"Removing {component} changed test PSNR by **{row['delta_psnr']:+.6f} dB** and test SSIM by **{row['delta_ssim']:+.6f}** relative to the Full Model. Its signed PSNR contribution is therefore **{-row['delta_psnr']:+.6f} dB**. Measured latency changed by **{row['delta_latency_ms']:+.3f} ms** from the Full Model's {full['latency_ms']:.3f} ms.",
            "",
        ]
    lines += [
        "## Interpretation limits",
        "",
        "All five removals improved PSNR and SSIM in this single-seed study, so none showed a positive isolated contribution to these two metrics at the tested settings. This does not establish that the components are generally harmful: the study is one-factor-at-a-time, interactions are unmeasured, and QR decode success or robustness objectives may not be fully represented by PSNR/SSIM. Loss and EMA variants have unchanged inference graphs, so their latency differences are measurement variation rather than architectural speedups. The Mamba bypass retains inert Mamba parameters in the instantiated model and should be interpreted as a branch-compute ablation, not a parameter-matched redesign.",
    ]
    (OUTPUT / "component_contribution.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_statistical_summary(rows: list[dict[str, Any]]) -> None:
    """Write descriptive statistics without inventing inferential evidence."""
    psnr = [row["test_psnr"] for row in rows]
    ssim = [row["test_ssim"] for row in rows]
    latency = [row["latency_ms"] for row in rows]
    ranked = sorted(rows, key=lambda row: row["test_psnr"], reverse=True)
    lines = [
        "# Statistical summary",
        "",
        "## Study design",
        "",
        "Each configuration has one completed 30-epoch run with seed 42 on the same fixed train/validation/test split. Consequently, `n = 1` per configuration. No variance across seeds, confidence interval, hypothesis test, or statistical significance claim can be computed without fabrication.",
        "",
        "## Descriptive results",
        "",
        f"- Test PSNR across configurations: mean **{statistics.fmean(psnr):.6f} dB**, median **{statistics.median(psnr):.6f} dB**, range **{min(psnr):.6f}–{max(psnr):.6f} dB**.",
        f"- Test SSIM: mean **{statistics.fmean(ssim):.6f}**, median **{statistics.median(ssim):.6f}**, range **{min(ssim):.6f}–{max(ssim):.6f}**.",
        f"- Batch-one latency: mean **{statistics.fmean(latency):.3f} ms**, median **{statistics.median(latency):.3f} ms**, range **{min(latency):.3f}–{max(latency):.3f} ms**.",
        "",
        "## Quality ranking",
        "",
    ]
    lines.extend(
        f"{index}. {row['label']}: {row['test_psnr']:.6f} dB PSNR, {row['test_ssim']:.6f} SSIM"
        for index, row in enumerate(ranked, 1)
    )
    lines += [
        "",
        "The configurations are experimental conditions, not independent replicates; aggregate means and medians are descriptive summaries only.",
    ]
    (OUTPUT / "statistical_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_reproducibility(rows: list[dict[str, Any]]) -> None:
    """Record dataset, protocol, configuration, and checkpoint provenance."""
    manifest = ROOT / "datasets" / "first_real_qr_10k" / "dataset_manifest.csv"
    lines = [
        "# Reproducibility",
        "",
        "## Shared protocol",
        "",
        "- Training budget: 30 epochs for every row",
        "- Random seed: 42",
        "- Dataset: `datasets/first_real_qr_10k` with the fixed train/validation/test split",
        f"- Dataset manifest SHA-256: `{_sha256(manifest)}`",
        "- Image size: 128 × 128",
        "- Batch size: 16",
        "- Optimizer: AdamW, learning rate 0.0002, weight decay 0.0001",
        "- Schedule: two-epoch linear warmup followed by cosine decay to 0.000001 over 30 epochs",
        "- Backend: lightweight state-space backend",
        "- EMA: decay 0.995 except the named `without_ema` ablation",
        "- Latency: batch size 1, 10 warmups, 100 timed iterations",
        "- Base implementation commit: `b8fee1fcbc2da8fa39a78ce4b2a183e0c8bdfdbc`",
        f"- Finalization environment: Python {platform.python_version()}, Torch {torch.__version__}, {platform.platform()}",
        "",
        "## Artifact identities",
        "",
        "| Variant | Epochs | Best epoch | Config SHA-256 | Checkpoint SHA-256 | Latency source |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['label']} | {row['epochs']} | {row['best_epoch']} | `{row['config_hash']}` | `{row['checkpoint_hash']}` | {row['latency_source']} |"
        for row in rows
    )
    lines += [
        "",
        "All values were read from saved CSV, YAML, JSON, and checkpoint artifacts. This finalization script performs no model construction, inference, dataset generation, or training.",
    ]
    (OUTPUT / "reproducibility.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_unified_outputs(rows: list[dict[str, Any]]) -> None:
    """Write the unified six-row deliverables and verify scratch-run budgets."""
    if any(row["epochs"] != 30 for row in rows):
        raise ValueError("every unified ablation row must contain exactly 30 epochs")
    scratch_variants = (
        "without_decode_consistency_loss",
        "without_ema",
        "without_refinement_decoder",
    )
    for variant in scratch_variants:
        metadata = json.loads(
            (SOURCE / variant / "run_metadata.json").read_text(encoding="utf-8")
        )
        if metadata.get("resume", {}).get("resumed") is not False:
            raise ValueError(f"{variant} was not a verified scratch run")

    for suffix in (".md", ".csv", ".tex"):
        (SOURCE / f"ablation_table{suffix}").write_bytes(
            (OUTPUT / f"ablation_table{suffix}").read_bytes()
        )
    manifest_hash = _sha256(
        ROOT / "datasets" / "first_real_qr_10k" / "dataset_manifest.csv"
    )
    fields = (
        "variant",
        "training_epochs",
        "params",
        "test_psnr",
        "test_ssim",
        "latency_ms",
        "git_commit",
        "configuration_hash",
        "checkpoint_hash",
        "dataset_manifest_hash",
        "backend_identity",
    )
    with (SOURCE / "ablation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variant": row["variant"],
                    "training_epochs": row["epochs"],
                    "params": row["params"],
                    "test_psnr": row["test_psnr"],
                    "test_ssim": row["test_ssim"],
                    "latency_ms": row["latency_ms"],
                    "git_commit": BASE_GIT_COMMIT,
                    "configuration_hash": row["config_hash"],
                    "checkpoint_hash": row["checkpoint_hash"],
                    "dataset_manifest_hash": manifest_hash,
                    "backend_identity": row["backend"],
                }
            )

    identity_rows = "\n".join(
        f"| {row['label']} | {row['epochs']} | `{BASE_GIT_COMMIT}` | `{row['config_hash']}` | `{row['checkpoint_hash']}` | `{manifest_hash}` | `{row['backend']}` |"
        for row in rows
    )
    comparison = f"""# Unified 30-epoch ablation comparison

All six rows use the same fixed 7,000/1,500/1,500 split of the 10,000-pair dataset, seed 42, batch size 16, AdamW optimizer, two-epoch warmup followed by cosine decay, best-validation-PSNR checkpoint selection, held-out test evaluation, and batch-one latency with 10 warmups plus 100 timed iterations.

The three newly requested variants were trained from epoch 1 with `--resume never`; each `run_metadata.json` records `resumed: false`. No path under `results/ablation_pilot` or `checkpoints/ablation_pilot` was used. Pre-rerun artifacts were preserved separately under `results/ablation_pre_rerun` and `checkpoints/ablation_pre_rerun`.

{(OUTPUT / 'ablation_table.md').read_text(encoding='utf-8')}

## Reproducibility identities

The Git field is the recorded Phase 3 base implementation commit shared by the controlled recipes.

| Variant | Epochs | Git commit | Configuration SHA-256 | Checkpoint SHA-256 | Dataset manifest SHA-256 | Backend |
| --- | ---: | --- | --- | --- | --- | --- |
{identity_rows}

## Budget verification

Every row contains exactly 30 saved training-history epochs. The unified outputs exclude the separate 15-epoch pilot directory entirely.
"""
    (SOURCE / "comparison.md").write_text(comparison, encoding="utf-8")

    by_variant = {row["variant"]: row for row in rows}
    full = by_variant["full_model"]
    no_mamba = by_variant["without_mamba"]
    no_structure = by_variant["without_qr_structure_loss"]
    no_decode = by_variant["without_decode_consistency_loss"]
    no_ema = by_variant["without_ema"]
    no_refinement = by_variant["without_refinement_decoder"]
    report = f"""# Controlled 30-epoch ablation report

## Scope

The main publication analysis contains only the six controlled 30-epoch runs shown below. All use the same 10,000-pair dataset and fixed 7,000/1,500/1,500 split, seed 42, batch size 16, AdamW settings, two-epoch warmup plus cosine decay, best-validation-PSNR checkpoint selection, test protocol, and batch-one latency protocol with 10 warmups and 100 timed iterations. The three rerun variants started at epoch 1 with `--resume never`; their metadata records `resumed: false`.

The 15-epoch pilot artifacts remain archived under `results/ablation_pilot/` and `checkpoints/ablation_pilot/` for provenance. They are not read by this aggregation and do not appear in the publication table or conclusions.

{(OUTPUT / 'ablation_table.md').read_text(encoding='utf-8')}

## Findings

The Full Model was not the best-performing configuration under this controlled protocol: it recorded **{full['test_psnr']:.6f} dB** test PSNR and **{full['test_ssim']:.6f}** test SSIM, the lowest values among the six configurations.

Removing the refinement decoder produced the highest measured test PSNR and SSIM: **{no_refinement['test_psnr']:.6f} dB** and **{no_refinement['test_ssim']:.6f}**. These are changes of **{no_refinement['delta_psnr']:+.6f} dB** and **{no_refinement['delta_ssim']:+.6f}** relative to the Full Model.

Removing the lightweight state-space branch slightly improved test PSNR/SSIM to **{no_mamba['test_psnr']:.6f} dB**/**{no_mamba['test_ssim']:.6f}** while strongly reducing measured CPU latency from **{full['latency_ms']:.3f} ms** to **{no_mamba['latency_ms']:.3f} ms**. The bypass retains the branch parameters, so this is a branch-compute ablation rather than a parameter-matched redesign.

QRStructureLoss, DecodeConsistencyLoss, and EMA did not improve PSNR or SSIM under this controlled configuration. Removing them changed test PSNR by **{no_structure['delta_psnr']:+.6f} dB**, **{no_decode['delta_psnr']:+.6f} dB**, and **{no_ema['delta_psnr']:+.6f} dB**, respectively; the corresponding SSIM changes were **{no_structure['delta_ssim']:+.6f}**, **{no_decode['delta_ssim']:+.6f}**, and **{no_ema['delta_ssim']:+.6f}**.

Together, the uniformly positive removal deltas indicate interaction effects and possible over-regularization in the Full Model at this seed and training budget. This one-factor-at-a-time, single-seed study cannot isolate component interactions or establish statistical significance, and PSNR/SSIM do not fully represent QR payload recovery or robustness.

All runs used the project's `lightweight` state-space backend. These findings must not be generalized to the official `mamba-ssm` implementation.

## Reproducibility

Full configuration, checkpoint, dataset-manifest, base Git commit, backend, and 30-epoch budget identities for every row are recorded in `ablation_summary.csv` and `comparison.md`. No measured value was estimated or altered during aggregation.
"""
    (SOURCE / "ablation_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = collect()
    write_tables(rows)
    write_figure(rows)
    write_component_contribution(rows)
    write_statistical_summary(rows)
    write_reproducibility(rows)
    write_unified_outputs(rows)


if __name__ == "__main__":
    main()
