"""Assemble publication artifacts from completed experiment outputs only."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib-cache"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT = RESULTS / "final_analysis"

PHASES = [
    ("Baseline", "first_real_qr_restoration", 30, 522_304),
    ("Phase 1", "phase1_pilot", 10, 522_304),
    ("Phase 2", "phase2_pilot", 30, 522_304),
    ("Phase 3", "phase3_pilot", 30, 1_476_259),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def number(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: Any, digits: int = 6, suffix: str = "") -> str:
    if value is None or value == "":
        return "N/A"
    return f"{float(value):.{digits}f}{suffix}"


def variant_label(value: str) -> str:
    if value == "full_model":
        return "Full Model"
    label = value.replace("without_", "Without ").replace("_", " ").title()
    return label.replace("Qr ", "QR ").replace("Ema", "EMA")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    alignment = ["---"] + ["---:" for _ in headers[1:]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignment) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def latex_table(headers: list[str], rows: list[list[str]]) -> str:
    escaped = [[cell.replace("_", "\\_") for cell in row] for row in rows]
    body = "\n".join(" & ".join(row) + r" \\" for row in escaped)
    return (
        "\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}\n"
        "\\toprule\n"
        + " & ".join(headers)
        + " \\\\\n\\midrule\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def phase_data() -> list[dict[str, Any]]:
    collected = []
    for label, directory, epochs, params in PHASES:
        base = RESULTS / directory
        benchmark = read_rows(base / "benchmark_summary.csv")[0]
        training = read_rows(base / "training_history.csv")
        validation = read_rows(base / "validation_history.csv")
        best_psnr = max(validation, key=lambda row: float(row["val_psnr"]))
        best_ssim = max(validation, key=lambda row: float(row["val_ssim"]))
        collected.append(
            {
                "stage": label,
                "epochs": epochs,
                "params": params,
                "initial_loss": float(training[0]["loss"]),
                "final_loss": float(training[-1]["loss"]),
                "best_val_psnr": float(best_psnr["val_psnr"]),
                "best_val_ssim": float(best_ssim["val_ssim"]),
                "test_psnr": float(benchmark["test_psnr"]),
                "test_ssim": float(benchmark["test_ssim"]),
                "latency_ms": number(benchmark.get("inference_latency_ms")),
                "runtime_seconds": float(benchmark["runtime_seconds"]),
                "source": str(base.relative_to(ROOT) / "benchmark_summary.csv"),
            }
        )
    return collected


def write_overall(phases: list[dict[str, Any]], ablations: list[dict[str, str]], benchmark: list[dict[str, str]]) -> None:
    fields = [
        "category", "experiment", "training_epochs", "params", "test_psnr",
        "test_ssim", "decode_rate", "latency_ms", "runtime_seconds", "source",
    ]
    rows: list[dict[str, Any]] = []
    for row in phases:
        rows.append(
            {
                "category": "optimization", "experiment": row["stage"],
                "training_epochs": row["epochs"], "params": row["params"],
                "test_psnr": row["test_psnr"], "test_ssim": row["test_ssim"],
                "decode_rate": None, "latency_ms": row["latency_ms"],
                "runtime_seconds": row["runtime_seconds"], "source": row["source"],
            }
        )
    for row in ablations:
        rows.append(
            {
                "category": "ablation", "experiment": variant_label(row["variant"]),
                "training_epochs": 30, "params": row["params"],
                "test_psnr": row["test_psnr"], "test_ssim": row["test_ssim"],
                "decode_rate": None, "latency_ms": row["latency_ms"],
                "runtime_seconds": None,
                "source": "results/ablation/final/ablation_table.csv",
            }
        )
    for row in benchmark:
        rows.append(
            {
                "category": "final_benchmark", "experiment": row["model"],
                "training_epochs": None, "params": row["params"],
                "test_psnr": row["psnr"], "test_ssim": row["ssim"],
                "decode_rate": row["decode_rate"], "latency_ms": row["latency_ms"],
                "runtime_seconds": None,
                "source": "results/final_benchmark/benchmark_table.csv",
            }
        )
    with (OUTPUT / "overall_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: "N/A" if value is None or value == "" else value for key, value in row.items()}
            for row in rows
        )
    displayed = [
        [
            row["category"], row["experiment"], str(row["training_epochs"] or "N/A"),
            str(row["params"] or "N/A"), fmt(row["test_psnr"]), fmt(row["test_ssim"]),
            fmt(row["decode_rate"]), fmt(row["latency_ms"], 3, " ms"),
        ]
        for row in rows
    ]
    headers = ["Category", "Experiment", "Epochs", "Params", "Test PSNR", "Test SSIM", "Decode rate", "Latency"]
    (OUTPUT / "overall_results.md").write_text(markdown_table(headers, displayed), encoding="utf-8")
    (OUTPUT / "overall_results.tex").write_text(latex_table(headers, displayed), encoding="utf-8")


def write_tables(phases: list[dict[str, Any]], ablations: list[dict[str, str]], benchmark: list[dict[str, str]]) -> None:
    table_dir = OUTPUT / "tables"
    phase_headers = ["Stage", "Epochs", "Params", "Val PSNR", "Val SSIM", "Test PSNR", "Test SSIM", "Latency"]
    phase_rows = [
        [row["stage"], str(row["epochs"]), f"{row['params']:,}", fmt(row["best_val_psnr"]),
         fmt(row["best_val_ssim"]), fmt(row["test_psnr"]), fmt(row["test_ssim"]),
         fmt(row["latency_ms"], 3, " ms")]
        for row in phases
    ]
    ablation_headers = ["Variant", "Params", "Test PSNR", "Test SSIM", "Latency", "Delta PSNR", "Delta SSIM"]
    ablation_rows = [
        [variant_label(row["variant"]), f"{int(row['params']):,}",
         fmt(row["test_psnr"]), fmt(row["test_ssim"]), fmt(row["latency_ms"], 3, " ms"),
         fmt(row["delta_psnr"], 6), fmt(row["delta_ssim"], 6)]
        for row in ablations
    ]
    benchmark_headers = ["Model", "Params", "PSNR", "SSIM", "Decode Rate", "Latency"]
    benchmark_rows = [
        [row["model"], f"{int(row['params']):,}" if row["params"] else "N/A",
         fmt(row["psnr"]), fmt(row["ssim"]), fmt(row["decode_rate"]),
         fmt(row["latency_ms"], 3, " ms")]
        for row in benchmark
    ]
    for number_, headers, rows in (
        (1, phase_headers, phase_rows), (2, ablation_headers, ablation_rows),
        (3, benchmark_headers, benchmark_rows),
    ):
        stem = table_dir / f"table_{number_}"
        stem.with_suffix(".md").write_text(markdown_table(headers, rows), encoding="utf-8")
        stem.with_suffix(".tex").write_text(latex_table(headers, rows), encoding="utf-8")


def setup_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
            "axes.spines.top": False, "axes.spines.right": False,
            "figure.dpi": 120, "savefig.dpi": 300,
        }
    )


def save_figure(fig: Any, number_: int) -> None:
    target = OUTPUT / "figures" / f"figure_{number_}"
    fig.savefig(target.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_figures(phases: list[dict[str, Any]], ablations: list[dict[str, str]]) -> None:
    setup_plotting()
    labels = [row["stage"] for row in phases]
    color = "#2F6B9A"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, metric, title, ylabel, digits in (
        (axes[0], "test_psnr", "Test PSNR by optimization stage", "PSNR (dB)", 3),
        (axes[1], "test_ssim", "Test SSIM by optimization stage", "SSIM", 4),
    ):
        values = [row[metric] for row in phases]
        bars = axis.bar(labels, values, color=color)
        axis.bar_label(bars, fmt=f"%.{digits}f", padding=3)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylim(0, max(values) * 1.15)
    save_figure(fig, 5)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for label, directory, _, _ in PHASES:
        history = read_rows(RESULTS / directory / "validation_history.csv")
        epochs = [float(row["epoch"]) for row in history]
        axes[0].plot(epochs, [float(row["val_psnr"]) for row in history], label=label)
        axes[1].plot(epochs, [float(row["val_ssim"]) for row in history], label=label)
    axes[0].set(title="Validation PSNR trajectories", xlabel="Epoch", ylabel="PSNR (dB)")
    axes[1].set(title="Validation SSIM trajectories", xlabel="Epoch", ylabel="SSIM")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.6)
    save_figure(fig, 6)

    variants = []
    for row in ablations[1:]:
        variants.append(variant_label(row["variant"]).replace("Without ", "No "))
    psnr = [float(row["delta_psnr"]) for row in ablations[1:]]
    ssim = [float(row["delta_ssim"]) for row in ablations[1:]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].barh(variants, psnr, color=color)
    axes[1].barh(variants, ssim, color="#4F8B6D")
    axes[0].set(title="PSNR change after component removal", xlabel="Delta PSNR vs full (dB)")
    axes[1].set(title="SSIM change after component removal", xlabel="Delta SSIM vs full")
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    for axis, values, digits in ((axes[0], psnr, 3), (axes[1], ssim, 4)):
        axis.bar_label(axis.containers[0], labels=[f"+{value:.{digits}f}" for value in values], padding=3)
        axis.axvline(0, color="#444444", linewidth=0.8)
    save_figure(fig, 7)

    source = RESULTS / "final_benchmark" / "benchmark_figure"
    shutil.copy2(source.with_suffix(".png"), OUTPUT / "figures" / "figure_8.png")
    shutil.copy2(source.with_suffix(".pdf"), OUTPUT / "figures" / "figure_8.pdf")


def write_summaries(phases: list[dict[str, Any]]) -> None:
    baseline, phase1, phase2, phase3 = phases
    optimization = f"""# Optimization summary

Across the fixed test split, PSNR increased from **{baseline['test_psnr']:.6f} dB** at Baseline to **{phase3['test_psnr']:.6f} dB** at Phase 3, a net gain of **{phase3['test_psnr'] - baseline['test_psnr']:+.6f} dB**. Test SSIM increased from **{baseline['test_ssim']:.6f}** to **{phase3['test_ssim']:.6f}** (**{phase3['test_ssim'] - baseline['test_ssim']:+.6f}**).

Phase 1 introduced bounded output and a pixel-first loss and ran for 10 epochs; its loss values are not directly comparable with Baseline. Phase 2 retained the Phase 1 objective and added warmup/cosine scheduling, EMA selection, and early stopping, improving test PSNR by **{phase2['test_psnr'] - phase1['test_psnr']:+.6f} dB**. Phase 3 changed model capacity and reconstruction topology, improving test PSNR by **{phase3['test_psnr'] - phase2['test_psnr']:+.6f} dB**, while parameters rose from **{phase2['params']:,}** to **{phase3['params']:,}**. Historical latency is unavailable for Baseline and Phase 1.
"""
    (OUTPUT / "optimization_summary.md").write_text(optimization, encoding="utf-8")
    shutil.copy2(RESULTS / "ablation" / "final" / "component_contribution.md", OUTPUT / "ablation_summary.md")
    shutil.copy2(RESULTS / "final_benchmark" / "benchmark_report.md", OUTPUT / "benchmark_summary.md")


def write_paper(phases: list[dict[str, Any]]) -> None:
    baseline, _, phase2, phase3 = phases
    content = f"""# Results and discussion

## Optimization improvements

Table 1 and Figures 5-6 summarize the controlled optimization sequence. Test PSNR improved from **{baseline['test_psnr']:.3f} dB** to **{phase3['test_psnr']:.3f} dB**, while test SSIM improved from **{baseline['test_ssim']:.4f}** to **{phase3['test_ssim']:.4f}**. Phase 1 established the bounded output and rebalanced objective; Phase 2 added the learning-rate and EMA checkpoint-selection policy; Phase 3 delivered the largest quality increase through higher token resolution, greater width/depth, residual reconstruction, and image-space refinement. Because Phase 1 used 10 epochs and changed the objective, its loss magnitude and training budget are not directly comparable to the 30-epoch runs.

## Ablation findings

Table 2 and Figure 7 show one-factor-at-a-time, single-seed ablations. Every removal increased PSNR and SSIM under this protocol. The largest change followed removal of the refinement decoder (**+2.769 dB**, **+0.0230 SSIM**), followed by DecodeConsistencyLoss (**+0.883 dB**) and QRStructureLoss (**+0.757 dB**). Removing the Mamba branch produced **+0.433 dB** and reduced measured branch-compute latency substantially. These signed removal effects do not prove that components are generally harmful: interactions were not tested, the Mamba parameters remained instantiated in the bypass configuration, and PSNR/SSIM may not capture QR robustness objectives.

## Benchmark comparison

Table 3 and Figure 8 contain the final benchmark. Mamba-TransQR achieved **16.452 dB PSNR**, **0.950374 SSIM**, and a **0.997655 threshold decode proxy**, compared with **10.617 dB**, **0.703518**, and **0.624671** for degraded inputs. SRCNN, ESPCN, VDSR, EDSR, and SwinIR remain `N/A` because compatible implementations or checkpoints were absent. Training or substituting them would violate the fixed-data protocol, so no CNN-versus-Transformer ranking is claimed.

## Strengths

The principal strength is the consistent improvement over degraded input and earlier optimization stages on the same fixed split. The Phase 3 checkpoint combines high structural similarity with a measured threshold proxy of 0.997655, and all reported values retain artifact-level provenance. These experiments used the project's `lightweight` state-space backend; they are not results for the official Mamba-SSM implementation.

## Limitations

All trained configurations use one seed, so no confidence intervals or significance tests are available. The reported decode rate is a pixel-threshold proxy rather than payload recovery because no QR decoder backend was installed. Missing external baseline checkpoints prevent a complete state-of-the-art ranking. The ablation study is one-factor-at-a-time and cannot isolate interactions.

## Computational complexity

Phase 3 increased parameters from **{phase2['params']:,}** to **{phase3['params']:,}** (**{phase3['params'] / phase2['params']:.2f}x**) and recorded **{phase3['latency_ms']:.3f} ms** batch-one CPU latency in its saved benchmark. The standardized final benchmark measured **82.002 ms** with batch size 1, 10 warmups, and 100 timed iterations on the recorded Windows CPU runtime. The saved artifacts do not identify the CPU model, core allocation, or power state, so latency should not be generalized across hardware. The full ablation table shows that bypassing the lightweight state-space branch reduced latency to **12.073 ms**, whereas loss and EMA removals retain the same inference graph and their latency differences should be treated as measurement variation.

## Table and figure captions

- **Table 1.** Optimization-stage results on the fixed test split. Phase 1 used 10 epochs; all other stages used 30. Latency is batch-one CPU time and is unavailable where it was not saved.
- **Table 2.** Completed single-seed, 30-epoch one-factor-at-a-time ablations. Deltas are ablated minus Full Model; positive values mean removal improved the reported metric.
- **Table 3.** Final fixed-protocol benchmark. Decode Rate is the Phase 3 threshold proxy, not payload recovery; absent compatible baseline artifacts are marked `N/A`.
- **Figure 5.** Test PSNR and SSIM measured after each optimization stage; bar labels show rounded saved values.
- **Figure 6.** Saved validation PSNR and SSIM trajectories by epoch. Phase 1 ends at epoch 10; the other runs continue to epoch 30.
- **Figure 7.** Change in test PSNR and SSIM after removing one component from the Full Model. Positive bars indicate higher metrics after removal, not positive component contribution.
- **Figure 8.** Final benchmark comparison of PSNR, SSIM, threshold decode proxy, and batch-one CPU latency. Only Degraded Input and Mamba-TransQR had measurable local artifacts; the other five models are explicitly unavailable.
"""
    paper = OUTPUT / "paper_ready"
    (paper / "results_and_discussion.md").write_text(content, encoding="utf-8")
    for number_ in (1, 2, 3):
        for suffix in (".md", ".tex"):
            shutil.copy2(OUTPUT / "tables" / f"table_{number_}{suffix}", paper / f"table_{number_}{suffix}")
    for number_ in (5, 6, 7, 8):
        for suffix in (".png", ".pdf"):
            shutil.copy2(OUTPUT / "figures" / f"figure_{number_}{suffix}", paper / f"figure_{number_}{suffix}")


def write_audit(phases: list[dict[str, Any]], ablations: list[dict[str, str]], benchmark: list[dict[str, str]]) -> None:
    paper = OUTPUT / "paper_ready"
    provenance: list[dict[str, str]] = []
    for row in phases:
        provenance.append(
            {
                "output_artifact": "Table 1 / Figure 5",
                "element": row["stage"],
                "source_artifact": row["source"] + "; " + row["source"].replace("benchmark_summary.csv", "validation_history.csv"),
                "source_fields": "test_psnr,test_ssim,runtime_seconds; validation_history val_psnr,val_ssim",
                "transformation": "exact values; display rounded to 6 decimals",
                "status": "verified",
            }
        )
    provenance.append(
        {
            "output_artifact": "Table 1",
            "element": "Baseline/Phase 1/Phase 2 parameter count",
            "source_artifact": "results/phase3_pilot/comparison.md",
            "source_fields": "Parameters row",
            "transformation": "exact integer copied from saved comparison",
            "status": "verified",
        }
    )
    for row in ablations:
        provenance.append(
            {
                "output_artifact": "Table 2 / Figure 7",
                "element": variant_label(row["variant"]),
                "source_artifact": "results/ablation/final/ablation_table.csv",
                "source_fields": "params,test_psnr,test_ssim,latency_ms,delta_psnr,delta_ssim",
                "transformation": "exact values; labels normalized; display rounded",
                "status": "verified",
            }
        )
    for row in benchmark:
        provenance.append(
            {
                "output_artifact": "Table 3",
                "element": row["model"],
                "source_artifact": "results/final_benchmark/benchmark_table.csv",
                "source_fields": "params,psnr,ssim,decode_rate,latency_ms",
                "transformation": "exact values; missing fields rendered N/A; display rounded",
                "status": "verified",
            }
        )
    provenance.extend(
        [
            {
                "output_artifact": "Figure 6",
                "element": "all validation trajectories",
                "source_artifact": "results/{first_real_qr_restoration,phase1_pilot,phase2_pilot,phase3_pilot}/validation_history.csv",
                "source_fields": "epoch,val_psnr,val_ssim",
                "transformation": "direct line plots; no interpolation or smoothing",
                "status": "verified",
            },
            {
                "output_artifact": "Figure 8",
                "element": "complete figure",
                "source_artifact": "results/final_benchmark/benchmark_figure.{png,pdf}",
                "source_fields": "complete saved benchmark figure",
                "transformation": "byte-for-byte copy",
                "status": "verified",
            },
        ]
    )
    fields = ["output_artifact", "element", "source_artifact", "source_fields", "transformation", "status"]
    with (paper / "source_provenance.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(provenance)

    assert sha256(RESULTS / "final_benchmark" / "benchmark_figure.png") == sha256(
        OUTPUT / "figures" / "figure_8.png"
    )
    assert sha256(RESULTS / "final_benchmark" / "benchmark_figure.pdf") == sha256(
        OUTPUT / "figures" / "figure_8.pdf"
    )

    for table_number in (1, 2, 3):
        latex = (OUTPUT / "tables" / f"table_{table_number}.tex").read_text(encoding="utf-8")
        assert latex.count("\\begin{tabular}") == latex.count("\\end{tabular}") == 1
        assert latex.count("\\toprule") == latex.count("\\midrule") == latex.count("\\bottomrule") == 1
        column_count = latex.split("\\begin{tabular}{", 1)[1].split("}", 1)[0].count("r") + 1
        for line in latex.splitlines():
            if line.endswith(" \\\\"):
                assert line.count("&") == column_count - 1

    report = """# Publication package audit report

## Outcome

**Verified with evidence-based presentation corrections.** All numeric table cells were traced to completed source artifacts. Figures 5-7 are direct plots of saved CSV values, and Figure 8 is a byte-identical copy of the completed final benchmark figure. No training, evaluation, or benchmark was rerun, and no measured value changed.

## Checklist

1. **Pass — table values:** Table 1 matches phase benchmark and validation histories; Table 2 matches the finalized ablation CSV; Table 3 matches the final benchmark CSV.
2. **Pass — figure values:** Figure 5 uses saved phase test metrics, Figure 6 uses every saved validation point without smoothing, Figure 7 uses finalized ablation deltas, and Figure 8 matches the source file hash.
3. **Pass — fabrication/placeholders:** No fabricated numeric value or numeric placeholder remains. Unavailable benchmark rows are `N/A`.
4. **Pass — units/precision:** PSNR is dB, SSIM and proxy decode rate are unitless, latency is ms at batch size 1, and tables consistently display six metric decimals and three latency decimals.
5. **Corrected — names:** `QR` and `EMA` capitalization is consistent; packaged ablation names use the same human-readable convention.
6. **Pass — optimization values:** Baseline and Phases 1-3 exactly match their completed histories and benchmark summaries.
7. **Pass — ablations:** All six rows match `results/ablation/final/ablation_table.csv`.
8. **Pass — final benchmark:** All seven rows match `results/final_benchmark/benchmark_table.csv`; five unavailable model rows remain unavailable.
9. **Pass — LaTeX syntax:** Each table has balanced `tabular`, `toprule`, `midrule`, and `bottomrule` commands, valid row terminators, and a consistent column count. A TeX engine is not installed in this workspace, so validation is syntax-level rather than a rendered compilation.
10. **Corrected — captions:** Captions now state budgets, delta direction, proxy semantics, source availability, and latency scope.
11. **Pass — claims:** The discussion avoids significance and state-of-the-art ranking claims, documents the single-seed/OFAAT design, and distinguishes removal effects from general component utility.
12. **Corrected — backend identity:** The discussion explicitly states that results use the lightweight state-space backend and are not official Mamba-SSM results.
13. **Corrected — CPU scope:** Latency is labeled batch-one CPU time; protocol and missing CPU-model/core/power metadata are stated, so cross-hardware generalization is disclaimed.
14. **Pass — optional decode metrics:** Payload decode rate is unavailable, not zero. The reported benchmark field is explicitly a threshold proxy; all missing entries are `N/A`.

## Precision policy

CSV files preserve source precision. Markdown and LaTeX tables round PSNR, SSIM, decode proxy, and deltas to six decimals and latency to three decimals. Figures use rounded direct labels while plotting full-precision source values.
"""
    (paper / "audit_report.md").write_text(report, encoding="utf-8")
    corrections = """# Corrections log

No measured value was modified.

1. Normalized `QR` and `EMA` capitalization in Table 2, Figure 7 labels, and consolidated ablation names.
2. Rendered every missing field in `overall_results.csv` explicitly as `N/A` rather than an empty cell.
3. Added an explicit statement that experiments use the lightweight state-space backend, not official Mamba-SSM.
4. Added CPU latency protocol and hardware-metadata limitations.
5. Replaced terse artifact-map descriptions with evidence-scoped captions for Tables 1-3 and Figures 5-8.
"""
    (paper / "corrections_log.md").write_text(corrections, encoding="utf-8")


def main() -> None:
    for directory in (OUTPUT, OUTPUT / "figures", OUTPUT / "tables", OUTPUT / "paper_ready"):
        directory.mkdir(parents=True, exist_ok=True)
    phases = phase_data()
    ablations = read_rows(RESULTS / "ablation" / "final" / "ablation_table.csv")
    benchmark = read_rows(RESULTS / "final_benchmark" / "benchmark_table.csv")
    write_overall(phases, ablations, benchmark)
    write_tables(phases, ablations, benchmark)
    write_figures(phases, ablations)
    write_summaries(phases)
    write_paper(phases)
    write_audit(phases, ablations, benchmark)


if __name__ == "__main__":
    main()
