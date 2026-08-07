"""Finalize the controlled benchmark without training or model evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "final_benchmark"
PAPER_READY = ROOT / "results" / "final_analysis" / "paper_ready"
MODELS = (
    "Degraded Input",
    "SRCNN",
    "ESPCN",
    "VDSR",
    "EDSR",
    "SwinIR",
    "Mamba-TransQR",
)
BASELINES = ("SRCNN", "ESPCN", "VDSR", "EDSR", "SwinIR")
FIELDS = ("model", "params", "psnr", "ssim", "decode_rate", "latency_ms")


def _read_rows() -> list[dict[str, str]]:
    with (OUTPUT / "benchmark_table.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    if tuple(row["model"] for row in rows) != MODELS:
        raise ValueError("benchmark rows are missing or out of canonical order")
    mamba = rows[-1]
    if any(not mamba[field] for field in FIELDS[1:]):
        raise ValueError("the completed Mamba-TransQR row is incomplete")
    return rows


def _checkpoint_inventory() -> dict[str, list[str]]:
    paths = [path for path in (ROOT / "checkpoints").rglob("*.pt") if path.is_file()]
    inventory: dict[str, list[str]] = {}
    for model in BASELINES:
        key = model.lower()
        inventory[model] = [
            str(path.relative_to(ROOT))
            for path in paths
            if key in path.as_posix().lower()
        ]
        if inventory[model]:
            raise ValueError(
                f"{model} has candidate local checkpoints requiring explicit provenance: "
                + ", ".join(inventory[model])
            )
    return inventory


def _value(row: dict[str, Any], field: str, digits: int) -> str:
    value = row.get(field)
    return "N/A" if value in (None, "") else f"{float(value):.{digits}f}"


def _table(rows: list[dict[str, str]]) -> str:
    header = (
        "| Model | Params | PSNR | SSIM | Decode Rate | Latency |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |\n"
    )
    body = "".join(
        f"| {row['model']} | "
        f"{row['params'] if row['params'] else 'N/A'} | "
        f"{_value(row, 'psnr', 6)} | {_value(row, 'ssim', 6)} | "
        f"{_value(row, 'decode_rate', 6)} | "
        f"{_value(row, 'latency_ms', 3)}"
        f"{' ms' if row['latency_ms'] else ''} |\n"
        for row in rows
    )
    return header + body


def main() -> None:
    rows = _read_rows()
    inventory = _checkpoint_inventory()
    for row in rows:
        if row["model"] in BASELINES and any(row[field] for field in FIELDS[1:]):
            raise ValueError(f"unproven measured values found for {row['model']}")

    table = _table(rows)
    (OUTPUT / "benchmark_table.md").write_text(table, encoding="utf-8")
    # Rewrite the canonical CSV from its exact string values, preserving precision.
    with (OUTPUT / "benchmark_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    missing = "\n".join(
        f"- **{model}: unavailable.** No checkpoint matched `{model.lower()}` "
        "under `checkpoints/`, and the local baseline registry declares no "
        "implementation/checkpoint."
        for model in BASELINES
    )
    mamba = rows[-1]
    comparison = f"""# Controlled benchmark comparison

## Outcome

The local checkpoint audit found no compatible checkpoint for SRCNN, ESPCN, VDSR, EDSR, or SwinIR. Those models could not be evaluated under the controlled protocol and remain `N/A`; no literature value, estimate, downloaded weight, or substitute implementation was used.

{table}

## Checkpoint availability

{missing}

## Controlled protocol

- Fixed 1,500-image test split from `datasets/first_real_qr_10k`.
- RGB preprocessing and normalization to `[0, 1]`.
- The same Phase 3 PSNR and SSIM implementations.
- Decode Rate is the same threshold-based Phase 3 proxy, not payload recovery; no decoder backend was installed.
- Batch-one CPU latency uses 10 warmups and 100 timed iterations.

## Preserved completed result

The Mamba-TransQR row is unchanged: **{float(mamba['psnr']):.6f} dB PSNR**, **{float(mamba['ssim']):.6f} SSIM**, **{float(mamba['decode_rate']):.6f} decode proxy**, **{float(mamba['latency_ms']):.3f} ms latency**, and **{int(mamba['params']):,} parameters**.

Because all five requested baseline checkpoints are absent, this package does not support a numeric CNN/Transformer ranking.
"""
    (OUTPUT / "comparison.md").write_text(comparison, encoding="utf-8")

    publication = f"""# Publication Table 3

**Table 3.** Controlled fixed-split benchmark. Decode Rate is the Phase 3 threshold proxy rather than payload recovery. `N/A` denotes a missing compatible local checkpoint and implementation; it does not denote zero performance.

{table}
"""
    (OUTPUT / "publication_table3.md").write_text(publication, encoding="utf-8")
    PAPER_READY.mkdir(parents=True, exist_ok=True)
    (PAPER_READY / "publication_table3.md").write_text(
        publication, encoding="utf-8"
    )

    assert all(not paths for paths in inventory.values())


if __name__ == "__main__":
    main()
