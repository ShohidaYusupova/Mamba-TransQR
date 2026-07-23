"""Optional publication figure generation from measured benchmark rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_figures(rows: list[dict[str, Any]], directory: Path) -> list[Path]:
    """Write measured severity/decode scatter plots when matplotlib is installed."""
    try: import matplotlib.pyplot as plt
    except ImportError as error: raise ImportError("Figures require matplotlib; install a plotting dependency.") from error
    directory.mkdir(parents=True, exist_ok=True); paths: list[Path] = []
    for x, y, name in (("degradation_severity", "psnr", "psnr_by_severity"), ("model_inference_latency_ms", "combined_decode_success_rate", "latency_vs_decode_rate")):
        values = [(row.get(x), row.get(y)) for row in rows if row.get(x) is not None and row.get(y) is not None]
        if not values: continue
        figure, axis = plt.subplots(); axis.scatter([item[0] for item in values], [item[1] for item in values]); axis.set(xlabel=x, ylabel=y); figure.savefig(directory / f"{name}.png", dpi=200); figure.savefig(directory / f"{name}.pdf"); plt.close(figure); paths.append(directory / f"{name}.pdf")
    return paths
