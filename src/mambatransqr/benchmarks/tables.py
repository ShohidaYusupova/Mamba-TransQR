"""CSV, Markdown, and LaTex publication table export."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def export_table(rows: list[dict[str, Any]], stem: str, directory: Path) -> None:
    """Export a measured table in CSV, Markdown, and LaTeX formats."""
    directory.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with (directory / f"{stem}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    markdown = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)] + [
        "| " + " | ".join(str(row.get(key, "")) for key in columns) + " |"
        for row in rows
    ]
    (directory / f"{stem}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    latex = (
        [
            "\\begin{tabular}{" + "l" * len(columns) + "}",
            " & ".join(columns) + " \\\\ \\hline",
        ]
        + [
            " & ".join(str(row.get(key, "")) for key in columns) + " \\\\"
            for row in rows
        ]
        + ["\\end{tabular}"]
    )
    (directory / f"{stem}.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
