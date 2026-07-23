"""Raw benchmark persistence and resume identity validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def can_resume(root: Path, identity: dict[str, Any]) -> bool:
    path = root / "reproducibility.json"
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")) == identity


def write_rows(root: Path, rows: list[dict[str, Any]], identity: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "raw_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with (root / "raw_results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
    (root / "reproducibility.json").write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
