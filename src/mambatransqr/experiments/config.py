"""Experiment configuration persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    """Write configuration JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return destination
