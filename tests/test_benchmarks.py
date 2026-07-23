"""Benchmark registry and report smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from mambatransqr.benchmarks.metrics import aggregate  # noqa: E402
from mambatransqr.benchmarks.registry import (  # noqa: E402
    BaselineRegistry,
    BaselineSpec,
)
from mambatransqr.benchmarks.tables import export_table  # noqa: E402


def test_registry_rejects_unregistered_substitution() -> None:
    registry = BaselineRegistry()
    with pytest.raises(KeyError):
        registry.build("SRCNN")
    registry.register(
        BaselineSpec("SRCNN", factory=torch.nn.Identity, provenance="local test")
    )
    assert isinstance(registry.build("SRCNN"), torch.nn.Identity)


def test_aggregation_and_table_exports(tmp_path: Path) -> None:
    rows = aggregate(
        [{"model": "x", "psnr": 20.0}, {"model": "x", "psnr": 22.0}], ("model",)
    )
    assert rows[0]["psnr"] == 21.0
    export_table(rows, "overall", tmp_path)
    assert (tmp_path / "overall.tex").is_file()
