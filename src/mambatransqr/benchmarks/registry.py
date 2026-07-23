"""Explicit baseline registry with checkpoint provenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from torch import nn


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    """A benchmark model declaration; no weights are inferred or substituted."""

    name: str
    factory: Callable[[], nn.Module] | None = None
    checkpoint: str | None = None
    configuration: dict[str, object] | None = None
    provenance: str | None = None


class BaselineRegistry:
    """Registry for declared baselines and their provenance."""

    names = ("Degraded Input", "SRCNN", "ESPCN", "VDSR", "EDSR", "SwinIR", "Mamba-TransQR")

    def __init__(self) -> None: self._items: dict[str, BaselineSpec] = {}

    def register(self, spec: BaselineSpec) -> None:
        if spec.name not in self.names: raise ValueError(f"unknown baseline: {spec.name}")
        if spec.name in self._items: raise ValueError(f"baseline already registered: {spec.name}")
        if spec.checkpoint is not None and not Path(spec.checkpoint).is_file():
            raise FileNotFoundError(f"declared checkpoint does not exist: {spec.checkpoint}")
        self._items[spec.name] = spec

    def build(self, name: str) -> nn.Module:
        try: spec = self._items[name]
        except KeyError as error: raise KeyError(f"baseline is not registered: {name}") from error
        if spec.factory is None: raise RuntimeError(f"{name} has no local implementation/checkpoint; no substitution is permitted")
        return spec.factory()

    def provenance(self, name: str) -> dict[str, object]:
        spec = self._items[name]
        return {"name": spec.name, "checkpoint": spec.checkpoint, "configuration": spec.configuration, "provenance": spec.provenance}
