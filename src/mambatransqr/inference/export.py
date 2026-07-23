"""TorchScript and ONNX export helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


def export_torchscript(model: nn.Module, example_input: Tensor, path: str | Path) -> Path:
    """Export a model as a validated TorchScript artifact.

    Args:
        model: Model to export.
        example_input: Representative input tensor.
        path: Output ``.pt`` path.

    Returns:
        Created artifact path.
    """
    destination = _prepare(path)
    model.eval()
    with torch.no_grad():
        scripted = torch.jit.trace(model, example_input)
        scripted.save(str(destination))
    validate_torchscript(destination, example_input)
    return destination


def export_onnx(
    model: nn.Module,
    example_input: Tensor,
    path: str | Path,
    *,
    dynamic_batch: bool = True,
    opset_version: int = 17,
) -> Path:
    """Export a model to ONNX with optional dynamic batch axes.

    Args:
        model: Model to export.
        example_input: Representative input tensor.
        path: Output ``.onnx`` path.
        dynamic_batch: Add dynamic batch axes to inputs and outputs.
        opset_version: ONNX operator-set version.

    Returns:
        Created artifact path.
    """
    destination = _prepare(path)
    axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None
    model.eval()
    torch.onnx.export(
        model,
        example_input,
        destination,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=axes,
        opset_version=opset_version,
        dynamo=False,
    )
    validate_onnx(destination)
    return destination


def validate_torchscript(path: str | Path, example_input: Tensor) -> None:
    """Load a TorchScript artifact and run a validation inference."""
    model = torch.jit.load(str(path), map_location=example_input.device).eval()
    with torch.no_grad():
        output = model(example_input)
    if not isinstance(output, Tensor) or output.shape[0] != example_input.shape[0]:
        raise ValueError("TorchScript export validation produced an invalid output")


def validate_onnx(path: str | Path) -> None:
    """Validate ONNX graph structure when the optional ONNX package is installed."""
    try:
        import onnx
    except ImportError:
        return
    onnx.checker.check_model(str(path))


def _prepare(path: str | Path) -> Path:
    """Create parent directories and return a normalized export destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination
