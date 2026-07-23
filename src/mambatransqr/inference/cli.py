"""Command-line interface for Mamba-TransQR inference and export."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mambatransqr.data.dataset_builder import (
    SyntheticQRDatasetBuilder,
    load_generation_config,
)
from mambatransqr.inference.export import export_onnx, export_torchscript
from mambatransqr.inference.pipeline import InferencePipeline
from mambatransqr.inference.predictor import Predictor
from mambatransqr.inference.video import VideoPredictor
from mambatransqr.models import ModelConfig, build_model


def main() -> None:
    """Parse command-line arguments and dispatch an inference command."""
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.command == "generate-dataset":
        SyntheticQRDatasetBuilder(load_generation_config(arguments.config)).build(
            arguments.output
        )
        return
    config = ModelConfig(image_size=arguments.image_size)
    predictor = Predictor(
        config=config,
        checkpoint=getattr(arguments, "checkpoint", None),
        device=arguments.device,
        mixed_precision=not arguments.no_amp,
    )
    if arguments.command == "predict":
        pipeline = InferencePipeline(predictor)
        result = pipeline.predict_image(
            arguments.input, arguments.output, save_original=True
        )
        pipeline.write_reports([result], arguments.output)
    elif arguments.command == "predict-folder":
        pipeline = InferencePipeline(predictor)
        results = pipeline.predict_folder(arguments.input, arguments.output)
        pipeline.write_reports(results, arguments.output)
    elif arguments.command == "predict-video":
        VideoPredictor(predictor).predict_video(arguments.input, arguments.output)
    elif arguments.command == "export":
        model = build_model(config)
        if arguments.checkpoint:
            Predictor(
                model, config, checkpoint=arguments.checkpoint, device=arguments.device
            )
        inputs = torch.zeros(1, 3, arguments.image_size, arguments.image_size)
        if arguments.format == "torchscript":
            export_torchscript(model, inputs, arguments.output)
        else:
            export_onnx(
                model,
                inputs,
                arguments.output,
                dynamic_batch=not arguments.static_batch,
            )


def _parser() -> argparse.ArgumentParser:
    """Build the command parser with all supported subcommands."""
    parser = argparse.ArgumentParser(prog="mambatransqr")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("predict", "predict-folder", "predict-video"):
        command = subcommands.add_parser(name)
        command.add_argument("input")
        command.add_argument("output")
        command.add_argument("--checkpoint")
        command.add_argument("--device", default="auto")
        command.add_argument("--image-size", type=int, default=256)
        command.add_argument("--no-amp", action="store_true")
    export = subcommands.add_parser("export")
    export.add_argument("output", type=Path)
    export.add_argument("--checkpoint")
    export.add_argument(
        "--format", choices=("torchscript", "onnx"), default="torchscript"
    )
    export.add_argument("--device", default="cpu")
    export.add_argument("--image-size", type=int, default=256)
    export.add_argument("--no-amp", action="store_true")
    export.add_argument("--static-batch", action="store_true")
    generate = subcommands.add_parser("generate-dataset")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    return parser
