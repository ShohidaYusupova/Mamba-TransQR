"""Generate a synthetic paired QR dataset from a YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from mambatransqr.data.dataset_builder import (
    SyntheticQRDatasetBuilder,
    load_generation_config,
)


def main() -> None:
    """Parse arguments and create a synthetic QR dataset."""
    parser = argparse.ArgumentParser(prog="generate_qr_dataset")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = SyntheticQRDatasetBuilder(load_generation_config(args.config)).build(
        args.output
    )
    print(f"Generated {len(records)} paired QR samples in {args.output}")


if __name__ == "__main__":
    main()
