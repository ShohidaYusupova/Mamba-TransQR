"""Entry point placeholder for reproducible QR restoration training recipes."""

from __future__ import annotations

import argparse


def main() -> None:
    """Validate the recipe path; applications provide their dataset loaders."""
    parser = argparse.ArgumentParser(prog="train_qr_restoration")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(f"QR restoration recipe: {args.config}")


if __name__ == "__main__":
    main()
