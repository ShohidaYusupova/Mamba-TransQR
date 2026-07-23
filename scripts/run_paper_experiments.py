"""Paper experiment command entry point."""
from __future__ import annotations
import argparse
def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args(); print(f"Paper experiment recipe: {args.config}")
if __name__ == "__main__": main()
