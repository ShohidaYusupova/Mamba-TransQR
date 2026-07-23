# Contributing to Mamba-TransQR

Thank you for contributing. Please open an issue before starting substantial
work so the proposed change can be discussed.

## Development setup

1. Use Python 3.11 or later.
2. Create and activate a virtual environment.
3. Install the project with `python -m pip install -e ".[dev]"`.
4. Run `ruff check .`, `black --check .`, `mypy src tests`, and `pytest` before
   submitting a pull request.

## Pull requests

- Keep changes focused and include tests for behavior changes.
- Write clear commit messages and describe the motivation in the pull request.
- Do not commit secrets, private data, generated artifacts, or large datasets.
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
