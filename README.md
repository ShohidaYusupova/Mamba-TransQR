# Mamba-TransQR

Mamba-TransQR is a Python toolkit for developing, training, evaluating, and
deploying Mamba-Transformer models for quality-aware research workflows.

> The implementation is in its initial scaffold stage; public APIs may change
> before the first stable release.

## Requirements

- Python 3.11 or later

## Installation

For a development install:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Development

```bash
ruff check .
black --check .
mypy src tests
pytest
```

## Project layout

```text
src/mambatransqr/  Package source code
tests/              Automated tests
docs/               Documentation sources
configs/            Reproducible experiment configuration
examples/           Runnable examples
scripts/            Developer and operational scripts
notebooks/          Exploratory notebooks
benchmarks/         Performance benchmarks
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please also follow the
[Code of Conduct](CODE_OF_CONDUCT.md) and review our [Security Policy](SECURITY.md).

## License

Distributed under the [MIT License](LICENSE).
