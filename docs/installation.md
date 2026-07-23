# Installation

Mamba-TransQR supports Python 3.11 or newer and PyTorch 2.x.

## Install from a checkout

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development tools, use `python -m pip install -e ".[dev]"`. Install
TensorBoard support with `python -m pip install -e ".[tensorboard]"`.

## Verify

```bash
python -c "from mambatransqr import __version__; print(__version__)"
mambatransqr --help
```
