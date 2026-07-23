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

## Official Mamba backend

The production default is `mamba_backend="mamba_ssm"`. Install the official
backend after installing a CUDA-enabled PyTorch build:

```bash
python -m pip install -e ".[mamba]" --no-build-isolation
```

The upstream `mamba-ssm` project documents Linux, an NVIDIA GPU, CUDA 11.6+,
and PyTorch 1.12+ as requirements. It is not documented as a supported native
Windows or CPU-only installation. Use a supported Linux/NVIDIA environment (or
an appropriate Linux environment such as WSL with compatible GPU support); do
not expect this extra to make the official backend available on unsupported
platforms. For CPU smoke tests, set `mamba_backend="lightweight"`; that backend
is intentionally not official Mamba.

## Verify

```bash
python -c "from mambatransqr import __version__; print(__version__)"
mambatransqr --help
```
