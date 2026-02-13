# Tome

JAX/Flax implementation of OLMoE (Open Language Mixture of Experts) with cross-platform device support.

## Installation

### CPU / Apple Silicon (MPS)

```bash
uv sync
```

### NVIDIA GPU (CUDA 12)

```bash
uv sync --extra cuda12
```

> **Note**: Apple Silicon MPS/Metal support is included in the base JAX installation. CUDA support uses optional dependency extras to install the appropriate JAX variant.

## Device Configuration

The project **automatically detects** and uses the best available device:

- **CUDA** (NVIDIA GPU) - highest priority
- **MPS** (Apple Silicon Metal) - second priority
- **CPU** - fallback

### Check Your Device

```bash
uv run python jax-impl/check_device.py
```

### Force Specific Device

Use the `JAX_PLATFORM` environment variable:

```bash
# Force CPU
JAX_PLATFORM=cpu uv run test_olmoe.py

# Force MPS (Apple Silicon)
JAX_PLATFORM=mps uv run test_olmoe.py

# Force CUDA (NVIDIA GPU)
JAX_PLATFORM=gpu uv run test_olmoe.py
```

## Running Tests

```bash
cd jax-impl
uv run test_olmoe.py
```

First run will download the OLMoE-1B-7B checkpoint from HuggingFace.

## Project Structure

- `jax-impl/model.py` - OLMoE model implementation using Flax NNX
- `jax-impl/device.py` - Cross-platform device configuration
- `jax-impl/load_weights.py` - HuggingFace checkpoint loading
- `jax-impl/test_olmoe.py` - Validation tests vs HuggingFace Transformers
- `scheduler/` - Planned Rust inference scheduler

## Development

```bash
# Lint
uvx ruff check .

# Format
uvx ruff format .

# Type check
uvx pyright
```
