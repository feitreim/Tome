# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tome is an MLX implementation of Nanbeige4.1-3B, a 3B parameter dense decoder-only transformer with Grouped Query Attention, optimized for Apple Silicon. The implementation loads weights from HuggingFace and validates correctness against the HuggingFace Transformers reference. An inference server (Rust scheduler + MLX inference nodes) is described in `INFERENCE_SERVER.md`.

## Commands

- **Install dependencies**: `uv sync`
- **Run any python**: `uv run ___.py`
- **Run component tests**: `uv run mlx-impl/test_components.py` (compares MLX vs HF outputs)
- **Run KL divergence**: `uv run mlx-impl/measure_kl_div.py`
- **Benchmark**: `uv run mlx-impl/benchmark.py`
- **Benchmark kernels**: `uv run mlx-impl/benchmark_kernel.py`
- **Lint**: `uvx ruff check .`
- **Format**: `uvx ruff format .`
- **Type check**: `uvx pyright`

## Architecture

scheduler implementation in ./scheduler/
model implementation in ./mlx-impl/

**Model** (`mlx-impl/model.py`): Nanbeige4.1-3B built with MLX. Pipeline: token embedding → 32 DecoderLayers (RMSNorm → GQA Attention with RoPE → RMSNorm → SwiGLU MLP) → final RMSNorm → LM head. All parameters are bfloat16. Uses MLX Metal kernels (`mx.fast.rope`, `mx.fast.scaled_dot_product_attention`, `mx.fast.rms_norm`).

**Weight loading** (`mlx-impl/load_weights.py`): Downloads SafeTensors checkpoints via `huggingface_hub`, loads and converts to MLX bfloat16 arrays. Expert weights are stacked per-layer.

**KV Cache** (`mlx-impl/kvcache.py`): Cache for autoregressive generation, concatenates new keys/values per layer.

**Tests** (`mlx-impl/test_components.py`): Comparison tests against HuggingFace Transformers — tests RMSNorm, Attention, MLP, and full model individually against pre-generated reference data.

**KL Divergence** (`mlx-impl/measure_kl_div.py`): Measures distributional distance between MLX and HuggingFace logits. See `KLDIV.md`.

**Custom Metal Kernels** (`mlx-impl/model.py`): Fused norm+RoPE kernel for models with QK normalization. See `mlx-impl/MLX.md` for details and benchmarks.

## Conventions

- Python 3.14, line length 120
- Ruff for linting/formatting with `F722` ignored (jaxtyping annotations)
- Type annotations use `jaxtyping` guarded behind `TYPE_CHECKING`
- Dont comment dimensions, use jaxtyping
- Pyright in standard mode

- For rust, write clean idiomatic rust code.
