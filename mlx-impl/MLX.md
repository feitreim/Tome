# MLX Implementation of Qwen3-0.6B

MLX/Metal implementation of Qwen3-0.6B, a 0.6B parameter dense language model with Grouped Query Attention and QK normalization.

**Model**: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

## Model Architecture

| Parameter | Value |
|---|---|
| Architecture | `Qwen3ForCausalLM` |
| Parameters | ~0.6B |
| Layers | 28 |
| Hidden size | 1024 |
| Vocabulary | 151,936 tokens |
| Max context | 40,960 tokens |
| Data type | bfloat16 |
| RMSNorm eps | 1e-6 |

### Attention

- **Type**: Grouped Query Attention (GQA)
- **Query heads**: 16
- **KV heads**: 8 (2:1 ratio)
- **Head dimension**: 128
- **RoPE**: Non-interleaved Llama-style (theta=1,000,000)
- **QK normalization**: Yes
- **Bias**: None

### MLP

- **Type**: SwiGLU
- **Intermediate size**: 3,072
- **Structure**: `down_proj(silu(gate_proj(x)) * up_proj(x))`

### Decoder Layer

```
x → RMSNorm → Attention (with QK Norm + RoPE) → residual
  → RMSNorm → MLP (SwiGLU) → residual
```

### Embedding

- **Tied embeddings**: Yes (shared `lm_head` and `embed_tokens` weights)

## Commands

```bash
# Run KL divergence measurement against HuggingFace reference
uv run mlx-impl/measure_kl_div.py

# Run component tests (RMSNorm, Attention, MLP, full model)
uv run mlx-impl/test_components.py

# Regenerate HF reference test data
uv run mlx-impl/generate_test_data.py

# Benchmark Metal kernels
uv run mlx-impl/benchmark_kernel.py
```

## KL Divergence

Primary correctness metric. Measures distributional distance between our MLX logits and HuggingFace reference logits. See `KLDIV.md` for details.

**Current results** (32 random tokens, bf16):

| Metric | Value |
|---|---|
| `kl_ref_to_ours_mean` | 0.0069 |
| `kl_ref_to_ours_max` | 0.039 |
| `symmetric_kl_mean` | 0.0070 |

These are within expected bf16 precision for 32 decoder layers.

## Component Test Tolerances

| Component | Max diff | Tolerance |
|---|---|---|
| RMSNorm | 0.000 | 0.1 |
| Attention (layer 0) | 0.004 | 0.1 |
| MLP (layer 0) | 0.063 | 0.1 |
| Full model (32 layers) | 1.031 | 2.0 |

## Custom Metal Kernels

### Fused Norm+RoPE Kernel (`fused_norm_rope`)

Fuses four ops into a single GPU pass for models with QK normalization:

1. Reshape `(B, S, NH*HD)` → `(B, S, NH, HD)`
2. Transpose → `(B, NH, S, HD)`
3. Per-head RMSNorm
4. Non-interleaved RoPE

One threadgroup per (batch, head, seq_pos) with HD threads per group. Uses SIMD reduction for RMSNorm and threadgroup memory for RoPE partner access.

**Performance**: 1.67x speedup over separate MLX ops at S=1024.

### Fused RoPE Kernel (`fused_rope`)

Fuses three ops for models without QK normalization:

1. Reshape `(B, S, NH*HD)` → `(B, S, NH, HD)`
2. Transpose → `(B, NH, S, HD)`
3. Non-interleaved RoPE

**Performance**: Slower than MLX's native `mx.fast.rope` (0.35-0.72x). MLX's built-in rope is already highly optimized, so the Nanbeige inference path uses native MLX ops. The kernel is available for benchmarking and correctness validation.

### Kernel Correctness

Both kernels pass correctness checks against MLX baseline ops across sequence lengths (1, 16, 128, 512) and RoPE offsets (0, 7, 100) with max diff < 0.05.

## RoPE Convention

Nanbeige uses Llama-style non-interleaved RoPE where rotation pairs are `[i, i+d/2]` (HF's `rotate_half`). In MLX this is `traditional=False`.

- `traditional=True` (MLX) = interleaved pairs `[2i, 2i+1]` (GPT-J style)
- `traditional=False` (MLX) = non-interleaved pairs `[i, i+d/2]` (Llama style)

## File Structure

```
mlx-impl/
├── MLX.md                  # This file
├── model.py                # Qwen3/Llama model + Metal kernels
├── load_weights.py         # SafeTensors → MLX weight loading
├── kvcache.py              # KV cache for autoregressive generation
├── measure_kl_div.py       # KL divergence vs HuggingFace
├── test_components.py      # Component tests vs HF reference data
├── generate_test_data.py   # Generate HF reference outputs
├── benchmark.py            # End-to-end inference benchmark
├── benchmark_kernel.py     # Metal kernel microbenchmarks
└── test_inputs/            # Pre-generated HF reference data
    ├── tokens.npy
    ├── embeddings.npy
    ├── rmsnorm_layer0.npy
    ├── attention_layer0.npy
    ├── mlp_input_layer0.npy
    ├── mlp_output_layer0.npy
    ├── full_model_logits.npy
    └── metadata.json
```

## Notes

- **Memory**: ~6 GB for weights (bf16), ~8-10 GB total with activations
- **Inference only**: No backward pass
- **BOS/EOS**: 166100 / 166101
- The model class is named `Qwen3` since the architecture is shared across Llama/Qwen3 families (dense decoder-only with SwiGLU + GQA + RoPE). Config flags (`use_qk_norm`, `rope_traditional`, `tie_word_embeddings`) distinguish variants.
