# KL Divergence Validation for `mlx-impl` vs HuggingFace Transformers

This document explains how to measure distributional drift between:

- Reference implementation: HuggingFace Transformers (`Nanbeige/Nanbeige4.1-3B`)
- Your implementation: `mlx-impl` (`mlx-impl/model.py` + `mlx-impl/load_weights.py`)

The script is `mlx-impl/measure_kl_div.py`.

## Why KL divergence matters

Comparing only `argmax` next-token predictions can miss meaningful errors. Two models can pick the same top token while having very different full probability distributions.

KL divergence measures this distributional distance:

- `KL(P_ref || P_ours)` answers: how much information is lost when your model approximates the reference.
- Lower is better.
- `0` means identical distributions (up to numerical precision).

This is useful for validating:

- Weight loading correctness
- Numerical stability changes (e.g., bf16 behavior)
- Kernel changes (fused ops, custom attention paths)
- Regression risk after refactors

## What the script does

For the same input token IDs, it:

1. Runs HF forward pass to get reference logits.
2. Runs your MLX model forward pass to get logits.
3. Converts logits to probabilities via stable softmax.
4. Computes per-position KL divergence over the full vocabulary:
   - `KL(P_ref || P_ours)`
   - `KL(P_ours || P_ref)` (reverse direction)
   - Symmetric KL: `0.5 * (forward + reverse)`
5. Reports summary stats: mean, median, min, max.

## Formula

For one token position with vocab probabilities `P` and `Q`:

`KL(P || Q) = sum_i P(i) * (log P(i) - log Q(i))`

The script clips probabilities with a small epsilon before `log` for numerical safety.

## How to run

From repo root:

```bash
uv run mlx-impl/measure_kl_div.py
```

Optional flags:

```bash
uv run mlx-impl/measure_kl_div.py \
  --tokens-path mlx-impl/test_inputs/tokens.npy \
  --json-out mlx-impl/test_inputs/kl_metrics.json
```

If `--tokens-path` does not exist, random tokens are generated. To persist them:

```bash
uv run mlx-impl/measure_kl_div.py \
  --tokens-path mlx-impl/test_inputs/tokens.npy \
  --save-generated-tokens
```

## Interpreting results

- Focus on `kl_ref_to_ours_mean` as the primary metric.
- Track trends over time (commit-to-commit), not only one absolute value.
- Use `max` to catch isolated pathological positions.
- If KL rises unexpectedly, inspect:
  - weight mapping in `mlx-impl/load_weights.py`
  - fused kernel path in `mlx-impl/model.py`
  - dtype/casting differences
  - attention masking / RoPE offsets / cache position handling

## Notes

- The script compares logits directly from full forward pass (no sampling).
- This is an offline parity metric, complementary to component diffs in `mlx-impl/test_components.py`.
