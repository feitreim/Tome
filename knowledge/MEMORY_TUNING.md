# Memory Tuning Guide

If you are experiencing high memory usage or OOM (Out of Memory) errors on Metal, you can tune the concurrency parameters and KV cache size of the inference node.

## Concurrency Parameters

The `mlx-impl/node.py` script (and `scripts/start_node.sh`) now supports the following parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-rollout-batch-size` | 64 | Maximum number of concurrent sequences during the generation phase of rollouts. Total sequences = `num_prompts * group_size`. |
| `--max-ref-batch-size` | 32 | Maximum number of concurrent sequences when computing reference model log-probs. |
| `--max-judge-batch-size` | 32 | Maximum total judge items active at once during decoding. This is critical for memory if you have many items to judge. |
| `--num-blocks` | 512 | Total number of KV blocks in the pool. Each block is 128 tokens. This is the **biggest RAM consumer**. |

## Recommended Settings for 16GB Macs

For a 16GB M-series Mac running a 0.6B model (like Qwen3-0.6B), the following settings are recommended to avoid heavy SWAP:

```bash
uv run mlx-impl/node.py \
    --max-rollout-batch-size 32 \
    --max-ref-batch-size 16 \
    --max-judge-batch-size 16 \
    --num-blocks 256
```

### Why `--num-blocks` matters:
The KV cache is pre-allocated on the first request. 
- **1024 blocks (old default):** ~15GB (Forces 16GB Macs into heavy SWAP).
- **512 blocks (new default):** ~7.5GB (Safer, but still tight with two model copies).
- **256 blocks:** ~3.75GB (**Recommended** if you want to avoid SWAP entirely).

If you reduce `num_blocks` too much, you will get a `MemoryError: Out of KV blocks` if your total active tokens (Prefixes + Prompts + Generation) exceed the capacity. Each block stores 128 tokens. 256 blocks = 32,768 tokens.
