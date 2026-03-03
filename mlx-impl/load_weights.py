from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import torch
from safetensors.torch import load_file


def download_qwen3(model_name: str = "Qwen/Qwen3-0.6B") -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_name,
        allow_patterns=["*.safetensors", "config.json"],
    )


def _to_mlx(tensor: torch.Tensor) -> mx.array:
    # Use float16 as it's often faster on Metal, as noted in rl-values
    return mx.array(tensor.to(torch.float32).numpy()).astype(mx.float16)


def load_qwen3_weights(model: nn.Module, checkpoint_path: str | Path):
    checkpoint_path = Path(checkpoint_path)
    shard_files = sorted(checkpoint_path.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors files in {checkpoint_path}")

    state_dict: dict[str, torch.Tensor] = {}
    for f in shard_files:
        state_dict.update(load_file(str(f), device="cpu"))

    num_layers = len(model.layers)

    # Embedding (shared with lm_head when tie_word_embeddings=True)
    model.embed_tokens.weight = _to_mlx(state_dict.pop("model.embed_tokens.weight"))

    # Final norm
    model.norm.weight = _to_mlx(state_dict.pop("model.norm.weight"))

    # LM head (only if not tied)
    if not model.tie_word_embeddings:
        model.lm_head.weight = _to_mlx(state_dict.pop("lm_head.weight"))
    else:
        state_dict.pop("lm_head.weight", None)

    for i in range(num_layers):
        pfx = f"model.layers.{i}"
        layer = model.layers[i]
        attn = layer.self_attn

        # Fused QKV projection
        q_wt = _to_mlx(state_dict.pop(f"{pfx}.self_attn.q_proj.weight"))
        k_wt = _to_mlx(state_dict.pop(f"{pfx}.self_attn.k_proj.weight"))
        v_wt = _to_mlx(state_dict.pop(f"{pfx}.self_attn.v_proj.weight"))
        attn.qkv_proj.weight = mx.concatenate([q_wt, k_wt, v_wt], axis=0)
        
        # O projection
        attn.o_proj.weight = _to_mlx(state_dict.pop(f"{pfx}.self_attn.o_proj.weight"))

        # QK norms are present for Qwen-style configs and absent for Llama-style configs.
        q_norm_key = f"{pfx}.self_attn.q_norm.weight"
        k_norm_key = f"{pfx}.self_attn.k_norm.weight"
        if attn.q_norm is not None and attn.k_norm is not None:
            attn.q_norm.weight = _to_mlx(state_dict.pop(q_norm_key))
            attn.k_norm.weight = _to_mlx(state_dict.pop(k_norm_key))
        else:
            state_dict.pop(q_norm_key, None)
            state_dict.pop(k_norm_key, None)

        # Layer norms
        layer.input_layernorm.weight = _to_mlx(state_dict.pop(f"{pfx}.input_layernorm.weight"))
        layer.post_attention_layernorm.weight = _to_mlx(state_dict.pop(f"{pfx}.post_attention_layernorm.weight"))

        # Fused MLP gate/up projection
        gate_wt = _to_mlx(state_dict.pop(f"{pfx}.mlp.gate_proj.weight"))
        up_wt = _to_mlx(state_dict.pop(f"{pfx}.mlp.up_proj.weight"))
        layer.mlp.gate_up_proj.weight = mx.concatenate([gate_wt, up_wt], axis=0)
        
        # MLP down projection
        layer.mlp.down_proj.weight = _to_mlx(state_dict.pop(f"{pfx}.mlp.down_proj.weight"))

    if state_dict:
        print(f"Warning: {len(state_dict)} unused weights: {list(state_dict.keys())[:10]}")

    return model
