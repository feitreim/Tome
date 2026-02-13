from pathlib import Path

import jax.numpy as jnp
from safetensors.torch import load_file


def download_olmoe(model_name: str = "allenai/OLMoE-1B-7B-0924") -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_name,
        allow_patterns=["*.safetensors", "config.json"],
    )


def load_olmoe_weights(model, checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)
    shard_files = sorted(checkpoint_path.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors files in {checkpoint_path}")

    state_dict = {}
    for f in shard_files:
        state_dict.update(load_file(str(f), device="cpu"))

    num_layers = len(model.layers)
    num_experts = model.layers[0].moe.gate_proj.value.shape[0]

    # Embed
    model.embed.embedding.value = _to_jax(state_dict["model.embed_tokens.weight"])

    # Final norm + lm_head
    model.norm.weight.value = _to_jax(state_dict["model.norm.weight"])
    model.lm_head.kernel.value = _to_jax(state_dict["lm_head.weight"]).T

    for i in range(num_layers):
        pfx = f"model.layers.{i}"
        layer = model.layers[i]

        # Attention projections (transpose for nnx.Linear)
        layer.attn.q_proj.kernel.value = _to_jax(state_dict[f"{pfx}.self_attn.q_proj.weight"]).T
        layer.attn.k_proj.kernel.value = _to_jax(state_dict[f"{pfx}.self_attn.k_proj.weight"]).T
        layer.attn.v_proj.kernel.value = _to_jax(state_dict[f"{pfx}.self_attn.v_proj.weight"]).T
        layer.attn.o_proj.kernel.value = _to_jax(state_dict[f"{pfx}.self_attn.o_proj.weight"]).T

        # QK norms
        layer.attn.q_norm.weight.value = _to_jax(state_dict[f"{pfx}.self_attn.q_norm.weight"])
        layer.attn.k_norm.weight.value = _to_jax(state_dict[f"{pfx}.self_attn.k_norm.weight"])

        # Layer norms
        layer.input_norm.weight.value = _to_jax(state_dict[f"{pfx}.input_layernorm.weight"])
        layer.post_attn_norm.weight.value = _to_jax(state_dict[f"{pfx}.post_attention_layernorm.weight"])

        # MoE gate (transpose for nnx.Linear)
        layer.moe.gate.kernel.value = _to_jax(state_dict[f"{pfx}.mlp.gate.weight"]).T

        # Expert weights — stack per-expert into (num_experts, ...)
        gate_projs = []
        up_projs = []
        down_projs = []
        for j in range(num_experts):
            gate_projs.append(state_dict[f"{pfx}.mlp.experts.{j}.gate_proj.weight"])
            up_projs.append(state_dict[f"{pfx}.mlp.experts.{j}.up_proj.weight"])
            down_projs.append(state_dict[f"{pfx}.mlp.experts.{j}.down_proj.weight"])

        import torch

        layer.moe.gate_proj.value = _to_jax(torch.stack(gate_projs))
        layer.moe.up_proj.value = _to_jax(torch.stack(up_projs))
        layer.moe.down_proj.value = _to_jax(torch.stack(down_projs))

    return model


def _to_jax(tensor):
    import torch

    return jnp.array(tensor.to(torch.float32).numpy(), dtype=jnp.bfloat16)
