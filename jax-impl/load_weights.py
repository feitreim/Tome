from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import torch
from safetensors.torch import load_file

if TYPE_CHECKING:
    from jax.sharding import Mesh


def download_olmoe(model_name: str = "allenai/OLMoE-1B-7B-0924") -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_name,
        allow_patterns=["*.safetensors", "config.json"],
    )


def load_olmoe_weights(model, checkpoint_path: str | Path, mesh: Mesh | None = None):
    """
    Load OLMoE weights from SafeTensors checkpoint.

    When mesh is provided with tp > 1, expert weights are sharded along the
    expert dimension across devices. All other weights are replicated.
    """
    checkpoint_path = Path(checkpoint_path)
    shard_files = sorted(checkpoint_path.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors files in {checkpoint_path}")

    # Build sharding specs for multi-device
    expert_sharding = None
    replicated = None
    if mesh is not None and mesh.shape["tp"] > 1:
        from jax.sharding import NamedSharding, PartitionSpec

        expert_sharding = NamedSharding(mesh, PartitionSpec("tp", None, None))
        replicated = NamedSharding(mesh, PartitionSpec())
        print(f"Loading weights with tensor parallelism across {mesh.shape['tp']} devices...")

    def _put(arr, *, sharded: bool = False):
        if expert_sharding is None:
            return arr
        return jax.device_put(arr, expert_sharding if sharded else replicated)

    state_dict = {}
    for f in shard_files:
        state_dict.update(load_file(str(f), device="cpu"))

    num_layers = len(model.layers)
    num_experts = model.layers[0].moe.gate_proj.value.shape[0]

    model.embed.embedding.value = _put(_to_jax(state_dict.pop("model.embed_tokens.weight")))
    model.norm.weight.value = _put(_to_jax(state_dict.pop("model.norm.weight")))
    model.lm_head.kernel.value = _put(_to_jax(state_dict.pop("lm_head.weight")).T)

    for i in range(num_layers):
        pfx = f"model.layers.{i}"
        layer = model.layers[i]

        layer.attn.q_proj.kernel.value = _put(_to_jax(state_dict.pop(f"{pfx}.self_attn.q_proj.weight")).T)
        layer.attn.k_proj.kernel.value = _put(_to_jax(state_dict.pop(f"{pfx}.self_attn.k_proj.weight")).T)
        layer.attn.v_proj.kernel.value = _put(_to_jax(state_dict.pop(f"{pfx}.self_attn.v_proj.weight")).T)
        layer.attn.o_proj.kernel.value = _put(_to_jax(state_dict.pop(f"{pfx}.self_attn.o_proj.weight")).T)

        layer.attn.q_norm.weight.value = _put(_to_jax(state_dict.pop(f"{pfx}.self_attn.q_norm.weight")))
        layer.attn.k_norm.weight.value = _put(_to_jax(state_dict.pop(f"{pfx}.self_attn.k_norm.weight")))

        layer.input_norm.weight.value = _put(_to_jax(state_dict.pop(f"{pfx}.input_layernorm.weight")))
        layer.post_attn_norm.weight.value = _put(_to_jax(state_dict.pop(f"{pfx}.post_attention_layernorm.weight")))

        layer.moe.gate.kernel.value = _put(_to_jax(state_dict.pop(f"{pfx}.mlp.gate.weight")).T)

        gate_projs = []
        up_projs = []
        down_projs = []
        for j in range(num_experts):
            gate_projs.append(state_dict.pop(f"{pfx}.mlp.experts.{j}.gate_proj.weight"))
            up_projs.append(state_dict.pop(f"{pfx}.mlp.experts.{j}.up_proj.weight"))
            down_projs.append(state_dict.pop(f"{pfx}.mlp.experts.{j}.down_proj.weight"))

        layer.moe.gate_proj.value = _put(_to_jax(torch.stack(gate_projs)), sharded=True)
        layer.moe.up_proj.value = _put(_to_jax(torch.stack(up_projs)), sharded=True)
        layer.moe.down_proj.value = _put(_to_jax(torch.stack(down_projs)), sharded=True)

    return model


def _to_jax(tensor: torch.Tensor):
    return jnp.array(tensor.to(torch.float32).numpy(), dtype=jnp.bfloat16)
