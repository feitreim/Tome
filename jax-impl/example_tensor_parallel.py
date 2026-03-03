"""Example: Load and run OLMoE with tensor parallelism across multiple GPUs."""

import device
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from device import setup_mesh
from load_weights import download_olmoe, load_olmoe_weights
from model import OLMoE

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"


def main():
    device.print_device_info()

    mesh = setup_mesh()  # Uses all available devices

    print("Downloading checkpoint...")
    checkpoint_path = download_olmoe(MODEL_NAME)
    print(f"Checkpoint at: {checkpoint_path}")

    print("\nCreating OLMoE model with tensor parallelism...")
    with jax.set_mesh(mesh):
        model = OLMoE(
            vocab_size=50304,
            dim=2048,
            num_layers=16,
            num_heads=16,
            num_kv_heads=16,
            num_experts=64,
            active_experts=8,
            intermediate_size=1024,
            max_seq_len=2048,
            rngs=nnx.Rngs(0),
        )

        print("\nLoading weights...")
        load_olmoe_weights(model, checkpoint_path, mesh=mesh)

        print("\nRunning inference...")
        tokens = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
        logits, _ = model(tokens, cache=None, cur_pos=0)

    print(f"Output shape: {logits.shape}")
    print(f"Output dtype: {logits.dtype}")

    next_token = jnp.argmax(logits[0, -1])
    print(f"Next token: {next_token}")
    print("\nTensor parallel inference complete!")


if __name__ == "__main__":
    main()
