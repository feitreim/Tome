"""Example: Load and run OLMoE with tensor parallelism across multiple GPUs."""

import device
import flax.nnx as nnx
import jax.numpy as jnp
from device import set_global_mesh, setup_tensor_parallel_mesh
from load_weights_sharded import load_olmoe_weights_sharded
from model import OLMoE

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"


def main():
    # Check available devices
    device.print_device_info()

    # Setup tensor parallel mesh (uses all available GPUs by default)
    # For 2 GPUs with 11GB each, this will split the 13.84GB model across both
    mesh = setup_tensor_parallel_mesh()  # Or specify: setup_tensor_parallel_mesh(num_devices=2)
    set_global_mesh(mesh)

    # Create model
    print("Creating OLMoE model...")
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

    # Download checkpoint
    print("\nDownloading checkpoint...")
    from load_weights import download_olmoe

    checkpoint_path = download_olmoe(MODEL_NAME)
    print(f"Checkpoint at: {checkpoint_path}")

    # Load weights with tensor parallel sharding
    print("\nLoading weights with tensor parallelism...")
    with mesh:
        load_olmoe_weights_sharded(model, checkpoint_path, mesh)

    # Run inference
    print("\nRunning inference...")
    tokens = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)

    with mesh:
        logits, _ = model(tokens, cache=None, cur_pos=0)

    print(f"Output shape: {logits.shape}")
    print(f"Output dtype: {logits.dtype}")

    # Sample from logits
    next_token = jnp.argmax(logits[0, -1])
    print(f"Next token: {next_token}")

    print("\n✓ Tensor parallel inference complete!")


if __name__ == "__main__":
    main()
