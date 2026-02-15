"""Generate reference outputs from HuggingFace transformers for testing."""

import json
from pathlib import Path

import numpy as np
import torch

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
SEQ_LEN = 32
OUTPUT_DIR = Path("test_inputs")


def _get_hf_model():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cpu")


def _make_tokens(seq_len=SEQ_LEN):
    rng = np.random.RandomState(42)
    return rng.randint(0, 50304, size=(1, seq_len))


def _hf_position_embeddings(hf_model, x_pt, S):
    """Get (cos, sin) position embeddings from HF model's rotary_emb."""
    position_ids = torch.arange(S).unsqueeze(0)
    cos, sin = hf_model.model.rotary_emb(x_pt, position_ids)
    return cos, sin


def generate_test_data():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading HF model...")
    hf_model = _get_hf_model()
    hf_model.eval()

    print("Generating test tokens...")
    tokens = _make_tokens()
    np.save(OUTPUT_DIR / "tokens.npy", tokens)
    print(f"  Saved tokens.npy: shape={tokens.shape}")

    S = tokens.shape[1]

    print("\n[1/5] Generating embeddings...")
    x_pt = hf_model.model.embed_tokens(torch.tensor(tokens))
    np.save(OUTPUT_DIR / "embeddings.npy", x_pt.detach().float().numpy())
    print(f"  Saved embeddings.npy: shape={x_pt.shape}")

    print("\n[2/5] Generating RMSNorm outputs (layer 0)...")
    rmsnorm_out = hf_model.model.layers[0].input_layernorm(x_pt)
    np.save(OUTPUT_DIR / "rmsnorm_layer0.npy", rmsnorm_out.detach().float().numpy())
    print(f"  Saved rmsnorm_layer0.npy: shape={rmsnorm_out.shape}")

    print("\n[3/5] Generating Attention outputs (layer 0)...")
    causal_mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
    causal_mask = causal_mask[None, None, :, :]
    position_embeddings = _hf_position_embeddings(hf_model, rmsnorm_out, S)

    attn_out = hf_model.model.layers[0].self_attn(
        rmsnorm_out, position_embeddings=position_embeddings, attention_mask=causal_mask
    )[0]
    np.save(OUTPUT_DIR / "attention_layer0.npy", attn_out.detach().float().numpy())
    print(f"  Saved attention_layer0.npy: shape={attn_out.shape}")

    print("\n[4/5] Generating MoE outputs (layer 0)...")
    residual = x_pt + attn_out
    moe_input = hf_model.model.layers[0].post_attention_layernorm(residual)
    moe_out = hf_model.model.layers[0].mlp(moe_input)
    np.save(OUTPUT_DIR / "moe_input_layer0.npy", moe_input.detach().float().numpy())
    np.save(OUTPUT_DIR / "moe_output_layer0.npy", moe_out.detach().float().numpy())
    print(f"  Saved moe_input_layer0.npy: shape={moe_input.shape}")
    print(f"  Saved moe_output_layer0.npy: shape={moe_out.shape}")

    print("\n[5/5] Generating full model outputs...")
    with torch.no_grad():
        hf_out = hf_model(torch.tensor(tokens))
    hf_logits = hf_out.logits.float().numpy()
    np.save(OUTPUT_DIR / "full_model_logits.npy", hf_logits)
    print(f"  Saved full_model_logits.npy: shape={hf_logits.shape}")

    metadata = {
        "model_name": MODEL_NAME,
        "seq_len": SEQ_LEN,
        "vocab_size": 50304,
        "dim": 2048,
        "num_layers": 16,
        "num_heads": 16,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("\n  Saved metadata.json")

    print(f"\n{'=' * 50}")
    print("Test data generation complete!")
    print(f"All files saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    generate_test_data()
