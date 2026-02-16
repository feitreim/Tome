"""Generate reference outputs from HuggingFace transformers for testing."""

import json
from pathlib import Path

import numpy as np
import torch

MODEL_NAME = "Qwen/Qwen3-0.6B"
SEQ_LEN = 32
OUTPUT_DIR = Path(__file__).parent / "test_inputs"


def _get_hf_model():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cpu")


def _make_tokens(vocab_size=151936, seq_len=SEQ_LEN):
    rng = np.random.RandomState(42)
    return rng.randint(0, vocab_size, size=(1, seq_len))


def generate_test_data():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading HF model...")
    hf_model = _get_hf_model()
    hf_model.eval()

    print("Generating test tokens...")
    tokens = _make_tokens()
    np.save(OUTPUT_DIR / "tokens.npy", tokens)
    print(f"  Saved tokens.npy: shape={tokens.shape}")

    seq_len = tokens.shape[1]
    tokens_pt = torch.tensor(tokens)

    print("\n[1/6] Generating embeddings...")
    x_pt = hf_model.model.embed_tokens(tokens_pt)
    np.save(OUTPUT_DIR / "embeddings.npy", x_pt.detach().float().numpy())
    print(f"  Saved embeddings.npy: shape={x_pt.shape}")

    print("\n[2/6] Generating RMSNorm outputs (layer 0)...")
    rmsnorm_out = hf_model.model.layers[0].input_layernorm(x_pt)
    np.save(OUTPUT_DIR / "rmsnorm_layer0.npy", rmsnorm_out.detach().float().numpy())
    print(f"  Saved rmsnorm_layer0.npy: shape={rmsnorm_out.shape}")

    print("\n[3/6] Generating Attention outputs (layer 0)...")
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = hf_model.model.rotary_emb(rmsnorm_out, position_ids)
    position_embeddings = (cos, sin)

    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    causal_mask = causal_mask[None, None, :, :]

    attn_out = hf_model.model.layers[0].self_attn(
        rmsnorm_out, position_embeddings=position_embeddings, attention_mask=causal_mask
    )[0]
    np.save(OUTPUT_DIR / "attention_layer0.npy", attn_out.detach().float().numpy())
    print(f"  Saved attention_layer0.npy: shape={attn_out.shape}")

    print("\n[4/6] Generating MLP outputs (layer 0)...")
    residual = x_pt + attn_out
    mlp_input = hf_model.model.layers[0].post_attention_layernorm(residual)
    mlp_out = hf_model.model.layers[0].mlp(mlp_input)
    np.save(OUTPUT_DIR / "mlp_input_layer0.npy", mlp_input.detach().float().numpy())
    np.save(OUTPUT_DIR / "mlp_output_layer0.npy", mlp_out.detach().float().numpy())
    print(f"  Saved mlp_input_layer0.npy: shape={mlp_input.shape}")
    print(f"  Saved mlp_output_layer0.npy: shape={mlp_out.shape}")

    print("\n[5/6] Generating full model logits...")
    with torch.no_grad():
        hf_out = hf_model(tokens_pt)
    hf_logits = hf_out.logits.float().numpy()
    np.save(OUTPUT_DIR / "full_model_logits.npy", hf_logits)
    print(f"  Saved full_model_logits.npy: shape={hf_logits.shape}")

    print("\n[6/6] Generating full model top-k predictions...")
    top_k = 10
    top_indices = np.argsort(-hf_logits[0, -1, :])[:top_k]
    top_probs = hf_logits[0, -1, top_indices]
    print(f"  Top {top_k} predictions at last position:")
    for idx, prob in zip(top_indices, top_probs, strict=True):
        print(f"    token {idx}: logit {prob:.4f}")

    metadata = {
        "model_name": MODEL_NAME,
        "seq_len": SEQ_LEN,
        "vocab_size": 151936,
        "dim": 1024,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 3072,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("\n  Saved metadata.json")

    print(f"\n{'=' * 50}")
    print("Test data generation complete!")
    print(f"All files saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    generate_test_data()
