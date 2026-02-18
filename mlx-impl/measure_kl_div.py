"""Measure KL divergence between HuggingFace Transformers and MLX model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from load_weights import download_qwen3, load_qwen3_weights
from model import Qwen3

MODEL_NAME = "Nanbeige/Nanbeige4.1-3B"
VOCAB_SIZE = 166144
DEFAULT_SEQ_LEN = 32
DEFAULT_SEED = 42
DEFAULT_TOKENS_PATH = Path(__file__).parent / "test_inputs" / "tokens.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure KL divergence against HuggingFace reference logits.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="HF model id")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Local checkpoint directory. If omitted, uses huggingface_hub snapshot_download.",
    )
    parser.add_argument(
        "--tokens-path",
        type=Path,
        default=DEFAULT_TOKENS_PATH,
        help="Path to .npy token ids. If missing, random tokens are generated.",
    )
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Sequence length for generated tokens")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generated tokens")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for generated tokens")
    parser.add_argument("--save-generated-tokens", action="store_true", help="Persist generated tokens to --tokens-path")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional output path for JSON metrics")
    return parser.parse_args()


def build_our_model(checkpoint_path: str) -> Qwen3:
    model = Qwen3(
        vocab_size=VOCAB_SIZE,
        dim=2560,
        num_layers=32,
        num_heads=20,
        num_kv_heads=4,
        head_dim=128,
        intermediate_size=10496,
        max_seq_len=262144,
        rope_theta=70000000.0,
        eps=1e-5,
        tie_word_embeddings=False,
        use_qk_norm=False,
        rope_traditional=False,
    )
    load_qwen3_weights(model, checkpoint_path)
    return model


def build_hf_model(model_name: str) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    return model


def get_tokens(
    tokens_path: Path, *, batch_size: int, seq_len: int, seed: int, save_generated_tokens: bool
) -> tuple[np.ndarray, str]:
    if tokens_path.exists():
        tokens = np.load(tokens_path)
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens to have shape [batch, seq], got {tokens.shape}")
        return tokens.astype(np.int64), "loaded_from_file"

    rng = np.random.RandomState(seed)
    tokens = rng.randint(0, VOCAB_SIZE, size=(batch_size, seq_len), dtype=np.int64)
    if save_generated_tokens:
        tokens_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(tokens_path, tokens)
    return tokens, "generated_random"


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    logits64 = logits.astype(np.float64, copy=False)
    shifted = logits64 - logits64.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    denom = exp.sum(axis=-1, keepdims=True)
    return exp / denom


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=-1)


def summarize(name: str, values: np.ndarray) -> dict[str, float]:
    flat = values.reshape(-1)
    return {
        f"{name}_mean": float(np.mean(flat)),
        f"{name}_median": float(np.median(flat)),
        f"{name}_max": float(np.max(flat)),
        f"{name}_min": float(np.min(flat)),
    }


def main() -> None:
    args = parse_args()

    print("Loading tokens...")
    tokens, token_source = get_tokens(
        args.tokens_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        save_generated_tokens=args.save_generated_tokens,
    )
    print(f"  token_source={token_source}, shape={tokens.shape}")

    print("Resolving checkpoint...")
    checkpoint_path = args.checkpoint_path or download_qwen3(args.model_name)
    print(f"  checkpoint_path={checkpoint_path}")

    print("Loading models...")
    hf_model = build_hf_model(args.model_name)
    our_model = build_our_model(checkpoint_path)

    print("Running HuggingFace forward pass...")
    tokens_pt = torch.tensor(tokens, dtype=torch.long)
    with torch.no_grad():
        hf_logits = hf_model(tokens_pt).logits.float().numpy()

    print("Running MLX forward pass...")
    our_logits, _ = our_model(mx.array(tokens, dtype=mx.uint32), cache=None, cur_pos=0)
    mx.eval(our_logits)
    our_logits_np = np.array(our_logits.astype(mx.float32))

    print("Computing KL divergence...")
    p_ref = logits_to_probs(hf_logits)
    p_ours = logits_to_probs(our_logits_np)

    kl_ref_to_ours = kl_divergence(p_ref, p_ours)
    kl_ours_to_ref = kl_divergence(p_ours, p_ref)
    sym_kl = 0.5 * (kl_ref_to_ours + kl_ours_to_ref)

    results: dict[str, Any] = {
        "model_name": args.model_name,
        "checkpoint_path": str(checkpoint_path),
        "token_source": token_source,
        "tokens_path": str(args.tokens_path),
        "tokens_shape": list(tokens.shape),
        **summarize("kl_ref_to_ours", kl_ref_to_ours),
        **summarize("kl_ours_to_ref", kl_ours_to_ref),
        **summarize("symmetric_kl", sym_kl),
    }

    print(json.dumps(results, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Saved JSON metrics to {args.json_out}")


if __name__ == "__main__":
    main()
