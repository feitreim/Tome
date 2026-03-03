import mlx.core as mx
import numpy as np
import sys
import os
import asyncio

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Qwen3
from load_weights import download_qwen3, load_qwen3_weights
from node import InferenceNodeServicer

# Import generated gRPC code
sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__)) + "/generated"))
import inference_pb2

def test_grpo_rpc():
    print("\n[TEST] GRPO RPC (Full Pipeline)")
    
    # Standard Qwen3-0.6B config
    config = {
        "vocab_size": 151936,
        "dim": 1024,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 3072,
        "max_seq_len": 40960,
        "rope_theta": 1000000.0,
        "eps": 1e-6,
        "tie_word_embeddings": True,
        "use_qk_norm": True,
        "rope_traditional": False,
    }
    
    model = Qwen3(**config)
    ref_model = Qwen3(**config)
    
    servicer = InferenceNodeServicer(model, ref_model)
    
    # 1. Create a GRPO request
    prompt1 = inference_pb2.Prompt(prompt_id="p1", tokens=[1, 2, 3, 4, 5])
    prompt2 = inference_pb2.Prompt(prompt_id="p2", tokens=[10, 11, 12])
    
    judge_config = inference_pb2.JudgeConfig(
        rubric_tokens=[100, 101],
        temperature=0.0,
        max_tokens=5
    )
    
    request = inference_pb2.GRPORequest(
        batch_id="batch-42",
        prompts=[prompt1, prompt2],
        group_size=2, # G=2 rollouts per prompt
        temperature=0.7,
        max_tokens=10,
        max_concurrent=2,
        judge=judge_config
    )
    
    # 2. Call GRPO
    async def run_grpo():
        return await servicer.GRPO(request, None)
    
    print("  Running GRPO pipeline...")
    response = asyncio.run(run_grpo())
    
    assert response.batch_id == "batch-42"
    assert len(response.results) == 2
    
    for p_res in response.results:
        print(f"  Prompt {p_res.prompt_id}: {len(p_res.completions)} completions")
        assert len(p_res.completions) == 2
        for c_idx, completion in enumerate(p_res.completions):
            print(f"    Completion {c_idx}: {len(completion.tokens)} tokens, judge_score={completion.judge_score}")
            assert len(completion.tokens) > 0
            assert len(completion.log_probs) == len(completion.tokens)
            assert len(completion.ref_log_probs) == len(completion.tokens)
            assert 0.0 <= completion.judge_score <= 9.0
            
    print("  GRPO RPC PASSED")

if __name__ == "__main__":
    test_grpo_rpc()
