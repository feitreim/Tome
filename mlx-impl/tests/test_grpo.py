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

def test_grpo_new_api():
    print("\n[TEST] GRPO New API (Rollout & Judge)")
    
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
    
    # Use small models for testing to avoid huge memory usage
    config.update({
        "num_layers": 2,
        "dim": 128,
        "intermediate_size": 256,
        "num_heads": 4,
        "num_kv_heads": 2,
    })
    
    model = Qwen3(**config)
    ref_model = Qwen3(**config)
    
    servicer = InferenceNodeServicer(model, ref_model)
    
    # --- PHASE 1: ROLLOUT ---
    print("  Testing Rollout...")
    prompt1 = inference_pb2.Prompt(prompt_id="p1", tokens=[1, 2, 3])
    prompt2 = inference_pb2.Prompt(prompt_id="p2", tokens=[10, 11])
    
    rollout_req = inference_pb2.RolloutRequest(
        batch_id="rollout-batch",
        prompts=[prompt1, prompt2],
        group_size=2, # G=2 rollouts per prompt
        temperature=0.7,
        max_tokens=10
    )
    
    async def run_rollout():
        return await servicer.Rollout(rollout_req, None)
    
    rollout_resp = asyncio.run(run_rollout())
    
    assert rollout_resp.batch_id == "rollout-batch"
    assert len(rollout_resp.results) == 2
    
    completions_to_judge = []
    for p_res in rollout_resp.results:
        print(f"    Prompt {p_res.prompt_id}: {len(p_res.completions)} completions")
        assert len(p_res.completions) == 2
        for c_idx, completion in enumerate(p_res.completions):
            assert len(completion.tokens) > 0
            assert len(completion.log_probs) == len(completion.tokens)
            assert len(completion.ref_log_probs) == len(completion.tokens)
            # Collect for judging phase
            completions_to_judge.append((p_res.prompt_id, completion.tokens))

    # --- PHASE 2: JUDGE ---
    print("  Testing Judge...")
    # Rubric + Prompt + Completion
    rubric_tokens = [100, 101]
    judge_items = []
    for i, (p_id, c_tokens) in enumerate(completions_to_judge):
        # The trainer would construct Rubric + Prompt + Completion
        # Tome just executes it.
        judge_items.append(inference_pb2.JudgeItem(
            item_id=f"item-{i}",
            prompt_tokens=[1, 2, 3] + list(c_tokens) # Simplified: using prompt1 tokens for all
        ))
    
    judge_req = inference_pb2.JudgeRequest(
        batch_id="judge-batch",
        rubric_tokens=rubric_tokens,
        items=judge_items,
        temperature=0.0,
        max_tokens=5
    )
    
    async def run_judge():
        return await servicer.Judge(judge_req, None)
    
    judge_resp = asyncio.run(run_judge())
    
    assert judge_resp.batch_id == "judge-batch"
    assert len(judge_resp.results) == len(judge_items)
    
    for res in judge_resp.results:
        print(f"    Item {res.item_id}: verdict len={len(res.verdict_tokens)}")
        assert len(res.verdict_tokens) > 0
        assert len(res.log_probs) == len(res.verdict_tokens)
            
    print("  GRPO NEW API PASSED")

if __name__ == "__main__":
    test_grpo_new_api()
