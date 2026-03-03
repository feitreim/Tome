# Tome API Reference

Tome provides a high-performance API for GRPO (Group Relative Policy Optimization) training loops, optimized for Apple Silicon via MLX.

## GRPO Rollout API

Generates multiple rollouts per prompt and computes both policy and reference model log-probs in a single pass.

### `POST /v1/grpo/rollout`

**Request Body:**
```json
{
  "batch_id": "string",
  "prompts": [
    {
      "prompt_id": "string",
      "prompt": "string"
    }
  ],
  "group_size": 16,
  "temperature": 0.7,
  "max_tokens": 512
}
```

**Response Body:**
```json
{
  "batch_id": "string",
  "results": [
    {
      "prompt_id": "string",
      "completions": [
        {
          "tokens": [1, 2, 3],
          "log_probs": [-0.1, -0.2, -0.3],
          "ref_log_probs": [-0.15, -0.25, -0.35]
        }
      ]
    }
  ]
}
```

- **Efficiency Note:** Tome uses **Copy-on-Write (CoW)** KV caching. All `group_size` rollouts for a single prompt share the prompt's KV blocks, only diverging when they generate different tokens.
- **Reference Model:** `ref_log_probs` are computed using a frozen copy of the base model kept in memory.

---

## GRPO Judge API

Executes batch judging of rollouts using a shared rubric prefix.

### `POST /v1/grpo/judge`

**Request Body:**
```json
{
  "batch_id": "string",
  "rubric": "string",
  "items": [
    {
      "item_id": "string",
      "prompt": "string"
    }
  ],
  "temperature": 0.0,
  "max_tokens": 16
}
```

**Response Body:**
```json
{
  "batch_id": "string",
  "results": [
    {
      "item_id": "string",
      "verdict_tokens": [10, 11, 12],
      "log_probs": [-0.01, -0.02, -0.03]
    }
  ]
}
```

- **Efficiency Note:** The `rubric` is prefilled once and cached across all `items` in the batch. Tome uses a **Radix Tree** prefix cache to ensure maximum KV reuse between prompts and completions.
- **Usage:** Typically, the trainer constructs `item.prompt` as `Original Prompt + Completion`. Tome executes `Rubric + Original Prompt + Completion` to get the judge's verdict.

---

## Weight Update API

Applies LoRA weight updates to the policy model in-place.

### `POST /v1/weights`

**Request Body:**
```json
{
  "updates": [
    {
      "layer_idx": 0,
      "param_name": "self_attn.q_proj",
      "lora_a": "base64_encoded_bf16_bytes",
      "lora_b": "base64_encoded_bf16_bytes",
      "shape_a": [16, 1024],
      "shape_b": [1024, 16]
    }
  ]
}
```

**Response Body:**
```json
{
  "node_id": {
    "success": true,
    "version": 1
  }
}
```

- **In-place Merge:** Tome computes $W_{new} = W_{base} + B 	imes A$ and applies it to the active policy model.
- **Consistency:** Weight updates invalidate the policy model's prefix cache to ensure future rollouts use the updated parameters. The reference model remains frozen.

---

## Management API

### `GET /v1/models`
Returns the currently loaded model ID.

### `GET /v1/nodes`
Returns a list of active inference nodes and their current status (cached tokens, queue depth, etc).
