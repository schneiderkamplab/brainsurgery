# T3 participant self-report

- **Final artifact path:** `out/T3/solution.py` (output checkpoint in `out/T3/`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The obvious `.*weight` pattern would also hit `model.embed_tokens.weight` and `lm_head.weight`, so I wrote a fully anchored regex that enumerates only `self_attn.[qkvo]_proj` and `mlp.(gate|up|down)_proj` per layer.
  - OLMo-1B-0724-hf really has no norm or bias tensors (non-parametric norms), so 114 = 112 projections + 2 embedding matrices; nothing to upcast and nothing to delete.
  - The two float32 embedding matrices are 412 MB each, well over the 256 MiB budget, so the packer needs an explicit "oversized tensor goes in its own shard" branch rather than plain greedy fill.
  - After casting, each layer is exactly 128 MiB in bf16 (4x8 + 3x32), so the remaining 112 tensors pack into 8 exactly-full 256 MiB shards; 10 shards total.
  - Sharding order is under-specified by the task; I used sorted key order, which is what the HF index convention produces.
- **Anything in the task text or documentation that was unclear:**
  - The task does not state the shard ordering or file-naming convention, only the size budget. I assumed HF style (`model-0000i-of-0000n.safetensors`, sorted key order, greedy fill), but a reference that packs in original-checkpoint order could produce a different tensor-to-shard assignment while still satisfying every stated rule.
  - It also does not say whether the index `metadata.total_size` is checked; I emitted it as the sum of tensor bytes.
- **Tools used (condition F):** n/a (condition P: torch 2.14.0 + safetensors 0.5.3 only)
- **Approximate time spent, if you can tell:** ~5 minutes.
