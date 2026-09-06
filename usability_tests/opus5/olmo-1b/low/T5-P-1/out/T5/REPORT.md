# T5 — Participant self-report

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the first execution passed all checks.
- Pitfalls or surprises you hit (one line each):
  - PEFT name prefix: adapter keys carry `base_model.model.` in front of the base name, stripped to map onto `model.layers.<i>....weight`.
  - The base checkpoint's own sharding (113 + 1 tensors) does not follow the 512 MiB rule, so the output sharding had to be recomputed rather than reused.
  - `model.embed_tokens.weight` / `lm_head.weight` are 412 MB each, i.e. below the 512 MiB budget, so only the greedy pack decides whether they end up alone; lm_head did, embed_tokens shares a shard with one small tensor.
  - Shard file sizes exceed 512 MiB slightly because of the safetensors header; the budget is on tensor data only.
- Anything in the task text or documentation that was unclear:
  - The task calls embed/lm_head tensors "larger than" 512 MiB and says they are stored alone, but 412 MB < 512 MiB, so the "own shard" outcome depends on packing order, not on the oversize rule.
  - The shard file naming scheme and the key iteration order used by the hidden reference are not specified; I used HF-style `model-000NN-of-000NN.safetensors` with greedy packing in base-index key order.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: ~5 minutes.
