# Participant self-report: T3 (olmo-1b, condition P)

- Final artifact path: `out/T3/solution.py` (output checkpoint in `out/T3/`: 10 shard files plus `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Two bf16 layers sum to exactly 268,435,456 bytes, so the "at most" budget matters: a strict `<` would produce 15 shards instead of 8 layer shards.
  - The 412 MB float32 embeddings and lm_head exceed the budget, so they need the explicit "oversized tensor goes alone" branch rather than plain greedy packing.
  - Used an anchored regex over the seven projection names rather than `.*weight`, so embed_tokens and lm_head are never matched; the script also verifies the matched set equals the expected 112 names.
- Anything in the task text or documentation that was unclear: the task does not say which tensor order to pack shards in or whether the grader checks a specific shard assignment; I packed greedily in the input index order.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: a few minutes.
