# Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Shard file sizes come out slightly above 268,435,456 bytes on disk (by ~1.6KB) because that's safetensors header overhead, not tensor data; the spec explicitly excludes headers from the budget so this is expected, not a bug.
  - Had to be careful the bfloat16 regex matched only the 7 named projection types (`q/k/v/o_proj`, `gate/up/down_proj`) and not `embed_tokens.weight` or `lm_head.weight`, which also end in `.weight`.
- Anything in the task text or documentation that was unclear: none; the tensor list, shard-size rule, and the "oversized tensor gets its own shard" exception were all spelled out clearly enough to implement directly.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: ~10 minutes.
