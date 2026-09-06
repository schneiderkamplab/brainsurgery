## Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The two 412 MB tensors (`embed_tokens`, `lm_head`) each exceed the 256 MiB
    shard budget on their own, so they each need a dedicated shard; a naive
    greedy bin-pack handles this if oversized tensors are special-cased first.
  - Casting the projection matrices had to be restricted with an exact regex
    (`self_attn.(q|k|v|o)_proj` / `mlp.(gate|up|down)_proj`) so it wouldn't
    accidentally also match `embed_tokens.weight` or `lm_head.weight`, both of
    which end in `.weight` too.
- Anything in the task text or documentation that was unclear: none.
- Tools used (condition F): name, version, and why: N/A (condition P)
- Approximate time spent, if you can tell: ~5 minutes
