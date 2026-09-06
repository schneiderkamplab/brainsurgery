# Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Had to be careful to match only the 7 named projection weight names per layer
    (via an anchored regex on `model.layers.<i>.(self_attn|mlp).<proj>.weight`) so that
    `model.embed_tokens.weight` and `lm_head.weight` are not accidentally caught by a
    broader `.*weight` pattern.
  - Shard packing must bound tensor-data bytes per shard (256 MiB), not file size; the
    two embedding/lm_head tensors (412 MB each, float32) each exceed the budget alone and
    must go in their own shard, so packing needs an explicit "oversized tensor gets its
    own shard" branch rather than a fixed tensors-per-shard count.
- Anything in the task text or documentation that was unclear: none; the required tensor
  names, shapes and shard-budget rule were specified explicitly enough to check against.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes to write and verify the script.
