# Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run succeeded.
- Pitfalls or surprises you hit (one line each):
  - Had to be careful to match projection keys by exact suffix (`attn.c_attn.weight`,
    `attn.c_proj.weight`, `mlp.c_fc.weight`, `mlp.c_proj.weight`) rather than a loose
    `*.weight` pattern, since that would also catch `wte.weight`, `wpe.weight`, and
    layer-norm weights.
  - `h.<i>.attn.bias` is a non-parameter causal-mask buffer, not a weight matrix; it
    had to be dropped explicitly and excluded from the bfloat16 set (it isn't in that
    set anyway, but it's easy to mentally lump it in with the other `attn.*` keys).
  - Sharding needed a greedy bin-pack over insertion order with an explicit oversized-
    tensor special case, since `wte.weight` (154 MB) alone exceeds the 64 MiB shard cap.
- Anything in the task text or documentation that was unclear: none; the tensor list,
  shapes, and required checks in TASK.md were sufficient to write the script directly.
- Tools used (condition F): n/a (condition P: plain PyTorch + `safetensors`).
- Approximate time spent, if you can tell: ~10 minutes.
