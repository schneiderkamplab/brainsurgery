## Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Had to be careful to match `attn.bias` buffer names exactly (`h.<i>.attn.bias`) rather than a broad regex, to avoid dropping any real parameter.
  - Sharding required treating `wte.weight` (154 MB) as a special case since it alone exceeds the 64 MiB shard budget.
- Anything in the task text or documentation that was unclear: none; the spec was explicit about which 48 tensors to cast and which buffers to drop.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~10 minutes
