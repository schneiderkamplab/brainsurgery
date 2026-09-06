# Participant self-report

- Final artifact path: out/T3/solution.py
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `h.<i>.attn.bias` (causal-mask buffer) shares a suffix with real bias parameters, so it was matched with an anchored regex rather than a `.bias` suffix match.
  - Projection matrices were matched with an anchored regex on the four exact module names so `ln_*.weight`, `wte`/`wpe` were never touched.
  - `wte.weight` (154 MB) exceeds the 64 MiB budget and was placed alone in its own shard.
- Anything in the task text or documentation that was unclear: whether shards must follow the original key order or a specific naming scheme; used greedy fill in original key order with `model-0000i-of-0000n.safetensors` names.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~2 minutes
