# Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Upcasting the two embedding matrices to float32 nearly doubles their size (206 MB f16 -> ~393 MiB f32), which exceeds the 256 MiB shard limit; the spec already anticipates this by requiring oversized tensors to get their own shard, so this is expected, not a bug, but worth flagging since it's easy to assume the "256 MiB per shard" rule is a hard cap.
  - Had to be careful that the buffer-name regex (`attention.bias`, `attention.masked_bias`, `attention.rotary_emb.inv_freq`) didn't also match `attention.dense.weight`/`attention.query_key_value.weight`, since both start with `attention.`.
- Anything in the task text or documentation that was unclear: none; the per-layer tensor names and buffer list were given explicitly, so pattern-matching was unambiguous.
- Tools used (condition F): N/A (condition P).
- Approximate time spent, if you can tell: ~10 minutes.
