# Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The oversized embedding tensors end up larger after upcasting to float32
    (about 412 MB, not the 206 MB float16 size quoted in the task) but the
    spec's "own shard, no budget applied" rule still covers this correctly.
  - Buffer names (`attention.bias`, `attention.masked_bias`,
    `rotary_emb.inv_freq`) needed exact per-layer regexes so they wouldn't
    accidentally overlap projection-weight patterns (they don't, but it's
    worth double-checking since both live under `attention.*`).
- Anything in the task text or documentation that was unclear: none; the
  spec's numbers (64 projections, 48 buffers, 196 output tensors) matched
  exactly what the script computed, which made it easy to self-check.
- Tools used (condition F): N/A (condition P).
- Approximate time spent, if you can tell: ~10 minutes.
