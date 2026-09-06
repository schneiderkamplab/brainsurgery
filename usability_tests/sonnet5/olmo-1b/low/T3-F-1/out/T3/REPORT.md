# Participant self-report — T3-F-1

- Final artifact path: `out/T3/solution.py` (invoked via `out/T3/run.sh`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none.
- Pitfalls or surprises you hit (one line each):
  - Accidentally deleted my own freshly-written `out/T3/` files mid-session
    with an overly broad `rm -rf out/T3 && ... cp .bak ...` cleanup command
    where the `.bak` never existed; had to rewrite `solution.py` and
    `run.sh` from scratch (no data loss to the task itself, since this was
    before the real run — just wasted a step).
  - The projection-matrix regex needed the `\d+` layer index and explicit
    alternation over `q|k|v|o` and `gate|up|down` to avoid also matching
    something outside the 112 target tensors; there are no norms/biases in
    this checkpoint so the "don't touch numerically sensitive tensors" risk
    was mainly about `embed_tokens`/`lm_head`, both excluded by construction
    (only projection-matrix names match the regex).
- Anything in the task text or documentation that was unclear: none; the
  input layout, exact tensor list, and shard-size rule were all specified
  precisely enough to write assertions directly from the spec.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3, used
  directly in a plain script rather than through `transformers`/`mergekit`/
  `torch-state-bridge`. The task is a pure per-tensor dtype cast plus
  re-sharding with no architecture-level operation (no merging, no key
  renaming, no adapter), so a script over the raw safetensors API gives
  full, auditable control over which 112 keys are cast and how the
  256 MiB shard budget is packed, without relying on a higher-level tool's
  own (undocumented, for this exact byte budget) sharding heuristic.
- Approximate time spent, if you can tell: a few minutes of active work
  (one script write, one execution, verification queries).
