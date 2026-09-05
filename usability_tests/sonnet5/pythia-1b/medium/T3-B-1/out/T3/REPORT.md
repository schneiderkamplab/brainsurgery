# Participant self-report

- Final artifact path: `out/T3/plan.yaml` (output written to `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `output.shard` (and `save.shard`) only accept an integer plus one of
    `b|kb|mb|gb|tb` (checked against the docpack); the `mb`/`gb` units are
    binary (1024-based), so `256MB` is the string for exactly 256 MiB
    (268,435,456 bytes) — `256MiB` is not accepted syntax.
  - `embed_in.weight`/`embed_out.weight` land at ~412 MB once upcast from
    float16 to float32 (double the 206 MB quoted for the float16 input), so
    each still needs to be alone in its own shard, just a bigger one than the
    task text's number would suggest at first glance.
  - Ordering matters: deleted the non-parameter buffers before the blanket
    `cast_ .* -> float32`, so the uint8 causal-mask buffer and the scalar
    `masked_bias` never had to be cast at all.
- Anything in the task text or documentation that was unclear: no; the
  README's tensor-reference and shard sections and the `help` output for
  `cast_`/`delete`/`assert` were sufficient to build the plan without trial
  and error beyond checking the shard-size string format.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes of reading + one
  successful run.
