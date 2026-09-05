# Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  1. `output.shard: 268435456` (bare int) failed to parse — the tool requires
     `shard` to be a non-empty string like `500MB`/`5GB`, not a raw byte count.
- Pitfalls or surprises you hit (one line each):
  - `output.shard` only accepts `\d+(b|kb|mb|gb|tb)` (integer, no decimals),
    so a literal 256 MiB cap can't be written as `256.0MB`; had to confirm
    what "MB" actually means here.
  - Despite the decimal-sounding unit names (`MB`/`GB`), the shard-size units
    turned out to be binary (1024-based) in practice — `256MB` produced shards
    capped at exactly 268,435,456 bytes, which is what the task calls
    "256 MiB". Confirmed by inspecting the written shard headers rather than
    trusting the name.
  - `cast_` (in-place) was the right primitive here since names must not
    change; `cast` (out-of-place) would have required copy+delete gymnastics.
  - Casting everything to float32 first, then casting only the 64 projection
    matrices to bfloat16 second, avoided needing two disjoint regexes (one
    for "everything except the 64") — order of the two `cast_` transforms
    matters.
  - No assert expression filters by dtype across a pattern match (`count`
    only counts by name, `dtype` checks a single tensor), so "exactly 64
    tensors are bfloat16" is asserted as "exactly 64 tensor names match my
    bfloat16-targeting regex", backed up by a spot-check `dtype` assert on
    one of them plus one float32 spot-check on the embedding.
- Anything in the task text or documentation that was unclear:
  - The README/help text never states whether shard-size units are decimal
    or binary; the multiplier only becomes clear operationally (or by reading
    source, which I avoided and instead verified against the written file
    sizes).
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: ~15 minutes
