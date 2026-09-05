## Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 3
  (1st: `output.shard` given as a bare int, rejected — must be a
  string; 2nd: `output.shard: "268435456"` rejected — the parser only
  accepts `<int><unit>` with unit in `b|kb|mb|gb|tb`, not a raw byte
  count; 3rd: `output.shard: "256MB"` succeeded.)
- Which executions failed, and why (one line each):
  - Exec 1: `PlanLoaderError: output.shard must be a non-empty string when provided` — passed `shard: 268435456` as a YAML int instead of a string.
  - Exec 2: `RuntimeError: invalid output.shard value '268435456'; expected values like 'none', '500MB', '5GB'` — the shard-size grammar wants a unit suffix, not raw bytes.
  - Exec 3: succeeded, all asserts passed, output written.
- Pitfalls or surprises you hit (one line each):
  - `output.shard` must be a unit string (`"256MB"`), and units are
    binary (1 MB = 1024^2 bytes per `parse_shard_size`), so `256MB` is
    exactly 268,435,456 bytes — matches the task's MiB figure exactly,
    which was convenient but not obvious from the README alone (it
    only shows decimal-looking examples like `5GB`/`500MB`).
  - `cast_` (in-place, target regex) was the right tool instead of
    `cast` (copy to new destination) since tensor names must not
    change.
  - `assert: { dtype: { of: <regex>, is: ... } }` and
    `assert: { count: { of: <regex>, is: ... } }` both accept a
    regex matching multiple tensors (dtype requires all matches to
    share the dtype), so a single precise regex for the 112 projection
    weights gave both a "count is exactly 112" and an "all of them are
    bfloat16" check without needing per-tensor assertions.
  - Verified afterwards outside the plan (not required by the task)
    that the bfloat16 casts are bit-exact against `tensor.to(torch.bfloat16)`
    and that untouched tensors (`embed_tokens`, `lm_head`) are
    byte-identical to the input, and that shard packing put the two
    412 MB tensors alone in their own shards while bf16 shards are
    packed to exactly 268,435,456 bytes of tensor data.
- Anything in the task text or documentation that was unclear:
  - The README documents `output.shard` only with decimal-looking
    example values (`5GB`, `500MB`); it doesn't state the parser is
    strict binary-only (`b|kb|mb|gb|tb`, powers of 1024) or that a
    bare integer byte count is rejected. Had to read the traceback
    (which surfaces the regex) to get the exact accepted grammar.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes.
