## Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run succeeded.
- Pitfalls or surprises you hit (one line each):
  - `cast` creates a new destination tensor and requires it not to already exist, so it can't
    rename-in-place under the same key; `cast_` (in-place, regex `target`) is the right tool
    for casting matched tensors without changing their names.
  - The `dtype` assert expression checks every tensor matched by its (possibly regex) `of`
    reference, so pairing it with a `count` assert on the same regex verifies both "exactly
    these N tensors" and "all of them are the right dtype" without a dedicated dtype-count
    check.
  - `output.shard: 256MB` parses as 256 * 1024^2 bytes (binary units despite the decimal-looking
    suffix), matching the 268,435,456-byte budget in the task exactly; the engine already puts
    any single oversized tensor (`embed_tokens`, `lm_head`) alone in its own shard, so no special
    handling was needed for those.
- Anything in the task text or documentation that was unclear: none.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes (single successful attempt after reading
  `help.txt` for `cast_` and the assert operators).
