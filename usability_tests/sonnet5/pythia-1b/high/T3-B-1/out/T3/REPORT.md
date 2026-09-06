# Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the plan succeeded on the first execution.
- Pitfalls or surprises you hit (one line each):
  - The dtype/count `assert` operators apply to every tensor matched by a regex, not just the first
    match, which I confirmed with a quick throwaway probe plan before relying on it for the
    "exactly 64 bfloat16" / "192 rest float32" checks.
  - After upcasting to float32, `embed_in.weight` and `embed_out.weight` grow to ~412 MB each,
    comfortably over the 256 MiB shard budget, so they land alone in their own shards as required;
    worth planning shard math against post-cast sizes, not input sizes.
  - Using `cast_` (in-place) rather than `cast` (copy-to-new-name) kept tensor names unchanged for
    free, which the task requires, and avoided having to `delete` an original after a `cast`+rename.
- Anything in the task text or documentation that was unclear: none; the buffer list, projection
  list, and shard-budget arithmetic in TASK.md matched the actual checkpoint exactly, and the
  README's dtype/count/exists assert docs were sufficient to write correct checks without trial and
  error on the real plan (only a disposable probe plan was run outside `out/`).
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: about 15 minutes, most of it verifying tensor names,
  regex matches, and expected byte sizes against the input file before writing the plan.
