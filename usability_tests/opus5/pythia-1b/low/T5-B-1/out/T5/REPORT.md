# T5 self-report

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`)
- Number of times you executed the script or plan: 3
- Which executions failed, and why (one line each):
  - 1: `crash` — `PlanLoaderError: transform #4: cast_ received unknown keys: ['dtype']`; `cast_` takes `to:`, not `dtype:` (unlike `cast`).
  - 2: `no_match` — `TransformError: matmul source_b missing` — I had written `from_b` with regex-escaped dots, but `from_b` is a literal rewrite of the `from_a` match (like `to` in `copy`), so the backslashes were taken literally.
  - 3: passed.
- Pitfalls or surprises you hit (one line each):
  - Only `from_a` is a pattern; `from_b` and `to` are rewrites where capture groups (`\1`) are substituted into a literal name — dots must be unescaped there.
  - `cast`/`cast_` have different key names (`dtype` vs `to`).
  - Intermediates had to live on the `model` alias (matmul destination), otherwise the output alias would be ambiguous; they were deleted before writing.
  - Shard budget is tensor data only, so `512MB` (binary MiB) put the two 206 MB embedding tensors into shards without exceeding the limit; 4 shards resulted.
- Anything in the task text or documentation that was unclear:
  - The doc pack does not state that `from_b` in two-source transforms is a rewrite rather than an independent pattern; that cost one execution.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes.
