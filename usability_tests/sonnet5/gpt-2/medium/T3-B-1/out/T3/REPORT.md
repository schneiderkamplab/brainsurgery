# Participant self-report

- Final artifact path: `out/T3/plan.yaml` (output written to `out/T3/`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `TransformError: count.of matched zero tensors` — used
    `assert: { count: { of: 'h\.\d+\.attn\.bias', is: 0 } }` to confirm the
    buffers were deleted, but `count.of` (like other tensor-ref resolutions)
    raises on zero matches instead of returning 0, so a post-delete
    "count is 0" check on that same pattern can never succeed; removed the
    check since the final total-tensor-count assert already covers it.
- Pitfalls or surprises you hit (one line each):
  - You cannot assert "a pattern matches nothing" via `assert: count: is: 0`
    for the same reason above; there's no direct way to assert absence of a
    pattern that never legitimately matched.
  - `cast_` (in-place) is simpler than `cast` here since it avoids inventing
    new destination names and possible name/dest-exists issues.
  - The projection-matrix regex must not use a bare `.*weight`, or it would
    also catch `wte.weight`/`wpe.weight`/`ln_*` and the projection biases;
    used `h\.\d+\.(attn\.(c_attn|c_proj)|mlp\.(c_fc|c_proj))\.weight` to hit
    exactly the 4 projection weights per layer, and a separate
    `h\.\d+\.attn\.bias` pattern for the causal-mask buffers (not
    `attn.*bias`, which would also match `attn.c_attn.bias`).
  - `--shard-size`/`output.shard` of `64MB` maps exactly to the 67,108,864-byte
    budget in the task; `wte.weight` (154 MB) landed alone in its own shard
    as expected since it exceeds the budget.
- Anything in the task text or documentation that was unclear:
  - None; the README's tensor-ref, assert, and sharding sections were enough
    to write the plan without needing `help` output beyond a couple of
    lookups (`cast`/`cast_`, `delete`, `assert.count`/`assert.dtype`).
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: ~10 minutes
