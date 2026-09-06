## Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 3 (2 failed on plan
  compilation before a working run; the plan itself ran to completion and
  passed all checks on the first execution attempt that compiled)
- Which executions failed, and why (one line each):
  - Execution 1: `PlanLoaderError: missing model alias in reference` on the
    `matmul` transform's `from_b` — a rewrite pattern still needs an
    `alias::` prefix even though it is only ever used via `re.sub` on the
    `from_a` match, not matched independently.
  - Execution 2: same error, this time on `add_`'s `to` — same cause,
    fixed by prefixing with `base::`.
- Pitfalls or surprises you hit (one line each):
  - `from_b`/`to` on `matmul` (and `to` on `add_`) are not independent
    tensor references; they are string-substitution templates applied to
    each `from_a` match via `re.sub`, so they still require an explicit
    `alias::` prefix to resolve to a model even though nothing is matched
    against them directly.
  - `output.shard: 512MB` maps exactly to the 536,870,912-byte budget in
    the task, and BrainSurgery already isolates any single tensor larger
    than the shard budget into its own shard, so no special-casing of
    `embed_tokens`/`lm_head` was needed.
- Anything in the task text or documentation that was unclear:
  - The README states rewrite semantics for `to` in `copy`/`move` and for
    `assert.equal`'s `right`, but doesn't call out that `matmul`'s
    `from_b` follows the same rewrite-of-`from_a` semantics (rather than
    being independently matched and paired by position) — I had to read
    the transform implementation to confirm the pairing was per-capture,
    not per-order.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~15 minutes
