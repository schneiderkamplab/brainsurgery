# Participant self-report

- Final artifact path: out/T3/plan.yaml
- Number of times you executed the script or plan: 1 (out/T3/plan.yaml), plus a
  separate scratch experiment outside `out/` against a synthetic checkpoint to
  determine the byte-size grammar and bin-packing behavior of `output.shard`
  before touching the real plan.
- Which executions failed, and why (one line each): none; the single
  execution of `out/T3/plan.yaml` passed on the first try.
- Pitfalls or surprises you hit (one line each):
  - The README's `shard: 100MB` example doesn't say whether the unit is
    decimal (10^6) or binary (2^20); the task wants exactly 67,108,864 bytes
    (64 MiB), so I verified with a throwaway plan/checkpoint that `MB` in
    `output.shard` is parsed as `1024**2`, i.e. `64MB` == 64 MiB here.
  - `cast_` (in-place) is the right choice over `cast` here because the task
    requires tensor names to stay unchanged; `cast` would need a temp
    destination plus a rename/move, which is extra steps for no benefit.
  - `assert: { exists: ... }` alone doesn't register an output model with the
    plan compiler (it's read-only), so a plan with only assert/dump
    transforms fails to infer which model to save; needed at least one
    mutating transform touching the model before `output` resolves.
- Anything in the task text or documentation that was unclear: the shard-size
  unit grammar (decimal vs. binary multiplier) isn't documented anywhere in
  the doc pack; had to verify empirically.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: well under 15 minutes of wall
  clock, dominated by reading the doc pack and one shard-size experiment.
