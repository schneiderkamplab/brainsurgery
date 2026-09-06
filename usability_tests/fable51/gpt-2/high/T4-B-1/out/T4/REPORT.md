# T4 (GPT-2 124M), condition B: participant self-report

- Final artifact path: `out/T4/plan.yaml` (output written to `out/T4/model.safetensors`, 160 tensors)
- Number of times you executed the script or plan: 1 execution of `out/T4/plan.yaml` (succeeded on the first run).
  Additionally 1 execution of a separate read-only verification plan, `out/T4/verify/check.yaml`
  (no `output` section; recomputes 0.2*base + 0.4*ft1 + 0.4*ft2 in a scratch alias and asserts it equals
  the written file within eps 1e-6, the 112 other tensors bit-identical to base, count 160). It passed.
- Which executions failed, and why (one line each): none.
- Pitfalls or surprises you hit (one line each):
  - With several inputs the output alias is inferred from the alias the transforms write to, so every
    scratch tensor (`tv1.*`, `tv2.*`) had to be created inside the `base` alias and deleted before saving.
  - Verifying "same tensor names" without a dedicated operator: I combined `count` (160 per checkpoint,
    48 MLP per checkpoint), bit-exact `equal` on the non-MLP complement (base -> ft1, base -> ft2, whose
    `right` must exist), and `equal` with a huge `eps` (1e30) on the MLP tensors ft1/ft2 -> base to prove
    existence + same shape/dtype while allowing the values to differ. `diff` only reports, it does not fail.
  - Ordering hazard handled by materializing both task vectors (copy from ft, `subtract_` the unmodified
    base, `scale_` by 0.4) before any `add_` into base.
  - "Exactly 48 merged" is asserted as `count` of 48 on each task-vector set before they are added, plus
    count 256 (160 + 2*48) at that point and count 160 / no `tv*` left after deletion.
- Anything in the task text or documentation that was unclear:
  - Whether a read-only verification plan counts as an "attempt"; I report it separately above.
  - `assert.reads`/`assert.writes` say "instrumented backends" without stating whether the default
    `inmemory` provider counts accesses (the dump output suggests it does: `reads=3 writes=1`), so I did
    not rely on `writes` for the "exactly 48 merged" check.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: about 3 minutes (mostly reading the doc pack).
