# T4 participant self-report (condition B, OLMo-1B-0724-hf)

- Final artifact path: `out/T4/plan.yaml` (output checkpoint `out/T4/model.safetensors`, 114 tensors; executed-plan summary in `out/T4/summary.yaml`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: failed_assertion/no_match at the very last check. After `delete` of the temporaries I asserted `count: { of: 'base::tv[12]\..*', is: 0 }`, but `count` raises "matched zero tensors" before comparing, so a zero count can never be asserted that way. The merge itself had already completed; only the output write was skipped. Replaced with `not: { exists: ... }` and execution 2 passed.
- Pitfalls or surprises you hit (one line each):
  - `assert: count` with `is: 0` is not expressible; use `not: { exists: ref }` for "nothing matches".
  - `subtract` / `add` (three-ref form) have no documented rewrite rule for `to` relative to `from_a`/`from_b`, so I built the task vectors with the two-ref in-place forms instead: `copy` ft -> `base::tv1.<i>.<proj>`, `subtract_` base MLP from it, `scale_` by 0.4, `add_` into the base MLP tensor, then `delete` the temporaries. Both task vectors are subtracted before either `add_` runs, so each is taken against the unmodified base.
  - The output alias is inferred from where transforms write, so every destination had to carry the `base::` prefix (temporaries included) to keep `base` as the single written alias.
  - `count`/`equal` with a negative lookahead regex (`(?!...mlp...).+`) plus `right: 'ft1::\g<0>'` worked as the README describes for the 66-tensor shared-backbone check.
- Anything in the task text or documentation that was unclear:
  - The README lists `writes` / `reads` asserts "for instrumented backends" without saying whether the default `inmemory` provider is instrumented, so I did not rely on them for the "exactly 48 tensors were merged" check and used counts of the task-vector temporaries instead.
  - The help for `add`/`subtract` (three-ref form) does not say how `to` is rewritten when `from_a` and `from_b` are patterns on different aliases.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes (two plan runs of ~15-20 s each plus reading the doc pack)
