# Participant self-report: T4 (condition B, OLMo-1B)

- Final artifact path: `out/T4/plan.yaml` (output checkpoint: `out/T4/model.safetensors`, 114 tensors)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `failed_assertion`/`no_match` — my post-cleanup check `assert: { count: { of: 'base::tv[12]\..*', is: 0 } }` raised "count.of matched zero tensors" because a reference that matches nothing is an error, not a count of 0. All preceding transforms (verification, task vectors, scale, add, delete) had already succeeded; no output was written. Replaced with `assert: { not: { exists: ... } }`.
- Pitfalls or surprises you hit (one line each):
  - `count ... is: 0` cannot be used to assert absence; use `not: { exists: ... }` instead.
  - `subtract`/`add` require an existing destination, so the task vector is built as `copy` (ft -> `base::tv1.<name>`) followed by in-place `subtract_` of the base; this also keeps both task vectors computed against the untouched base before any `add_`.
  - All writes must land on one alias for the output to be inferable, so scratch tensors live under the `base` alias with a `tv1.`/`tv2.` prefix and are deleted at the end.
  - Same-name verification across aliases is done with a negative-lookahead regex plus `right: 'ft1::\g<0>'`, combined with `count` checks (114 total, 48 MLP) so identical name sets are implied.
- Anything in the task text or documentation that was unclear:
  - The README does not state that a zero-match reference in `count` raises rather than returning 0.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: about 5 minutes
