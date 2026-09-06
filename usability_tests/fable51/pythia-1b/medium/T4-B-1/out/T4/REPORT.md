# T4 (Pythia-1B), condition B: participant self-report

- Final artifact path: `out/T4/plan.yaml` (output checkpoint `out/T4/model.safetensors`, 244 tensors)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: failed_assertion (my own extra check, not a required one). The final sanity assert `count: { of: 'base::(base32|tv1|tv2)\..*', is: 0 }` raised "count.of matched zero tensors" because a reference that matches nothing is an error, so `count ... is: 0` can never succeed. All merge steps and required checks had already passed; no output was written. Replaced it with `not: { exists: ... }`.
  - Execution 2: success.
- Pitfalls or surprises you hit (one line each):
  - A zero-match reference is an error everywhere (including `count.of`), so "assert nothing matches" has to be written as `not: { exists: ... }`.
  - `cast`/`copy`/`scale` require the destination not to exist, and `cast_` cannot change the slot dtype without a round trip, so casting the float32 result back into the original names needs `delete` of the original 64 tensors followed by `cast` into those names.
  - The output alias is inferred from the written alias, so all scratch tensors (tv1/tv2/base32) were created under `base::` and deleted before saving; asserts do not count as writes.
  - Verifying the 64 MLP names/shapes/dtypes are shared across the three checkpoints (values differ) was done with `equal` plus a huge `eps`; there is no direct "same name set" assert, so it is combined with `count ... is: 244` per alias.
- Anything in the task text or documentation that was unclear:
  - The README's `add` example `to: '.*.weight'` is misleading; the interfaces reference clarifies that capture-based rewrites (`\1`, `\g<0>`) are what actually map sources to destinations.
  - `help.txt` does not say that zero matches are an error rather than an empty match.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes (two plan runs of roughly 1 minute each plus reading the doc pack)
