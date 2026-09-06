# T2 self-report (condition B, Pythia-1B)

- Final artifact path: `out/T2/plan.yaml` (output checkpoint: `out/T2/model.safetensors`, 244 tensors)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: failed_assertion (my own extra post-check, not a required one). `assert: { count: { of: 'pruned\..*', is: 0 } }` raised "count.of matched zero tensors" instead of evaluating to 0: a zero-match reference is an error in `count`, so it cannot express "no leftovers". Replaced with `assert: { not: { exists: 'pruned\..*' } }`. All edits and the required checks had already passed; nothing was written.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each source reference to resolve to exactly one tensor, so the per-layer slicing cannot be written once with a pattern; the plan lists 16 x 3 concats (generated with a shell loop, but the YAML is fully explicit).
  - Destinations must not already exist, so the pruned tensors go to temporary names (`pruned.<i>.*`), then the originals are deleted by regex and the temporaries moved back with a `\1` capture rewrite; `delete` and `move` do accept patterns.
  - `count` with `is: 0` is not usable as an assertion; use `not: { exists: ... }`.
- Anything in the task text or documentation that was unclear:
  - The README does not say that zero-match references raise inside `count` (and presumably other `of` operators); a note in the assert section would save an execution.
  - Whether `concat` `from` accepts patterns had to be inferred from "each source reference must resolve to exactly one tensor" in `help.txt`.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes
