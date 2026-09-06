# Participant self-report

- Final artifact path: `out/T1/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the plan succeeded on the first run.
- Pitfalls or surprises you hit (one line each):
  - The renumbering map is not monotonic in a way that avoids overlap (e.g. old 4 -> new 3, old 3 -> new 2), so `move` steps had to be ordered by ascending old index; deleting the 4 target blocks first, then moving in that order, guarantees every destination name is already vacated (by an earlier delete or an earlier move in the same list) before it is written.
  - `move` requires whole-tensor names (no slicing) but does support regex `from`/`to` with capture-group rewrites, same as `copy`/`assign`, so each block's 7 tensors could be renamed in one `move` transform per old index instead of one per individual tensor.
- Anything in the task text or documentation that was unclear: none; the README's note that `equal`'s `right` rewrite rules also apply to `move`'s `to` was enough to confirm bulk regex renaming was supported before trying it.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes, mostly reading `help.txt`/README before writing the plan; the run itself was correct on the first attempt.
