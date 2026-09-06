# T2 participant self-report (condition B, olmo-1b)

- Final artifact path: `out/T2/plan.yaml` (output: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each `from` reference to resolve to exactly one tensor, so the plan needs 64 explicit concat entries (16 layers x 4 projections) instead of one pattern-based transform; I generated them with a shell loop.
  - `concat`/`copy` destinations must not exist, so the pruned tensors go to a temporary `pruned.<name>` slot, then the originals are deleted and the temporaries are moved back with a regex capture.
  - Slicing on tensor references (`::[:640, :]`, `::[:, 768:]`) is the only way to select head blocks; there is no slice-removal transform.
- Anything in the task text or documentation that was unclear:
  - The `concat` help example `concat: { from: , to: a::xy, dim: 0 }` has an empty `from`, which looks like a rendering bug.
  - The `move` help does not say whether regex patterns with capture groups are supported in `from`/`to`; I inferred it from the README's `assert.equal` text ("like `to` in copy/move") and it worked.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes (one plan execution of ~17 s).
