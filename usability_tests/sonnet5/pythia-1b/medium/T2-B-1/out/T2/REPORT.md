# Participant self-report

- Final artifact path: `out/T2/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the plan succeeded on the first run.
- Pitfalls or surprises you hit (one line each):
  - `concat`/`copy` cannot write to a tensor name that already exists, so each pruned
    projection had to be built under a temporary `*.pruned` name, then the original
    deleted and the temporary moved back into place (`concat` -> `delete` -> `move`).
  - Had to convert the task's inclusive row/column ranges (e.g. "rows 0..3839") into
    Python-style half-open slice bounds (`[0:3840, :]`) for the `::[slice]` syntax.
- Anything in the task text or documentation that was unclear: none; the per-head
  layout (interleaved q/k/v within each 768-row block, 256-wide column blocks for
  `dense.weight`) was specified precisely enough to compute exact slice bounds
  without guessing.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes of doc lookup (`concat`,
  `move`, `delete`, `assert.shape`, `assert.count` help text) plus one script to
  generate the 16-layer plan and one full run.
