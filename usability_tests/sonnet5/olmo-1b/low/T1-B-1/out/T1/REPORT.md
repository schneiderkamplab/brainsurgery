# Participant self-report

- Final artifact path: `out/T1/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The renumbering is a lookup table (old index -> new index shifted by however many deleted
    blocks precede it: -0, -0, -1, -1, -1, -2, -2, -2, -3, -3, -3, -4), not a uniform arithmetic
    shift, so it needed one explicit `move` per surviving block rather than a single regex.
  - To avoid a `move` destination collision (a still-to-be-moved block occupying the slot another
    block is about to move into), the ten `move` transforms had to run in increasing old-index
    order so each destination is already vacated (by the earlier `delete` or by a prior move in
    the list) before it is targeted.
  - Blocks 0 and 1 keep their original indices, so no `move` was written for them.
- Anything in the task text or documentation that was unclear: none; the `move`/`delete`/`assert`
  help entries and the regex capture-group rewrite semantics (documented under `assert: equal`,
  shared with `copy`/`move`) were sufficient.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes.
