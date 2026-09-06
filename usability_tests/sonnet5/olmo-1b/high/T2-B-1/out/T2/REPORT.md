## Participant self-report

- Final artifact path: `out/T2/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `concat`'s `to` is a single destination (no capture-group rewrite like `copy`/`move`/`delete`), so
    reassembling q/k/v/o_proj needed one `concat` call per (layer, projection) — 64 calls — while the
    slicing (`copy`) and cleanup (`delete`/`move`) steps could each be expressed as one regex-batched
    call across all 16 layers.
  - Had to route around `copy`'s "destination must not exist" / `move`'s "destination must not already
    exist" rules by working through temporary tensor names (`.tmpa`/`.tmpb`/`.pruned`) and only
    `delete`-ing the originals right before the final `move` back onto the original names.
- Anything in the task text or documentation that was unclear: none; the README's structured-expression
  and capture-rewrite notes (and the `assert.equal`/`copy` examples using `\1` capture groups) were
  sufficient to figure out the batching split between `copy`/`move`/`delete` (batchable) and `concat`
  (not batchable).
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: single-pass plan, no retries needed.
