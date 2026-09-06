## Participant self-report

- Final artifact path: `out/T2/plan.yaml` (executed to produce `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `concat`'s `from` list requires each source reference to resolve to exactly one
    tensor and `to` to be a single unsliced destination, so it cannot itself sweep
    across all 16 layers in one call the way `copy`/`move` capture-rewrite can;
    worked around this by emitting one `concat` per layer per tensor (48 total)
    into `<name>__pruned` temporaries, then collapsing the cleanup into 3 `delete`
    calls (regex over all layers) and 3 `move` calls (regex capture-rewrite
    `\1` over all layers) to rename the temporaries back onto the original names.
  - `concat`/`move` destinations must not already exist, so the pruned tensor had
    to land under a temporary name and only take the original name after the
    original was deleted; doing this in the other order would have failed on
    `dest_exists`.
- Anything in the task text or documentation that was unclear: none; the row/column
  boundaries for the fused QKV block and the interleaved q/k/v layout were spelled
  out explicitly enough to slice directly without trial and error.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes, single attempt.
