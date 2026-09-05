# Participant self-report

- Final artifact path: `out/T1/plan.yaml` (output written to `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1 (first execution succeeded)
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `move` requires the destination name to not already exist, so blocks
    must be renumbered in strictly ascending old-index order (each target
    slot is vacated by an earlier move, or was cleared by the initial
    delete, before it is written into) — doing it in a different order
    would raise a "destination already exists" error or, worse, overwrite
    a surviving block if the destination check were bypassed.
  - Confirmed with a throwaway copy of the checkpoint outside the sandbox
    that `move`'s `from`/`to` accept regex with a capture group
    (`gpt_neox\.layers\.3\.(.*)` -> `gpt_neox.layers.2.\1`), matching all
    15 tensors of a block in a single transform, before committing to the
    real plan.
- Anything in the task text or documentation that was unclear: none; the
  per-block tensor list and the source/destination index mapping in
  TASK.md were exact and sufficient. The `move` help text doesn't show a
  regex example, so I verified the pattern-based multi-tensor rename
  worked before relying on it.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: ~10 minutes
