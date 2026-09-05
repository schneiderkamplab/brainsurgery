# Participant self-report

- Final artifact path: `out/T1/solution.py` (invoked via `out/T1/run.sh`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, first execution succeeded
- Pitfalls or surprises you hit (one line each):
  - Tensors read via `safe_open(...).get_tensor(...)` can be non-contiguous
    views (e.g. after slicing across shards); safetensors' `save_file`
    requires contiguous tensors, so the script calls `.contiguous()` on every
    tensor before saving to avoid a save-time error.
  - The drop-list `{2, 6, 10, 14}` is not evenly spaced from the block
    count, so the old→new index map has to be built explicitly per
    surviving index (`enumerate` over the filtered list) rather than by a
    fixed arithmetic offset, to avoid renumbering collisions.
- Anything in the task text or documentation that was unclear: none; the
  explicit old→new mapping in TASK.md made verification straightforward.
- Tools used (condition F): `safetensors` 0.5.3 for reading/writing the
  sharded/single-file checkpoints directly. Chose a plain script over
  `mergekit`'s layer-slicing (which targets contiguous keep-ranges, not an
  arbitrary drop-list of non-adjacent blocks) and over `torch-state-bridge`
  (which would only wrap the same regex rename this script does directly,
  adding a dependency without reducing risk). `torch` was used incidentally
  for tensor equality checks during manual verification, not in the
  solution itself.
- Approximate time spent, if you can tell: a few minutes.
