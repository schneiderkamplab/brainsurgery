## Participant self-report

- Final artifact path: `out/T1/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each): the only real hazard is
  renumbering order — build a full old→new index table up front
  (`{old: new for new, old in enumerate(surviving)}`) rather than shifting
  indices in place, so a surviving block can never be overwritten by another
  surviving block regardless of iteration order.
- Anything in the task text or documentation that was unclear: none; the
  per-block tensor list and the required checks were fully specified.
- Tools used (condition F): `safetensors` 0.5.3 for load/save, `torch` 2.14.0
  for tensor equality in ad-hoc verification. Considered `torch-state-bridge`
  for the regex key rewrite and `mergekit` layer-slicing, but a ~70-line
  plain script using a single explicit remap dict was simpler and more
  auditable than fitting this into either tool's config format, and it lets
  the required checks (missing blocks 12-15, exactly 12 surviving blocks,
  exactly 184 tensors) run as hard asserts before any write.
- Approximate time spent, if you can tell: ~10 minutes
