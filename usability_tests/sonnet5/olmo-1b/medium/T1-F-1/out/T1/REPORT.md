# T1 self-report

- Final artifact path: `out/T1/solution.py` (invoked via `out/T1/run.sh`), output at `out/T1/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Renumbering has to build a brand-new dict from old->new index rather than
    mutating keys in place, otherwise writing an early new index (e.g. block 3
    -> 2) can collide with an old block that hasn't been processed yet
    (old block 2 is dropped here, so no actual collision occurred, but the
    hazard is general for other drop patterns).
  - safetensors requires contiguous tensors; loaded tensors from
    `load_file` already are, but I added an explicit `.contiguous()` pass
    before saving as a defensive measure.
- Anything in the task text or documentation that was unclear: no, the task
  spec, including the exact old->new index table, was unambiguous.
- Tools used (condition F): `safetensors` 0.5.3 for `load_file`/`save_file`
  only. Chose a plain script over `mergekit` layer-slicing or
  `torch-state-bridge` regex rewriting because the transform (drop 4 named
  blocks, renumber the rest contiguously, leave 2 tensors untouched) is a
  direct dict rebuild that's simpler and more auditable to write and verify
  directly than to express as a merge config or a set of regex capture
  rules, and it lets the required checks be asserted explicitly in the same
  script that produces the output.
- Approximate time spent, if you can tell: a few minutes; single attempt,
  no debugging needed.
