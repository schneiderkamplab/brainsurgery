# T4 self-report

- Final artifact path: `out/T4/plan.yaml` (output `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The ordering hazard is real: both task vectors have to be materialised as
    scratch tensors before either is added back, otherwise the second vector is
    taken against an already-merged base.
  - Scratch tensors have to live on the alias that is written (`base::`), or the
    run fails with "cannot infer output model uniquely"; they must be deleted
    again before the output is saved so the count stays at 114.
  - The `equal` assert resolves `right` as a rewrite of each `left` match, so a
    negative lookahead plus `\g<0>` checks all 66 non-MLP tensors in one line,
    and it fails on a missing counterpart, which also covers the name check.
- Anything in the task text or documentation that was unclear:
  - The README does not say explicitly whether `to` in the in-place commands
    (`add_`, `subtract_`) is a capture-group rewrite of `from`; the `add`
    example implies it, and it does work that way.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes.
