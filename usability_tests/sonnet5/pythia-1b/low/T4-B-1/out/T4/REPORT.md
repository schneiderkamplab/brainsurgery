## Participant self-report

- Final artifact path: `out/T4/plan.yaml`
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  1. `failed_assertion` — the equal-check comparing base vs. ft1/ft2 outside
     the MLP tensors failed on `attention.masked_bias`, because it is a
     `-inf` float16 constant and `-inf - (-inf) = NaN` makes any `eps`-based
     equal check fail even when the tensors are identical.
- Pitfalls or surprises you hit (one line each):
  - `masked_bias` buffers are `-inf`; had to compare them via a finite
    `clamp` (to `[-65504, 65504]`) instead of a raw `equal` assert.
  - `add`/`subtract` require the destination to already exist, so the
    float32 task-vector arithmetic was built from `copy` + in-place
    `subtract_`/`add_`/`scale_` rather than the non-in-place forms.
  - With three input aliases plus a scratch alias, transform destinations
    span more than one alias, so automatic output-alias inference would be
    ambiguous; used an explicit `save: { alias: base }` instead of a
    top-level `output:` block.
- Anything in the task text or documentation that was unclear:
  - Nothing about the merge itself; the doc pack didn't call out that
    `masked_bias`-style `-inf` constants break naive equality checks,
    which cost the one failed run.
- Tools used (condition F): name, version, and why: n/a (condition B)
- Approximate time spent, if you can tell: ~15 minutes
