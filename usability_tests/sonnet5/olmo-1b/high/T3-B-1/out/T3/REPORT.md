## Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 1 (the plan in `out/T3/plan.yaml`
  ran once, succeeded, and produced `out/T3`). Before writing that plan I ran two
  scratch probes outside `out/` (`/tmp/check.yaml`, `/tmp/check2.yaml`) against
  `inputs/base` with no `output:` section, to confirm the input tensor layout
  (114 tensors, shapes) and that `cast_` + a multi-match `dtype` assert behave
  as expected on the real checkpoint; neither wrote any output, so they aren't
  counted as attempts on the task artifact.
- Which executions failed, and why: none failed.
- Pitfalls or surprises you hit:
  - The doc pack's `assert.dtype` help text doesn't say whether `of` can match
    multiple tensors at once (unlike `reads`/`writes`, which explicitly say
    "every matched tensor"); I confirmed empirically with a scratch probe that
    a regex matching all 112 projections works and requires all matches to
    share the dtype, then used that directly as one of the required checks.
  - `cast_` (in-place) was the right tool over `cast`: it keeps tensor names
    unchanged and needs no separate delete-original step, unlike `cast` which
    writes to a new destination.
  - Shard budget arithmetic: "256MB" in `output.shard` is binary (256 x 1024 x
    1024 = 268,435,456 bytes), which is exactly the 256 MiB budget the task
    asks for, so no unit conversion was needed.
- Anything in the task text or documentation that was unclear:
  - Whether `assert.dtype`'s `of` reference aggregates over multiple matches
    (see pitfall above) isn't stated in `help.txt`; had to verify by running
    the tool rather than reading a spec.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~15 minutes (doc review, two
  scratch probes against `inputs/base`, one plan write, one real run, output
  verification).
