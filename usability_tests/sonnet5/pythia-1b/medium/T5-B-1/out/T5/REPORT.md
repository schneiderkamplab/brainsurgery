## Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  1. `matmul`'s `from_b` failed with "source_b missing" because `from_b` is a
     substitution template applied via `re.sub(from_a_pattern, from_b, name)`,
     not an independent regex to match against the second model's tensors —
     I had written it as an escaped regex pattern (`\.` etc.) instead of a
     literal replacement string with an unescaped `\1` backreference.
- Pitfalls or surprises you hit (one line each):
  - `matmul`'s `from_b` and `to` are both substitution templates keyed off
    `from_a`'s capture groups, not independently-matched references; only
    `from_a` is matched by regex.
  - `add_`/`cast`/`scale_` require the destination to already exist (for
    `add_`) or not exist (for `cast`), so the merge has to go
    matmul (new lora_delta) -> scale_ (in place) -> cast (new f16 tensor) ->
    add_ (into the existing base weight) -> delete both intermediates.
  - Shard size strings only accept `b|kb|mb|gb|tb` (binary multipliers, e.g.
    `mb` = 1024**2), not `MiB`/`GiB` suffixes.
- Anything in the task text or documentation that was unclear:
  - The README's transform list documents `matmul`'s `from_a`/`from_b`/`to`
    as ordinary references "may be regex or structured mappings" without
    stating that `from_b` and `to` are substitution templates derived from
    `from_a`'s match, rather than independently resolved references; this
    only became clear from the error message and reading the transform
    implementation.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~15 minutes
