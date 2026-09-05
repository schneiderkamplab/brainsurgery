## Participant self-report

- Final artifact path: `out/T4/plan.yaml`
- Number of times you executed the script or plan: 2 (first attempt failed to
  compile because it referenced a scratch model alias, `work`, that was never
  declared in `inputs`; fixed by keeping scratch tensors under the `base`
  alias with a `scratch.` name prefix and deleting them before save).
- Which executions failed, and why (one line each):
  - Execution 1: `PlanLoaderError: unknown model alias: 'work'` — plan
    aliases must come from `inputs` (or be produced as a transform's `to`
    that the loader recognizes); an ad-hoc alias like `work::` used as
    scratch space is not auto-registered, so scratch tensors were rewritten
    into the already-known `base` alias under `scratch.*` names instead.
- Pitfalls or surprises you hit (one line each):
  - `add`/`subtract`/`multiply` require the destination to already exist (they
    do not create new tensors), so building the two task vectors and the
    running merge needed a `copy` first to seed a same-shape/dtype
    placeholder, then `subtract`/`add_` to fill it in.
  - Bulk pairwise ops (`equal`, `add`, `subtract`, `copy`, `cast`, `assign`)
    resolve the right-hand/destination name by regex *substitution* of the
    left/source match, not by independently matching the same pattern on the
    other side — `\g<0>` (whole match) or an explicit `(...)` + `\1` group is
    needed to carry the original tensor name across aliases/scratch prefixes.
  - Scratch tensors used during the merge must be deleted before the final
    `save`, since `save` (and the count assertions) act on the whole alias's
    tensor set, not just the originally-loaded tensors.
- Anything in the task text or documentation that was unclear:
  - The doc pack doesn't say which field of `output`/`save` picks a model
    alias when multiple checkpoints are loaded (`output:` has no alias key at
    all); using the `save` transform with `alias: base` instead of top-level
    `output:` worked but this wasn't obvious from the README alone.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~20 minutes.
