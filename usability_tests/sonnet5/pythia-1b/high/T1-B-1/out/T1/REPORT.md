## Participant self-report

- Final artifact path: `out/T1/plan.yaml` (executed to produce `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 2 (first run failed to
  compile; second run succeeded and produced the output)
- Which executions failed, and why (one line each):
  1. `crash`: plan failed to compile — used `exists: { of: <pattern> }` for the
     `not`/`exists` assert, but `assert.exists`'s payload is a bare tensor-ref
     (string or list), not a mapping with an `of` key; fixed by passing the
     pattern directly as the value of `exists`.
- Pitfalls or surprises you hit (one line each):
  - `assert.exists` takes a raw tensor-ref payload while `assert.count` takes
    a mapping with `of`/`is` — the two are not symmetric, so it's easy to
    guess the wrong shape for `exists`.
  - The renumbering has no closed-form arithmetic (removed indices 2, 6, 10,
    14 are interspersed, so the shift is a step function of the old index,
    not a constant offset), and brainsurgery's capture-group rewrite only
    does literal string interpolation, not arithmetic on captured digits — so
    each surviving block needed its own explicit `move` with literal old/new
    numbers rather than one generic pattern.
  - Collision hazard: moves were ordered by ascending old block index. That
    guarantees every destination slot is empty when its move runs — either it
    was deleted up front (2, 6, 10, 14) or vacated by an earlier move in the
    same ascending pass (e.g. `4->3` only runs after `3->2` has already moved
    old block 3 out of the way) — so no move ever overwrote a still-live
    tensor. Doing the moves in a different order (e.g. descending, or grouped
    by destination) would have hit `move` destination-already-exists errors
    or, worse, silently apply in an order where an unlucky sequence could
    overwrite a not-yet-moved block if the checks weren't in place.
  - Anchoring matters: patterns like `gpt_neox\.layers\.1\.(.*)` must fully
    match the tensor name (which brainsurgery guarantees via full-match
    semantics), otherwise a pattern for block 1 could spuriously match block
    11 or 12. Used specific literal block numbers per move (never a shared
    single-digit alternation) and relied on full-match anchoring rather than
    substring search.
- Anything in the task text or documentation that was unclear:
  - The `help.txt` snippet for `assert.exists` just says `Payload:
    tensor-ref` without an inline example, unlike most other assert entries
    (e.g. `assert.equal`) that show one; had to infer the correct shape from
    the general "Tensor references" section of the README plus the error
    message from the failed compile.
- Tools used (condition F): n/a (condition B, brainsurgery plan only)
- Approximate time spent, if you can tell: ~15 minutes (mostly planning the
  block mapping and verifying no collisions; the plan itself compiled and ran
  successfully on the second try)
