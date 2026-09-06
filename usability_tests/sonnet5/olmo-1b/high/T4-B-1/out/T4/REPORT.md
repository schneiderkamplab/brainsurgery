# Participant self-report

- Final artifact path: `out/T4/plan.yaml` (output written to `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `add`/`subtract`/`scale`/`add_` resolve their multi-tensor batch mapping by
    running `re.sub` with the *first* ref's regex against the matched name, so
    `from`/`to` (or `from_a`/`from_b`/`to`) must be substitution templates
    (`\1`, `\2`, `\g<0>`) sharing capture groups with the driving ref, not
    independent regexes to be matched separately.
  - Output-alias inference counts every destination (including a `scale`
    landing pad and any `delete`) as a "write", and errors out if more than
    one alias receives writes. That rules out using a disposable scratch
    *alias* for intermediate task-vector tensors; the scratch tensors had to
    live under the same `base` alias as the real output (as new tensor names,
    later deleted) so the whole plan writes to exactly one alias.
  - Rather than literally compute `ft1 - base` and `ft2 - base` as separate
    steps (which risks the classic ordering hazard if `base` gets mutated
    in-place before both diffs are taken), I used the algebraically
    equivalent form `out = (1-2*lambda)*base + lambda*ft1 + lambda*ft2`. This
    sidesteps the ordering hazard by construction: the two `scale` calls that
    read `ft1`/`ft2` never touch `base`, so it does not matter that `base`'s
    own in-place `scale_` runs before or after them.
- Anything in the task text or documentation that was unclear: no; the
  `assert.equal` "right is a rewrite of each left match" example in the
  README/help pack, plus the negative-lookahead example given there, mapped
  directly onto the "identical outside MLP" precondition check.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: one exploration pass (reading
  docpack + inspecting installed package source to confirm regex-rewrite
  semantics of add/scale/add_) followed by a single successful plan run and a
  standalone numeric verification against `inputs/` recomputed independently;
  no iteration needed on the plan itself.
