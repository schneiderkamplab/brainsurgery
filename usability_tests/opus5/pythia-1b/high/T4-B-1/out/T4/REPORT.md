# T4 self-report (condition B)

- **Final artifact path:** `out/T4/model.safetensors`, produced by `out/T4/plan.yaml`
  (run as `brainsurgery out/T4/plan.yaml`, default `inmemory` provider).
  `out/T4/check.yaml` is a separate, optional verification plan (no `output:`
  section, writes nothing) that recomputes the merge in a rearranged form and
  compares it against the written file.

- **Number of times you executed the script or plan:** 1 execution of
  `out/T4/plan.yaml`. Plus 1 execution of the independent `out/T4/check.yaml`
  verification plan afterwards.

- **Which executions failed, and why (one line each):** none; the plan
  succeeded on the first execution, and the verification plan passed as well.

- **Pitfalls or surprises you hit (one line each):**
  - The real hazard is the one the task names: `bs_b32` (the float32 base copy)
    is also the accumulator, so both `subtract_` steps have to run before either
    `add_` step, otherwise the second task vector would be taken against an
    already-merged base. I ordered the plan as cast x3 -> subtract_ x2 ->
    scale_ x2 -> add_ x2 rather than doing one fine-tune at a time.
  - Output alias inference forced a design choice: with three inputs, every
    write must land on one alias, so the float32 scratch tensors had to be
    created *inside* the `base` alias (prefixes `bs_b32.` / `bs_tv1.` /
    `bs_tv2.`) and deleted again before the checkpoint is written. Writing them
    to a fourth alias would have made the output ambiguous.
  - There is no assert operator that compares tensor *name sets* without also
    comparing values, so the step-1 name check had to be assembled: `equal`
    (which fails when a right-hand name is missing) covers the 180 non-MLP
    names exactly and bit-exactly; for the 64 MLP names, whose values legitimately
    differ, name identity falls out of `count` = 64 on each alias plus the
    `subtract_` steps, which rewrite a `bs_b32.<name>` (base-derived) into a
    `bs_tv1.<name>` / `bs_tv2.<name>` (fine-tune-derived) destination that must
    already exist. That is 64 injective mappings into a 64-element set, i.e. set
    equality, and it fails loudly if any MLP name differs.
  - `(?!.*\.mlp\.).*` is the whole non-MLP selector, and it is worth pinning
    with `count: 180` in the same assert block: a lookahead typo would silently
    select nothing and make the `equal` check vacuous rather than failing.
  - `attention.bias` is stored as `U8` and `rotary_emb.inv_freq` as `F16`; both
    are in the untouched 180, so leaving the non-MLP tensors strictly alone
    (rather than casting anything wholesale) keeps them bit-exact for free.
  - `assign` (destination must exist, dtype must match) is the right final step
    instead of `delete` + `move`: it writes the merged values back over the base
    MLP tensors without disturbing names or state-dict order.
  - Inspecting the inputs did not need the tool at all: the safetensors header is
    a JSON blob after an 8-byte length prefix, so `od` + `dd` + `grep` was enough
    to confirm the three name sets are identical and to read every dtype and
    shape before writing a single transform.

- **Anything in the task text or documentation that was unclear:**
  - The docs do not state explicitly whether a `cast` (or `copy`) destination may
    name a *different* alias than the source. The README only shows it for `copy`
    in the output-alias note. It does work for `cast`, but I was guessing.
  - `assert: { count: { of: <pattern>, is: 0 } }` — the docs do not say whether a
    reference that matches nothing is a legal count of zero or an error. I dropped
    the "no scratch tensors left" check rather than risk it; it is implied anyway
    by 244 = 180 + 64.
  - `assert: equal`'s `eps` is documented as an absolute tolerance while the task
    grades on relative Frobenius error, so the two are not directly comparable;
    I picked an absolute `eps` (1e-3) for the verification plan by reasoning about
    the fp16 weight magnitudes rather than from the grading criterion.
  - The task says "computed in float32, then cast back to float16" but grades with
    a tolerance that admits reassociation, so it is not obvious whether the literal
    expression is required. I implemented it literally in the plan and used the
    algebraically equivalent `0.2*base + 0.4*ft1 + 0.4*ft2` only in the separate
    verification plan, precisely so the check is not a restatement of the solution.
  - Minor: `docpack/README.md` links to `/Users/petersk/...` absolute paths that do
    not exist in the sandbox, so the deeper docs it points at are unreachable.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** roughly 5 minutes wall clock;
  about 11 s for the merge run and about 8 s for the verification run.
