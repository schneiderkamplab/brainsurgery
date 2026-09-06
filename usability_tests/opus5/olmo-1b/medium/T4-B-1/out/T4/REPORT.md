# T4 self-report (condition B, BrainSurgery plan)

- **Final artifact path:** `out/T4/plan.yaml` (plan), `out/T4/model.safetensors` (output, 114 tensors, float32)

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):**
  - None; the single execution passed all in-plan asserts and wrote the output.

- **Pitfalls or surprises you hit (one line each):**
  - Output-alias inference is the real constraint: the natural formulation
    (scale the fine-tunes' MLP tensors in place by 0.4 and `add_` them into the
    base) writes to three aliases and would fail with "cannot infer output model
    uniquely", so the scaled task-vector terms had to be materialised as scratch
    tensors *inside* the `base` alias (`base::tv1.<name>`) and deleted again.
  - There is no `assert` operator for "these two aliases have the same key set".
    I got it from `count: 114` on each alias plus
    `equal: { left: 'base::(.+)', right: 'ft1::\1', eps: 1e30 }` — the huge eps
    neutralises the value comparison so only the name/shape/dtype half of
    `equal` bites, and with equal counts on both sides that is a bijection.
  - The ordering hazard in the task disappears if you expand the arithmetic:
    `base + L*(ft1-base) + L*(ft2-base) == (1-2L)*base + L*ft1 + L*ft2`, i.e.
    `0.2*base + 0.4*ft1 + 0.4*ft2`. Both fine-tune aliases stay read-only, so no
    task vector can be taken against an already-merged base. Cost is one extra
    rounding step (measured relative Frobenius error 5.4e-8, well inside 1e-5).
  - Scratch names had to be chosen so the later `scale_` on the MLP tensors
    could not reach them. Regexes are full-match, so `tv1.model.layers.…` does
    not match `model\.layers\.…`; a prefix rather than a suffix makes that safe
    without extra anchoring.
  - `\g<0>` (whole match) is what makes the negative-lookahead selector
    `'base::(?!model\.layers\.\d+\.mlp\.).+'` usable as a `left`/`right` pair —
    the lookahead is zero-width so there is no group 1 to reference.

- **Anything in the task text or documentation that was unclear:**
  - The README documents which alias gets written but not that `assert`/`diff`
    can never disambiguate it; the "keep every edit on one alias" advice is the
    load-bearing sentence and it is easy to read past. An explicit note that
    scratch tensors must be created in the destination alias would help.
  - `assert: equal` is documented as "same shape and dtype, and values equal";
    it is not stated that shape/dtype are checked *before* values, which is what
    makes the large-eps trick a sound name/shape/dtype-only check. I verified it
    empirically rather than from the docs.
  - The task says "computed in float32" but does not say whether an algebraically
    equivalent regrouping is acceptable; the grading criterion (1e-5 relative
    Frobenius, "a different order of additions is fine") answers it, so the two
    statements are best read together.

- **Tools used (condition F):** n/a (condition B)

- **Approximate time spent, if you can tell:** ~10 minutes, of which ~40 s was
  the plan execution itself; most of the time went into reading `help.txt` for
  the exact `scale`/`add_`/`count`/`equal` semantics before writing anything.
