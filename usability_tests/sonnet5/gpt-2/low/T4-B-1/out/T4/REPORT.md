# Participant self-report

- Final artifact path: out/T4/plan.yaml
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `crash` — plan failed to compile with `unknown model alias: 'tmp_base'`; scratch
    tensors need their alias declared (via `prefixes: { mode: add, ... }`) before they can be
    written to, or, better, they need to live under an existing alias.
- Pitfalls or surprises you hit (one line each):
  - Output-model inference requires every transform in the plan to write to a single alias;
    creating scratch aliases for the scale-by-lambda intermediates (`tmp_base`/`tmp_ft1`/`tmp_ft2`)
    made the destination alias ambiguous even after registering them with `prefixes`. Fixed by
    naming the scratch tensors under the `base` alias itself (`base::tmp1_<name>` etc.) and
    deleting them before saving.
  - The arithmetic `base + lambda*(ft1-base) + lambda*(ft2-base)` simplifies to
    `(1-2*lambda)*base + lambda*ft1 + lambda*ft2`, which sidesteps the base-vs-already-merged-base
    ordering hazard entirely: each term reads only original, untouched inputs.
  - `assert: { equal: ... }` compares values with a hard equality unless `eps` is given; used a
    very large `eps` (1e30) to turn it into a pure name/shape/dtype existence check for the 48 MLP
    tensors, since their values are legitimately expected to differ between checkpoints.
- Anything in the task text or documentation that was unclear:
  - None; the README section on output-model inference and the single-destination-alias rule was
    easy to miss on first read but is spelled out.
- Tools used (condition F): name, version, and why: n/a (condition B)
- Approximate time spent, if you can tell: ~10 minutes
