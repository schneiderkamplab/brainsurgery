# Participant self-report

- Final artifact path: `out/T4/plan.yaml` (produces `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `add`/`subtract` require a pre-existing destination, so the diff-then-add
    approach needs scratch tensors; rewrote the merge algebraically as
    `out = (1-2*lambda)*base + lambda*ft1 + lambda*ft2` and used `scale` +
    `add_` instead, which also sidesteps the ordering hazard entirely since
    every term reads the unmodified `base`/`ft1`/`ft2` directly.
  - Output alias inference requires every transform destination in the whole
    plan to land on one alias; scratch tensors had to be created under
    `base::` (e.g. `base::tmp0.<i>.<proj>`) rather than a separate alias, then
    deleted before the final tensor-count assert, or the run would fail with
    "cannot infer output model uniquely".
- Anything in the task text or documentation that was unclear: none; the
  README's note on output-alias inference and the `equal`/negative-lookahead
  pattern from the worked MoE example were enough to build the preflight
  checks directly.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: one plan draft, one execution, no
  retries.
