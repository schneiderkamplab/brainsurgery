# Participant self-report: T4 (GPT-2 124M), condition B

- Final artifact path: `out/T4/plan.yaml` (output checkpoint `out/T4/model.safetensors`, 160 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The output alias is inferred from write destinations, so scaled task-vector copies had to be created inside the `base` alias (as `base::tv1.*` / `base::tv2.*`) and deleted afterwards, rather than in the `ft1`/`ft2` aliases.
  - `add`/`add_` only write into existing tensors and there is no fused scaled add, so the merge was rewritten as `0.2*base + 0.4*ft1 + 0.4*ft2` using `scale`, `scale_` and `add_` (the fine-tune terms are computed from the untouched inputs, so ordering is safe).
  - `equal` with a negative-lookahead regex on `left` and `\g<0>` on `right` was the only way to express "all non-MLP tensors are identical across aliases"; the README example covering exactly this case was helpful.
- Anything in the task text or documentation that was unclear:
  - Whether the built-in `help` text for `scale` uses single or double backslashes in `'scaled.\\g<0>'` (the YAML single-quote form needs one backslash); resolved by trying it.
  - "Exactly 48 tensors were merged" cannot be asserted directly on write counts without an instrumented backend, so it is checked via `count` on the MLP pattern and on the temporary task-vector tensors (48 each) plus a `not: equal` check that the MLP tensors changed.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
