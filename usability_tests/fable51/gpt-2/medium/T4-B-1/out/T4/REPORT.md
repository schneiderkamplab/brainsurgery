# Participant self-report: T4 (GPT-2 124M), condition B

- Final artifact path: `out/T4/plan.yaml` (output checkpoint `out/T4/model.safetensors`, 160 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `subtract`/`add` require the destination to already exist, so the task vectors were built as `copy` + in-place `subtract_` + `scale_` on temporary names (`tv1.*`, `tv2.*`) inside the `base` alias.
  - With several inputs the output alias is inferred from write targets, so every edit (including temporaries and `delete`) had to be on `base::` to keep the output unambiguous.
  - Both task vectors are computed before any `add_` touches `base`, which avoids the ordering hazard; the plan asserts 48 temporaries per fine-tune before merging.
- Anything in the task text or documentation that was unclear:
  - `help.txt` does not state explicitly that in-place binary transforms (`subtract_`, `add_`) rewrite `to` from `from` capture groups like `copy` does; it worked, but I had to assume it from the README.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes
