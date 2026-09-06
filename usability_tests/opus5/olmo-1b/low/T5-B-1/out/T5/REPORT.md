# T5 self-report

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Output alias inference: the deltas had to be created on the `model` alias (not on `lora::`), otherwise transforms would write to two aliases and the run could not infer which model to save.
  - PEFT name prefix `base_model.model.model.` had to be rewritten to the base `model.` names via regex capture groups.
  - Regex dots must be escaped on the source side; the `to`/`from_b` rewrite side is a plain replacement template using `\1`, `\2`.
  - Shard budget is binary: `512MB` = 512 MiB, and the 412 MB embedding/lm_head tensors land alone in their own shards as required.
- Anything in the task text or documentation that was unclear:
  - The README does not state explicitly that `matmul`'s `from_b` is resolved as a rewrite of the `from_a` match (only `assert: equal`'s `right` and `copy`'s `to` are documented that way); it does work, but I had to infer it.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes.
