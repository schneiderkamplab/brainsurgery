## Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the plan succeeded on the first run.
- Pitfalls or surprises you hit (one line each):
  - With two inputs (`model`, `lora`) there is no default alias, so every tensor reference in the plan needs an explicit `model::`/`lora::` prefix, including inside regex rewrite targets — omitting it fails plan compilation with "missing model alias in reference".
  - `assert: count: { is: 0 }` raises "matched zero tensors" instead of succeeding, because `count` treats a zero-match resolve as an error; used `assert: { not: { exists: ... } }` to check "no `lora_` tensor remains" instead.
  - `matmul`/`permute`/`scale`/`add_` all support one regex sweep across all 12 layers via capture-group rewriting (`\1` in `from_b`/`to`), so no per-layer transform repetition was needed.
- Anything in the task text or documentation that was unclear: none; the Conv1D-vs-Linear layout note and the `fan_in_fan_out` explanation in TASK.md were enough to get the transpose direction right on the first try.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~15 minutes.
