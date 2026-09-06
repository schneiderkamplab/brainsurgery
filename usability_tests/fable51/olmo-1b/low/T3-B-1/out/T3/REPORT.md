# Participant self-report: T3 (olmo-1b, condition B)

- Final artifact path: `out/T3/plan.yaml` (output checkpoint in `out/T3/`, 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `assert: count` counts by name pattern, not by dtype, so "exactly 112 tensors are bfloat16" was expressed as count(projection pattern)=112 plus dtype=bfloat16 on that pattern plus dtype=float32 on its negative-lookahead complement.
  - The output directory `out/T3/` also holds `plan.yaml` and `REPORT.md` next to the shard files, as the task instructions require.
- Anything in the task text or documentation that was unclear:
  - Whether `assert: dtype` with a multi-match pattern checks every match; behaviour observed is consistent with that.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
