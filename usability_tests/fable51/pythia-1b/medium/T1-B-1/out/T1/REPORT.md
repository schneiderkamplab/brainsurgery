# Participant self-report: T1 (Pythia-1B depth pruning), condition B

- Final artifact path: `out/T1/plan.yaml` (output checkpoint `out/T1/model.safetensors`, 184 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed every assert and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - Regex references are full-match, so `layers\.1\.` style anchoring with the escaped trailing dot was needed to keep `1` from matching `10`..`15`.
  - Renumbering must be done in ascending source order (3->2, 4->3, ...) because `move` refuses existing destinations; that ordering guarantees each destination is already vacated.
  - `move` with a regex `from` and `\1` in `to` renames a whole block in one transform, so no per-tensor enumeration was needed.
- Anything in the task text or documentation that was unclear:
  - The README documents structured (list) patterns for `copy`/`move` but the help text only shows literal examples; that regex capture groups work in `move: to` had to be inferred from the `assert.equal` description ("like 'to' in copy/move").
  - No documented assert for "count of matches is zero"; used `not: { exists: ... }` instead.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes
