# Participant self-report: T3 (Pythia-1B), condition B

- Final artifact path: `out/T3/plan.yaml` (output checkpoint: `out/T3/model-0000{1..9}-of-00009.safetensors` + `out/T3/model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The "exactly 64 tensors are bfloat16" check has no direct dtype-count operator; expressed it as count-of-projection-pattern == 64, dtype bfloat16 on that pattern, and dtype float32 on the regex complement (negative lookahead) so any stray bfloat16 tensor fails the run.
  - Ordering matters: delete the buffers before the blanket `cast_ '.*' -> float32`, otherwise the uint8 causal mask and the scalar buffer would be cast too (harmless since they are deleted, but wasteful).
  - The embeddings become 412 MB in float32 (task text quotes their 206 MB float16 size), so each still lands alone in its own shard under the 256 MiB budget.
- Anything in the task text or documentation that was unclear:
  - Whether `assert.dtype` with a multi-match pattern checks every match; it does (the complement pattern passed on 132 tensors, and would have failed had any been non-float32).
  - README does not state that `output.path` pointing at an existing directory writes `model-XXXXX-of-XXXXX.safetensors` into it; inferred from the shard section and confirmed by the run.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
