# Participant self-report: T4 (condition P)

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The base is sharded; loaded both shards via `model.safetensors.index.json` and cross-checked the index against the shard contents.
  - Computed each task vector against the untouched base tensor rather than a running merged value, per the ordering hazard in the task text.
  - Used `torch.equal` (bit-exact) for the non-MLP verification rather than a tolerance.
- Anything in the task text or documentation that was unclear: nothing significant; the spec was explicit about names, shapes, lambda and output path.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: a few minutes; one script write and one run.
