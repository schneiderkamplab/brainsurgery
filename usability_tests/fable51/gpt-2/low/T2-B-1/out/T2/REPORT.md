# Participant self-report: T2 (GPT-2, condition B)

- Final artifact path: `out/T2/plan.yaml` (output checkpoint: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: plan-load error (`crash`): `output.shard must be a non-empty string when provided`; I had written `shard: false` to force a single file. Removed the key; a `.safetensors` file path already writes a single file.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each source to resolve to exactly one tensor, so the per-layer edit cannot be written once with a `$i` / regex pattern; the plan repeats the 3 concat + 3 delete + 3 move block for each of the 12 layers (generated with a shell loop).
  - No "slice in place" or "keep columns" transform: pruning is expressed as concat of slices into a temp name, delete the original, then move the temp back to the original name.
  - The README says `shard` can be disabled, but the accepted way to disable it is not stated; a boolean is rejected.
- Anything in the task text or documentation that was unclear:
  - How to force a single unsharded safetensors file from the `output` mapping (turned out: give a file path and omit `shard`).
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes
