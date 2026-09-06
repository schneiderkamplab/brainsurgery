# T4 participant self-report

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each): none; a plain script was simpler than a mergekit task-arithmetic config because the shared-tensor precondition check and the "exactly 64 merged" check must be enforced explicitly, which mergekit does not do.
- Anything in the task text or documentation that was unclear: nothing
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: load the three checkpoints and save the output.
  - `torch` 2.14.0: equality check of shared tensors, float32 arithmetic, cast back to float16.
- Approximate time spent, if you can tell: about 2 minutes
