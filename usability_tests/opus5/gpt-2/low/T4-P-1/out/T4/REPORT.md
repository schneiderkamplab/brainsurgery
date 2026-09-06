# T4 participant self-report

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The hazard flagged in the task is real: the two task vectors must both be taken
    against the untouched base, so I build the merged tensor from the original
    `base[key]` in one expression instead of accumulating in place.
  - `inputs/` also contains a `lora/` directory that this task does not use; ignored it.
  - GPT-2 uses Conv1D layout, so `c_fc.weight` is `[768, 3072]` and `c_proj.weight`
    is `[3072, 768]`; irrelevant to elementwise arithmetic but worth not "fixing".
- Anything in the task text or documentation that was unclear:
  - Step 1 says "identical"; I read that as bit-exact (`torch.equal`) rather than
    within a tolerance, which matches the grader's bit-exact check on the 112
    unchanged tensors.
- Tools used (condition F): n/a (condition P: torch + safetensors only).
- Approximate time spent, if you can tell: ~5 minutes.
