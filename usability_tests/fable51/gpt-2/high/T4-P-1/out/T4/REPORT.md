# Participant self-report: T4 (GPT-2 124M, condition P)

- Final artifact path: `out/T4/solution.py` (output checkpoint: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Compared non-MLP tensors on their raw int32 bit patterns instead of `torch.equal` so a NaN would not be treated as a mismatch against itself.
  - Computed both task vectors from the untouched `base` dict and only wrote results into a separate `out` dict, so the second task vector could not see the first merge.
  - Made every output tensor `.contiguous()` before `save_file` as a precaution; the inputs were already contiguous.
- Anything in the task text or documentation that was unclear: nothing material; the task text was explicit about names, counts, lambda, and grading tolerance.
- Tools used (condition F): not applicable (condition P: torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: about 2 minutes.
