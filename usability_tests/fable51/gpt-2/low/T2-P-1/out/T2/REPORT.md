# Participant self-report: T2 (GPT-2 124M, condition P)

- Final artifact path: `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Conv1D `[in, out]` layout means head slices are columns on `c_attn` and rows on `c_proj`; the task text spelled out the exact index ranges, so no discovery was needed.
  - Fancy-indexed slices are copied; `.contiguous()` added anyway before `save_file`.
- Anything in the task text or documentation that was unclear: nothing; index ranges and shapes were explicit.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 2 minutes
