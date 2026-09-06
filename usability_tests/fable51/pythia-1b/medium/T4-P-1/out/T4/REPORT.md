# T4 participant self-report (condition P)

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - None. The task vectors are each computed against the untouched base tensor (float32), summed, then cast back to float16; the base dict is never mutated.
  - Wrote a read-back check after saving to confirm 244 names and bit-exact shared tensors on disk.
- Anything in the task text or documentation that was unclear: nothing; the MLP tensor list and the formula were explicit.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
