# Participant self-report: T2 (condition P)

- Final artifact path: `out/T2/solution.py` (output `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: failed_assertion. A self-added check that every tensor is float16 fired; the attention buffers (bool mask, float32 inv_freq/masked_bias) are not float16. The check was beyond the spec and was removed. No output was written.
- Pitfalls or surprises you hit (one line each):
  - Not every tensor in a "float16 checkpoint" is float16; the non-parameter attention buffers keep their own dtypes and must pass through unchanged.
  - Indexing with a row/column index tensor yields a copy, but `.contiguous()` was kept for safety before `save_file`.
- Anything in the task text or documentation that was unclear: Nothing; the interleaved per-head row layout and the exact kept ranges were stated explicitly.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 2 minutes
