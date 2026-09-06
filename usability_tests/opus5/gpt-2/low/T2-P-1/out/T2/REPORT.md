# T2 self-report (condition P)

- Final artifact path: `out/T2/model.safetensors` (script: `out/T2/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded
- Pitfalls or surprises you hit (one line each):
  - GPT-2 Conv1D layout is `[in, out]`, so heads are columns of `c_attn.weight` but rows of `c_proj.weight`; the axes differ between the two tensors.
  - `c_attn` is fused `[q|k|v]`, so head 5 has to be dropped three times, at offsets 0, 768 and 1536, with the segment order preserved.
  - `attn.bias` is a mask buffer, not a projection bias; it must be left alone despite the name resembling a head-bearing tensor.
  - `index_select` returns a fresh tensor, but I added `.contiguous()` anyway to avoid safetensors complaining about storage sharing/strides.
- Anything in the task text or documentation that was unclear: nothing; the explicit index ranges in "Required result" removed all ambiguity.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: ~5 minutes
