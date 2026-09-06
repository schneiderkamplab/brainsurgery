# T2 self-report (condition P)

- **Final artifact path:** `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The Conv1D `[in, out]` layout means the head axis differs per tensor: columns (dim 1) for `c_attn.weight`, rows (dim 0) for `c_proj.weight` — using the same axis for both would have silently produced a wrong-but-loadable checkpoint.
  - `c_attn` is fused `[q | k | v]`, so head 5 has to be dropped three times, once inside each 768-wide segment, not once across the full 2304 width.
  - `c_attn.bias` follows the *columns* of `c_attn.weight`, so it uses the same 2112-long index list along dim 0.
  - `attn.c_proj.bias` and the `attn.bias` mask buffer are not per-head and must be left alone; a name-based match on `attn.bias` would wrongly catch the mask buffer, and a loose `c_proj` match would wrongly catch `mlp.c_proj`.
  - `index_select` on dim 1 yields a non-contiguous tensor, so `.contiguous()` before `save_file` avoids a save-time error.
- **Anything in the task text or documentation that was unclear:** nothing. The explicit kept-index ranges were unambiguous; I generated the indices from the head geometry and asserted them against the task's literal ranges as a cross-check.
- **Tools used (condition F):** n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- **Approximate time spent, if you can tell:** ~5 minutes.

## What the script does

1. Loads all 160 tensors, asserts the input tensor count.
2. Builds the keep-index list from head geometry (12 heads, 64 dims, drop head 5) for a 3-segment tensor (`c_attn`, 2112 indices) and a 1-segment tensor (`c_proj`, 704 indices), and asserts both equal the literal ranges given in TASK.md.
3. Per layer 0..11: asserts the input shapes, then `index_select` on dim 1 for `c_attn.weight`, dim 0 for `c_attn.bias`, dim 0 for `c_proj.weight`; asserts dtype is preserved.
4. Before writing: the four required checks, plus the same shape checks for all 12 layers, key-set identity with the input, identity (not just shape) of the 124 untouched tensors, and an order spot-check that kept block 6 of `c_attn` equals source columns 384..767 and kept block 6 of `c_proj` equals source rows 384..767.
5. Writes `out/T2/model.safetensors`, then reloads it and verifies count, key set, shapes, dtypes and bit-exact values against what was computed.
