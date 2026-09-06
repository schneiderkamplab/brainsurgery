# T2 participant self-report

- **Final artifact path:** `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - GPT-2's Conv1D `[in, out]` layout means the fused `c_attn` heads are column blocks and the `c_proj` heads are row blocks — the two head-bearing tensors are sliced on opposite axes, which is the easy thing to get backwards.
  - `c_attn` is fused `[q | k | v]`, so head 5 has to be dropped three times (at columns 320, 1088 and 1856), not once; I generated the keep-index from a segment loop and asserted it equals the literal ranges in the task to rule off-by-one errors out.
  - `attn.bias` is the causal mask buffer, not a projection bias, so a naive name match on `attn.*bias` would wrongly hit it (and `mlp.c_proj.weight` would be hit by a loose `c_proj` match) — I matched on the exact dotted path structure instead.
  - Slices from `index_select` needed `.contiguous().clone()` before `save_file` to avoid safetensors complaining about shared/non-contiguous storage.
  - I preserved the source file's `__metadata__` so the output header matches the input.
- **Anything in the task text or documentation that was unclear:** nothing. The explicit column/row ranges made the layout unambiguous and gave me something to assert my computed index against.
- **Tools used (condition F):** n/a (condition P).
- **Approximate time spent, if you can tell:** ~5 minutes: read the task and inspect the sandbox, write the script, one run.
