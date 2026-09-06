# T2 participant self-report (condition P)

- **Final artifact path:** `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - Input is sharded (two shards + index), so the script merges shards and cross-checks the merged key set against `weight_map` before touching anything.
  - Fancy indexing returns a fresh tensor, but untouched tensors are views into the memory-mapped shards; I `.contiguous().clone()` everything so `safetensors.save_file` never sees shared/mmapped storage (OLMo also has both `lm_head.weight` and `model.embed_tokens.weight` present, a classic shared-tensor trap).
  - Head 5 is rows/cols 640..767; the direction differs per tensor (rows for q/k/v, columns for o_proj) — I keyed that off regexes anchored with `^...$` and escaped dots so `mlp.*_proj` can never match.
  - Added a count check (48 row-pruned, 16 col-pruned) so a silent no-match or overmatch would fail loudly rather than produce a 114-tensor file with untouched layers.
- **Anything in the task text or documentation that was unclear:** nothing; the kept-row ranges and target shapes were stated explicitly. The only thing not stated is whether the output should also carry an index/config — the task says a single `out/T2/model.safetensors`, so I wrote only that.
- **Tools used (condition F):** n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- **Approximate time spent, if you can tell:** ~5 minutes.
