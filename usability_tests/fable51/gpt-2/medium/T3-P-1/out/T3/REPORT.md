# Participant self-report: T3 (GPT-2 124M), condition P

- Final artifact path: `out/T3/solution.py` (output: `out/T3/model-0000{1..4}-of-00004.safetensors` + `out/T3/model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `h.<i>.attn.bias` is a mask buffer, not a bias parameter, so the drop pattern had to be anchored to `attn.bias` exactly and must not touch `attn.c_attn.bias` / `attn.c_proj.bias`.
  - Projection selection used a fully anchored regex on the four module names so `wte`, `wpe`, `ln_*` and `ln_f` weights stay float32.
  - `wte.weight` (154 MB) exceeds the 64 MiB shard budget; greedy in-order packing puts it alone in the last shard since it sorts last alphabetically.
- Anything in the task text or documentation that was unclear:
  - Shard file naming and packing order are not specified; I used the HuggingFace convention (`model-NNNNN-of-NNNNN.safetensors`, greedy fill in key order, `metadata.total_size` in the index).
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
