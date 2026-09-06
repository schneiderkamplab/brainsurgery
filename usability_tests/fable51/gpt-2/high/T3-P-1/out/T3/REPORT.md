# T3 participant self-report (condition P, GPT-2 124M)

- Final artifact path: `out/T3/solution.py` (output checkpoint in `out/T3/`: 4 shards `model-0000N-of-00004.safetensors` plus `model.safetensors.index.json`, 148 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution passed all pre-write and post-write checks.
- Pitfalls or surprises you hit (one line each):
  - Targeting by anchored regex on the four exact projection names (`h.<i>.{attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj}.weight`) rather than a `.*weight` pattern, so embeddings and layer norms were never touched.
  - The causal-mask buffer is named `h.<i>.attn.bias`, which looks like a parameter bias; it needed its own anchored pattern so `attn.c_attn.bias` / `attn.c_proj.bias` were kept.
  - `wte.weight` (154 MB) exceeds the 64 MiB budget, so the sharder needed an explicit "oversized tensor goes alone" rule; greedy packing in key order put it in the last shard by itself.
- Anything in the task text or documentation that was unclear: the shard file naming scheme is not specified; I used the HuggingFace convention `model-XXXXX-of-YYYYY.safetensors`. Also unspecified whether the index `metadata.total_size` is required; I included it.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 3 minutes.
