# T3 participant self-report

- **Final artifact path:** `out/T3/solution.py` (output: `out/T3/model-0000{1..4}-of-00004.safetensors` + `out/T3/model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none — the first execution passed all checks.
- **Pitfalls or surprises you hit (one line each):**
  - `.*weight` would have swept in `wte.weight`, `wpe.weight` and all 48 layer-norm weights, so I built the 48 cast targets as an explicit name list (`h.<i>.{attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj}.weight`) instead of matching by pattern.
  - `h.<i>.attn.bias` is the causal-mask buffer, not a bias parameter — a name filter on `attn.bias` must not be confused with `attn.c_attn.bias` / `attn.c_proj.bias`, which are parameters and stay float32.
  - The oversized-tensor case is real: `wte.weight` is 154,389,504 bytes, well over the 64 MiB budget, so plain greedy packing has to special-case it into a shard of its own rather than just overflowing a shard.
  - I asserted that every targeted name actually exists in the input before transforming; a silent no-match here produces an output that looks plausible (right tensor count) but has the wrong dtypes.
- **Anything in the task text or documentation that was unclear:**
  - The grading line says it compares "sharding rules, exact key set, shapes, dtypes and bit-exact values". It is not stated whether the *shard assignment itself* must match the hidden reference file-for-file (same tensor→shard mapping and same filenames), or only whether the stated rules hold. I assumed the latter and used the conventional HuggingFace layout: greedy packing in sorted key order, `model-{i:05d}-of-{n:05d}.safetensors`, index with `metadata.total_size` and `weight_map`. That yields 4 shards (66,259,968 / 66,256,896 / 40,983,552 bytes, then `wte.weight` alone at 154,389,504). A different but equally rule-conforming packing order would produce a different mapping.
  - The task fixes the index filename (`model.safetensors.index.json`) but not the shard filename pattern; I picked the HF convention.
  - "round-to-nearest-even, as `tensor.to(torch.bfloat16)` does" made the cast unambiguous — useful, since a manual bit-truncation would differ.
- **Tools used (condition F):** n/a — condition P (torch 2.14.0 + safetensors 0.5.3 only).
- **Approximate time spent, if you can tell:** ~5 minutes: read the safetensors header directly from the shell to confirm the 160 names and that everything is float32, then wrote and ran the script once.
