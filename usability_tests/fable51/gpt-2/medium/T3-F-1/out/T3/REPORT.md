# T3 participant self-report (condition F)

- Final artifact path: `out/T3/solution.py` (output shards and `model.safetensors.index.json` in `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `wte.weight` (154 MB) exceeds the 64 MiB shard budget, so the sharder needs an explicit "oversized tensor alone in its own shard" rule.
  - Anchored regex on `h.<i>.(attn.c_attn|attn.c_proj|mlp.c_fc|mlp.c_proj).weight` avoids hitting embeddings, layer norms and biases; `h.<i>.attn.bias` is a mask buffer, matched separately and dropped.
- Anything in the task text or documentation that was unclear: shard file naming and the ordering of tensors into shards are unspecified; used HF-style `model-0000i-of-0000n.safetensors` names and greedy packing in original key order.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: `tensor.to(torch.bfloat16)` for the cast, `torch.equal` for value checks.
  - `safetensors` 0.5.3: `load_file` / `save_file` for reading the input and writing each shard.
  - Plain script rather than `transformers.save_pretrained(dtype=...)`, because that applies one dtype to the whole model and would also cast embeddings and norms, and it does not drop the causal-mask buffer selectively.
- Approximate time spent, if you can tell: about 2 minutes.
