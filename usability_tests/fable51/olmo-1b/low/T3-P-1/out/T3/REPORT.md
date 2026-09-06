# Participant self-report: T3 (condition P)

- Final artifact path: `out/T3/solution.py` (output checkpoint in `out/T3/`, 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Used an anchored regex on `model.layers.<i>.(self_attn.[qkvo]_proj|mlp.(gate|up|down)_proj).weight` so embeddings and `lm_head` are not matched.
  - Shard packing is greedy in input tensor order; `embed_tokens` and `lm_head` exceed 256 MiB and are placed alone in their own shards.
- Anything in the task text or documentation that was unclear: shard file naming and the packing order are not specified; I assumed HF-style `model-XXXXX-of-XXXXX.safetensors` names and greedy fill in the original index order.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 2 minutes
