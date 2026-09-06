# Participant self-report: T3 (OLMo-1B-0724-hf, condition P)

- Final artifact path: `out/T3/solution.py` (output checkpoint in `out/T3/`: 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Targeting by an anchored regex over `model.layers.<i>.(self_attn.{q,k,v,o}_proj|mlp.{gate,up,down}_proj).weight` avoided the `.*weight` overmatch onto `embed_tokens` / `lm_head`; asserted the match count is exactly 112.
  - Shard packing: greedy in index order with a 256 MiB budget; 14 bf16 projection matrices per layer-pair fill a shard exactly (268,435,456 bytes), so the boundary must be `<=`, not `<`.
  - The two 412 MB float32 embedding/lm_head tensors exceed the budget and were placed alone in their own shards.
- Anything in the task text or documentation that was unclear: whether the grader expects a specific shard file naming or packing order; I used HF-style `model-0000i-of-0000n.safetensors` and index order.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~3 minutes
