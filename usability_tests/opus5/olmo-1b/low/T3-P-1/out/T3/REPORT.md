# T3 self-report (condition P)

- Final artifact path: `out/T3/solution.py` (output in `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why: none.
- Pitfalls or surprises you hit:
  - Over-broad targeting is the trap: I used an anchored regex
    `^model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$`
    so `model.embed_tokens.weight` / `lm_head.weight` / norms cannot match.
  - The two 412 MB embedding/lm_head tensors exceed the 256 MiB shard budget,
    so the greedy packer has to allow a single oversized tensor in its own
    shard rather than erroring; the "start a new shard only if the current one
    is non-empty" condition handles both cases.
  - Shard assignment order matters for matching a reference: the input index
    lists keys in sorted order, so I sharded in sorted key order too.
- Anything unclear: the task does not state the shard file naming scheme or the
  ordering used to assign tensors to shards; I inferred the HF convention
  (`model-0000i-of-0000n.safetensors`, sorted key order, greedy fill).
- Tools used (condition F): n/a
- Approximate time spent: ~5 minutes.
