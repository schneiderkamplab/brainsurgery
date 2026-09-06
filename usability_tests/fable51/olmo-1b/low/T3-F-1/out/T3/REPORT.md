# T3 self-report (condition F)

- Final artifact path: `out/T3/solution.py` (output checkpoint in `out/T3/`, 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Avoided the `.*weight` overmatch by anchoring a regex to `model.layers.<i>.{self_attn.{q,k,v,o}_proj,mlp.{gate,up,down}_proj}.weight` and asserting exactly 112 matches.
  - Embeddings and lm_head (412 MB each) exceed the 256 MiB budget, so the packer places any oversized tensor alone in its own shard; the greedy packer follows the input index order.
- Anything in the task text or documentation that was unclear: whether the grader cares about shard file names or grouping order; I used the HF `model-XXXXX-of-NNNNN.safetensors` naming and index-file order.
- Tools used (condition F): torch 2.14.0 (dtype cast, equality checks) and safetensors 0.5.3 (`safe_open`, `save_file`); plain script, because the task is a dtype cast plus custom sharding rules and transformers' `save_pretrained` does not support per-tensor dtypes or the "oversized tensor alone" rule directly.
- Approximate time spent, if you can tell: about 2 minutes.
