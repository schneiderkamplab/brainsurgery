# T5 self-report

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - PEFT name prefix: adapter keys carry `base_model.model.` and must be stripped to reach `model.layers.<i>....weight`.
  - The task text calls `model.embed_tokens.weight` / `lm_head.weight` "larger than" the 512 MiB budget, but each is 412,057,600 bytes (~393 MiB), so they are not oversized; greedy packing in sorted key order still places each first in its own shard region and every shard stays under the limit.
- Anything in the task text or documentation that was unclear: only the above shard-size wording; the exact shard file naming/partition of the hidden reference is unspecified, so I relied on the stated constraint (<= 512 MiB per shard, index `weight_map` complete).
- Tools used (condition F):
  - `torch` 2.14.0 — float32 matmul `scale * B @ A` and the tensor add.
  - `safetensors` 0.5.3 — reading the base shards and the adapter, writing the sharded output and the index.
  - I deliberately did not use `peft.merge_and_unload`: it requires instantiating the full OLMo model through `transformers`, is slower and heavier, and gives no control over the required shard layout/index; `mergekit` likewise wraps a model-level pipeline. A direct state-dict script keeps the operation at the checkpoint-file level, which is what the task asks for.
- Required checks enforced in `solution.py` before writing (each calls `die()` -> exit 1): exactly 32 merged adapter pairs, no `lora_` name in the output, `model.layers.0.self_attn.q_proj.weight` shape `[2048, 2048]`, exactly 114 output tensors; plus dtype, delta-shape and per-shard size checks.
- Approximate time spent, if you can tell: ~5 minutes.
