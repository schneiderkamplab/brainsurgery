# T3 participant self-report (condition F)

- Final artifact path: `out/T3/solution.py` (run as `.venv/bin/python out/T3/solution.py` from the sandbox root)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The task text quotes the embedding matrices as 206 MB; that is their float16 size. After the required upcast they are 412 MB each, still over the 256 MiB budget, so each still lands alone in its own shard.
  - `safetensors.load_file` returns keys sorted lexicographically, so `embed_out.weight` sorts before `gpt_neox.*` and becomes shard 1; the index `weight_map` handles this regardless of order.
- Anything in the task text or documentation that was unclear: nothing material.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: dtype casts (`to(torch.bfloat16)` / `to(torch.float32)`) and post-write bit-exact comparison.
  - `safetensors` 0.5.3: `load_file` / `save_file` for reading the input and writing each shard.
  - Plain Python (`re`, `json`) for the two anchored name patterns (projection matrices, buffers), the required pre-write checks, greedy 256 MiB sharding in key order, and writing `model.safetensors.index.json`.
  - Not used: `transformers.save_pretrained` would have loaded the model through the GPTNeoX class, which re-creates rotary buffers and applies one dtype to the whole model; mixing bf16 projections with fp32 everything else and controlling the exact key set was simpler and more auditable as a 90-line script. `mergekit` and `torch-state-bridge` add nothing for a pure dtype/shard export.
- Approximate time spent, if you can tell: about 3 minutes.

Result: 9 shards, 196 tensors, 64 bfloat16 projection matrices, 132 float32 tensors, 48 buffers dropped, total tensor data 2,436,513,792 bytes. The script refuses to overwrite existing shards and re-reads every shard after writing to verify dtypes and bit-exact values.
