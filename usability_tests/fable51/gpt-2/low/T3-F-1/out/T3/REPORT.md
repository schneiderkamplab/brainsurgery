# T3 participant self-report

- Final artifact path: `out/T3/solution.py` (run as `.venv/bin/python out/T3/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `wte.weight` (154 MB) exceeds the 64 MiB shard budget, so the sharder places oversized tensors alone in their own shard.
  - The causal-mask buffers share the `bias` suffix with real parameters, so they are dropped by an anchored regex `^h\.\d+\.attn\.bias$` rather than a suffix match.
- Anything in the task text or documentation that was unclear: nothing significant; I assumed key order from the input file for greedy shard packing and that shard file names are free-form as long as the index maps every tensor.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: dtype casting with `tensor.to(torch.bfloat16)` (round-to-nearest-even).
  - `safetensors` 0.5.3: `load_file` / `save_file` for reading the input and writing the shards.
  - Plain Python script with anchored regexes; transformers `save_pretrained` was not used because it would re-tie/rename tensors and could not express the exact per-tensor dtype mix and buffer dropping as directly.
- Approximate time spent, if you can tell: about 2 minutes
