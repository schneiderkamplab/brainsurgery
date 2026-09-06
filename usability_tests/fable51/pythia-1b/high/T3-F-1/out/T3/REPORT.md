# T3 participant self-report (condition F)

- Final artifact path: `out/T3/solution.py` (output: `out/T3/model-0000N-of-00009.safetensors` x9 plus `out/T3/model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution passed all checks and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - The buffers are float16/uint8 tensors named like parameters (`attention.bias`, `attention.masked_bias`), so a loose `.*bias` or `.*weight` pattern would either drop a real bias or keep a mask; I matched full anchored names for both the 64 projections and the 48 buffers and asserted the match counts (64 and 48) before writing.
  - After upcasting, each embedding is 412 MB, well over the 256 MiB budget, so the sharder needs an explicit "oversized tensor goes alone" branch; greedy packing in key order otherwise.
- Anything in the task text or documentation that was unclear:
  - The sharding rule does not pin the packing order or shard naming; I used the HuggingFace convention (`model-XXXXX-of-XXXXX.safetensors`, greedy in input key order) on the assumption that grading checks the rules rather than a specific layout.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: `safe_open` to stream tensors from the input and `save_file` to write each shard; direct control over the key set and dtypes.
  - `torch` 2.14.0: dtype casts (`.to(torch.bfloat16)` / `.to(torch.float32)`).
  - Python stdlib `json` / `re` for the index file and anchored name matching.
  - I did not use `transformers.save_pretrained` with a dtype, because it applies one dtype to the whole model and would reintroduce the buffers and require a second pass to fix embeddings/norms/biases; a direct script gives exact per-tensor control and lets the required checks run before any write. `mergekit` and `torch-state-bridge` were unnecessary since no keys are renamed.
- Approximate time spent, if you can tell: about 3 minutes (one inspection pass, one script write, one run).
