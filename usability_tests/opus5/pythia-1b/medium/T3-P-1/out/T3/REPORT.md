# T3 participant self-report

- **Final artifact path:** `out/T3/solution.py` (output: `out/T3/model-0000N-of-00009.safetensors` + `out/T3/model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution passed all checks.
- **Pitfalls or surprises you hit (one line each):**
  - Over-broad targeting is the obvious trap: I used a fully anchored regex naming the four projection matrices explicitly rather than anything like `.*weight`, so embeddings, layer norms and `*.bias` cannot be caught.
  - `attention.bias` is a uint8 causal-mask buffer, not a parameter — it has to be dropped, and it would also have been wrongly upcast by a blanket float32 pass.
  - Shard order is unspecified by the task; I used the input file's own header order (which equals its data-layout order) as the deterministic ordering.
  - The two 206 MB embedding tensors exceed the 256 MiB budget on their own, so the greedy packer must place an oversized tensor alone instead of erroring.
  - Tensors were made contiguous before saving, since safetensors rejects shared/non-contiguous storage.
- **Anything in the task text or documentation that was unclear:**
  - The shard file naming convention and the iteration order used for packing are not specified; I assumed HF's `model-{i:05d}-of-{n:05d}.safetensors` and checkpoint order.
  - Whether the index `metadata.total_size` is graded is not stated; I included it as HF does.
- **Tools used (condition F):** n/a (condition P: Python 3.13, torch 2.14.0, safetensors 0.5.3).
- **Approximate time spent:** ~5 minutes.
