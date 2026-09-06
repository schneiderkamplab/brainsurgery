# T3 self-report

- Final artifact path: `out/T3/solution.py` (output shards + `model.safetensors.index.json` in `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The task text says the embedding tensors are 206 MB each; that is their float16 size. After the required upcast to float32 they are 412 MB, still over the 256 MiB budget, so each still lands alone in its own shard.
  - Anchored regexes were used for both the projection set and the buffer set so that `.*weight` cannot reach embeddings, norms, or biases; a check confirms every non-projection tensor is float32.
- Anything in the task text or documentation that was unclear: the shard file naming scheme is not specified; I used the HuggingFace `model-XXXXX-of-NNNNN.safetensors` convention and greedy packing in original key order.
- Tools used (condition F): torch 2.14.0 (dtype casts), safetensors 0.5.3 (load/save, `format: pt` metadata). A plain script was chosen because `transformers` `save_pretrained(dtype=...)` applies one dtype to everything and would need a post-pass anyway, and mergekit cannot express mixed precision per tensor.
- Approximate time spent, if you can tell: about 2 minutes.
