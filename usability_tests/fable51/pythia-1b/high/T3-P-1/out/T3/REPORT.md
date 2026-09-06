# Participant self-report: T3 (Pythia-1B, condition P)

- Final artifact path: `out/T3/solution.py` (output checkpoint in `out/T3/`: 9 shard files `model-0000N-of-00009.safetensors` plus `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution passed all checks and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - Projection matrices were matched with an anchored regex on the four exact module names ending in `.weight`, so embeddings, layer norms and biases could not be swept up by a broad `.*weight` pattern.
  - Buffers were removed by an anchored regex on `attention.bias`, `attention.masked_bias` and `attention.rotary_emb.inv_freq`; the projection biases (`attention.dense.bias`, `mlp.*.bias`, `query_key_value.bias`) do not match it and are kept.
  - After upcasting, `embed_in.weight` and `embed_out.weight` are 412 MB each (206 MB was their float16 size), so they exceed the 256 MiB budget and each gets its own shard; the greedy packer handles that case explicitly.
- Anything in the task text or documentation that was unclear:
  - The shard file naming scheme is not specified; I used the HuggingFace convention `model-XXXXX-of-XXXXX.safetensors` and added `metadata.total_size` to the index.
  - The "206 MB each" figure for the embeddings refers to the float16 input size, not the float32 output size that actually determines sharding.
- Tools used (condition F): n/a (condition P): torch 2.14.0, safetensors 0.5.3.
- Approximate time spent, if you can tell: about 3 minutes.
