# Participant self-report: T3 (Pythia-1B), condition P

- Final artifact path: `out/T3/solution.py` (output checkpoint in `out/T3/`: 9 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Anchored regexes on the four exact projection names and the three buffer names, and counted matches (64 / 48) as checks, to avoid `.*weight` overmatching embeddings and norms.
  - Sharding is greedy in input tensor order; the two 412 MB float32 embedding matrices each exceed the 256 MiB budget and go alone in their own shard.
- Anything in the task text or documentation that was unclear:
  - The shard file naming scheme and whether the index needs `metadata.total_size` are not specified; used the HF convention (`model-0000i-of-0000n.safetensors`, `total_size` included).
  - Whether the reference shards in input order or sorted-key order is unspecified; used input order.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
