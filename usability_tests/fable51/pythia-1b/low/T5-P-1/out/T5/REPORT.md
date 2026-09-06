# Participant self-report: T5 (condition P)

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`, 6 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each): none failed; the first run succeeded but packed the embedding matrices greedily with other tensors, so I changed the sharding to store `embed_in` and `embed_out` alone as the task text says and re-ran.
- Pitfalls or surprises you hit (one line each):
  - The sharding paragraph says a tensor larger than 512 MiB gets its own shard, then names the 206 MB embeddings as examples; I followed the literal statement that the two embeddings are stored alone.
  - PEFT adapter names carry a `base_model.model.` prefix that must be stripped to find the base key.
- Anything in the task text or documentation that was unclear: whether the two 206 MB embeddings must be alone in their shards (they are under the 512 MiB budget), and the exact shard file naming expected by the grader.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
