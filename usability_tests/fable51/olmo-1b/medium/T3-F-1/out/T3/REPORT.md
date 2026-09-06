# T3 self-report (condition F)

- Final artifact path: `out/T3/solution.py` (run with `python out/T3/solution.py` from the sandbox root); output checkpoint in `out/T3/` (10 shards + `model.safetensors.index.json`).
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none.
- Pitfalls or surprises you hit (one line each):
  - Avoided the `.*weight` overmatch by using an anchored regex that names the seven projection modules per layer explicitly; embeddings and `lm_head` are not matched.
  - The shard budget is on tensor bytes, not file size, so shards of exactly 256 MiB tensor data end up slightly larger on disk due to the header; that is per spec.
  - Verified float32 tensors bit-exact against the input and re-read every shard after writing to confirm dtypes, values and the index mapping.
- Anything in the task text or documentation that was unclear: the ordering used to pack tensors into shards is not specified; I used greedy first-fit in sorted tensor-name order, with oversized tensors (`lm_head.weight`, `model.embed_tokens.weight`) alone in their own shards.
- Tools used (condition F): `torch` 2.14.0 (dtype cast with `.to(torch.bfloat16)`, equality checks) and `safetensors` 0.5.3 (`safe_open` for reading, `save_file` for writing). A plain script was chosen over `transformers.save_pretrained(dtype=...)` because that API applies one dtype to the whole model and its sharding logic would need post-hoc patching to keep embeddings float32; mergekit dtype conversion is likewise whole-model. Direct control gave exact targeting and explicit checks before writing.
- Approximate time spent, if you can tell: about 2 minutes.
