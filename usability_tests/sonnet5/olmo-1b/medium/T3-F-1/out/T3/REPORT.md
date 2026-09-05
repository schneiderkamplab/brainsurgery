# Participant self-report — T3 (condition F)

- **Final artifact path:** `out/T3/solution.py` (invoked via `out/T3/run.sh`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none — first execution passed all in-script checks.
- **Pitfalls or surprises you hit (one line each):**
  - The largest tensors (`model.embed_tokens.weight`, `lm_head.weight`, 412 MB each) exceed the 256 MiB shard budget, so the greedy packer needs a special case to put them alone in their own shard rather than trying to co-pack them.
  - Shard file sizes on disk are slightly above 268,435,456 bytes because of the safetensors header; the spec's 256 MiB limit is on tensor *data* only, so I verified against raw tensor byte counts, not file size.
- **Anything in the task text or documentation that was unclear:** No, the spec was precise (exact tensor names, dtypes, and shard rule were all given).
- **Tools used (condition F):** `safetensors` 0.5.3 (`safe_open`/`save_file`) for reading/writing shards, and `torch` 2.14.0 for the bfloat16 cast. Chose a plain script over `transformers ... save_pretrained(dtype=...)` because that API casts the whole model to one dtype uniformly; this task needs per-tensor dtype selection (112 specific projections to bf16, everything else stays fp32) plus a custom shard-size budget, which is simplest to express directly against the state dict rather than fighting a model-level save API.
- **Approximate time spent, if you can tell:** ~10 minutes.
