# Participant self-report

- Final artifact path: `out/T3/solution.py` (writes shards + `model.safetensors.index.json` into `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single execution succeeded on the first run.
- Pitfalls or surprises you hit (one line each):
  - `gpt_neox.embed_in.weight` / `embed_out.weight` are ~206 MB in float16 but the output requires float32, so they become ~393 MiB each — already over the 256 MiB shard budget, so each must be a solo shard by size alone, not just because they're "the big ones" in the prompt text; the bin-packing logic checks each tensor's *output* byte size against the limit, not the input size.
  - Buffers must be matched by full dotted name (`attention.bias`, `attention.masked_bias`, `attention.rotary_emb.inv_freq`); a looser regex on `bias` alone would also catch `attention.dense.bias` and `mlp.dense_h_to_4h.bias`, which are parameters that must survive as float32.
- Anything in the task text or documentation that was unclear: none — TASK.md's per-tensor listing was complete enough to write exact regexes without guessing.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 only, via a plain script (`solution.py`). No merge/adapter tool was needed since this task is a per-tensor dtype cast, buffer drop, and re-shard on a single checkpoint — not a merge or adaptation — so `mergekit` and `peft` don't apply, and `torch-state-bridge`'s key-rewriting features aren't needed since no tensor names change. Direct `safe_open`/`save_file` gives the most explicit control over per-tensor dtype and shard membership, which is what the required checks and the bit-exact grading demand.
- Approximate time spent, if you can tell: a few minutes of scripting plus one verification pass.
