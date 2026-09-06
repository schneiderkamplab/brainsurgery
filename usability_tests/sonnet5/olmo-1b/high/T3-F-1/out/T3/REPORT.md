# Participant self-report

- Final artifact path: `out/T3/solution.py` (invoked via `out/T3/run.sh`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - This checkpoint has no norms/biases and no buffers, so the "keep in float32"
    set collapses to exactly `{model.embed_tokens.weight, lm_head.weight}`;
    I still enumerated the 112 target names explicitly (layer index x
    q/k/v/o/gate/up/down) and asserted the count, rather than relying on the
    complement, so the script fails loudly if that assumption is ever wrong.
  - The 256 MiB budget lands exactly on 2 transformer layers per shard
    (4 x 2048x2048 + 3 x 8192x2048, bf16 = 128 MiB/layer), so 8 of the 10
    shards pack to exactly the 268,435,456-byte limit; a strict `<` budget
    check instead of `<=` would have broken this.
- Anything in the task text or documentation that was unclear: no
- Tools used (condition F): `torch` 2.14.0 (dtype casting, tensor equality
  checks) and `safetensors` 0.5.3 (`safe_open`/`save_file`) directly, in a
  plain script. No merging/adapter tool applies here (no adapters or model
  arithmetic involved), and HF's `save_pretrained(..., torch_dtype=...)`
  sharding path casts the whole state dict to one dtype rather than giving
  per-tensor control, so a small custom script with an explicit bin-packer
  was the most direct route.
- Approximate time spent, if you can tell: ~10 minutes including verification
