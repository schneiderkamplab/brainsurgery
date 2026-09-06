# T3 self-report (condition P)

- Final artifact path: `out/T3/solution.py` (output in `out/T3/`: 4 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none.
- Pitfalls or surprises you hit (one line each):
  - `h.<i>.attn.bias` is the causal-mask buffer while `h.<i>.attn.c_attn.bias` etc. are real
    parameters, so the delete pattern had to be anchored exactly (`^h\.\d+\.attn\.bias$`).
  - A naive `.*weight` cast would have hit `wte`/`wpe`/layer norms; I matched the four
    projection names explicitly and asserted the bfloat16 count is exactly 48.
  - `wte.weight` (154 MB) exceeds the 64 MiB shard budget on its own; the greedy packer
    naturally leaves it alone in the last shard, but only because it never merges an
    over-budget tensor with anything already buffered.
- Anything in the task text or documentation that was unclear:
  - The shard file naming convention was not specified; I used the HF convention
    `model-0000i-of-0000N.safetensors`.
  - Shard ordering was not specified; I kept the input key order (already sorted).
  - Whether the index needs `metadata.total_size` was not stated; I included it.
- Tools used (condition F): n/a.
- Approximate time spent, if you can tell: ~5 minutes.
