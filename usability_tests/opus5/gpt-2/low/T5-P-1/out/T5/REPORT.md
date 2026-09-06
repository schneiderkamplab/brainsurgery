# T5 — Participant self-report

- Final artifact path: `out/T5/solution.py` (output in `out/T5/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Conv1D layout: the base `[768, 2304]` is `[in, out]`, so `B @ A` (`[out, in]`) had to be transposed before adding; `fan_in_fan_out = true` is the signal.
  - The sharding budget counts tensor data only, so `wte.weight` (154 MB) exceeds it on its own; the greedy packer had to let an oversized tensor sit alone rather than reject it.
  - Adapter names carry the `base_model.model.` prefix, so the mapping to base names is a prefix strip plus `.weight`.
- Anything in the task text or documentation that was unclear:
  - Shard file naming and index `metadata.total_size` are not specified; I used the HF convention (`model-0000N-of-0000M.safetensors`).
  - Shard packing order/strategy is unspecified; I packed in base key order with a greedy fill.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: ~5 minutes
