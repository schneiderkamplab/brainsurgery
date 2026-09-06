# T3 (condition F) — participant self-report

- **Final artifact path:** `out/T3/solution.py` (invoked via `out/T3/run.sh`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — first execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The causal-mask buffer (`h.<i>.attn.bias`) and the attention bias
    *parameter* (`h.<i>.attn.c_attn.bias`) both end in `.bias` and both
    match `attn.bias` as a substring — an anchored regex
    (`^h\.\d+\.attn\.bias$`) is required to avoid deleting real biases; a
    loose `endswith("attn.bias")` filter (used only in my own verification
    check, not in the solution) would have wrongly flagged the biases as
    mask buffers.
  - Sharding by insertion order (as in the source file) rather than sorting
    by name keeps `wte.weight`/`wpe.weight` and each layer's tensors
    reasonably grouped, and naturally isolates the one oversized tensor
    (`wte.weight`, ~154 MB) into its own shard once the running total would
    exceed the 64 MiB budget.
- **Anything in the task text or documentation that was unclear:** No —
  the spec's required-checks list and shard-budget definition (tensor data
  only, headers excluded) were unambiguous.
- **Tools used (condition F): name, version, and why:**
  - `torch` 2.14.0 — load tensors, `.to(torch.bfloat16)` cast, byte-size
    accounting for shard packing.
  - `safetensors` 0.5.3 (`safe_open`, `save_file`) — read the input
    checkpoint and write each shard file.
  - Plain Python (`json`, `re`) for the index file and name matching.
  - No merge/adapter/HF-auto-sharding tool was a better fit: this task is
    precision casting + buffer pruning + manual reshard, not a model merge
    (mergekit) or LoRA operation (peft), and hand-rolling the shard packer
    gives exact control over the 64 MiB budget and the single-oversized-
    tensor rule that the spec requires.
- **Approximate time spent, if you can tell:** ~10 minutes.

## Verification performed

Before writing, the script asserts: exactly 48 bfloat16 tensors,
`h.0.attn.c_attn.weight` is bfloat16, `wte.weight` is float32, exactly 148
output tensors, exactly 12 dropped buffers (all matching the anchored mask
buffer pattern), and no parameter accidentally dropped.

After writing, I independently re-verified from disk (not just the
in-process assertions): shard byte totals (≤64 MiB each, except the single
oversized `wte.weight` shard), `weight_map` keys exactly match the union of
tensor names across shard files, bit-exact equality of unchanged float32
tensors against the input, and bit-exact equality of a cast tensor against
`tensor.to(torch.bfloat16)` applied directly to the corresponding input
tensor.
