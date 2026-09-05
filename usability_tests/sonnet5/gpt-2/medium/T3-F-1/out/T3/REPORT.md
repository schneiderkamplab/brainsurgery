# T3 self-report

- Final artifact path: `out/T3/solution.py` (invoked via `out/T3/run.sh`)
- Number of times you executed the script or plan: 2 (identical: direct
  `python solution.py`, then via `run.sh` to confirm the wrapper works;
  no code changes between runs, both succeeded)
- Which executions failed, and why (one line each): none failed.
- Pitfalls or surprises you hit (one line each):
  - The 12 `h.<i>.attn.bias` causal-mask buffers share the `.bias` suffix
    with real parameter biases (`c_attn.bias`, `c_proj.bias`), so the drop
    set had to be named exactly, not pattern-matched on `attn.bias`-style
    substrings.
  - Only the 4 named projection `.weight` tensors per layer go to bfloat16;
    a naive `.*weight` regex would also have caught `wte.weight` and
    `wpe.weight`.
  - `wte.weight` (154 MB) alone exceeds the 64 MiB shard budget, so the
    greedy packer has to flush it into its own shard rather than erroring
    or splitting it.
- Anything in the task text or documentation that was unclear: none;
  the tensor list, shard budget, and required-checks list were precise
  enough to implement directly.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 only,
  as a plain script — no merge/adapter tool (mergekit, peft,
  torch-state-bridge) fits this task, since it's a per-tensor dtype
  cast plus buffer removal and manual sharding, not a merge, LoRA op, or
  key-renaming problem. A direct script over the safetensors API gives full
  control over per-tensor dtype and the shard-packing invariant, and keeps
  the assertions (48 bf16 tensors, `wte.weight` fp32, exact tensor count)
  inline before any file is written.
- Approximate time spent, if you can tell: ~10 minutes (single pass, no
  failed attempts).
