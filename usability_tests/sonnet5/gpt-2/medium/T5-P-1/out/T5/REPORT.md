## Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Had to keep straight that `lora_A`/`lora_B` follow the `nn.Linear` convention
    (`B @ A` is `[out, in]`), so with `fan_in_fan_out=true` the product must be
    transposed before adding to the Conv1D `[in, out]` base weight.
  - Adapter tensor names are prefixed with `base_model.model.`, which has to be
    stripped to map to base checkpoint names (`h.<i>.attn.c_attn.weight`).
  - Sharding by "≤100 MiB of tensor data per shard, oversized tensor alone in
    its own shard" needed a simple greedy bin-packer over `numel() * element_size()`,
    not file size (which includes header overhead).
- Anything in the task text or documentation that was unclear: none; the task
  spelled out the exact formula, scale, and layout convention, which made
  correctness straightforward to verify against a manual recomputation.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes to write and verify
  the script.
