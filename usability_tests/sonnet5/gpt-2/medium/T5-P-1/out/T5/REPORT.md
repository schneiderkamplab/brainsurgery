# Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Had to be careful with the Conv1D `[in, out]` vs Linear `[out, in]` layout: `B @ A` is `[out, in]`, so it needs a transpose before adding to the `[in, out]` base weight (this is what `fan_in_fan_out=true` signals).
  - Adapter tensor names are prefixed with `base_model.model.` and use `lora_A`/`lora_B`, which had to be stripped/mapped to the base names (`h.<i>.attn.c_attn.weight`).
  - Sharding rule needed explicit handling of the oversized `wte.weight` (154 MB > 100 MiB budget): it must be isolated in its own shard rather than folded into the greedy bin-packing.
- Anything in the task text or documentation that was unclear: none; the spec's formula (`scale = lora_alpha / r`, `(B @ A).T`) and sharding rule were explicit enough to implement directly.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes to write and verify the script.
