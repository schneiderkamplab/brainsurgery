# Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The base weight is Conv1D `[in, out]` while `B @ A` produces `[out, in]`
    (nn.Linear convention), so the `fan_in_fan_out=True` flag requires
    transposing the low-rank product before adding it, per the task text.
  - `wte.weight` (154 MB) exceeds the 100 MiB shard budget on its own, so it
    must be excluded from the greedy bin-packing and placed alone in its own
    shard rather than triggering an off-by-one shard split.
- Anything in the task text or documentation that was unclear: none; the
  spec's formula and shard rule were unambiguous enough to implement directly.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: one pass, a few minutes to write
  and verify the script (including a manual bit-exact/relative-error check
  against the base and adapter tensors).
