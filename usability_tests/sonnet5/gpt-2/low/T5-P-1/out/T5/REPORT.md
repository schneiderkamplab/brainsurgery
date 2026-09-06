## Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Conv1D `[in, out]` base layout vs `nn.Linear` `[out, in]` adapter factors means the `B @ A` product must be transposed before adding, per `fan_in_fan_out=true`.
  - Shard budget (100 MiB tensor data) is smaller than `wte.weight` (154 MB), so that tensor must land alone in its own shard rather than forcing an even split.
- Anything in the task text or documentation that was unclear: none
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~5 minutes
