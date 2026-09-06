# T5 self-report

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`)
- Number of times you executed the script or plan: 3
- Which executions failed, and why (one line each):
  - 1: `crash` / no_match — `assert: count: {of: 'model::.*lora_.*', is: 0}` errored with "count.of matched zero tensors"; `count` cannot express "zero matches".
  - 2: `crash` — `assert: not: {exists: {of: ...}}` did not parse; `exists` takes the reference directly, not an `of:` mapping.
- Pitfalls or surprises you hit (one line each):
  - `count ... is: 0` is unusable as an emptiness check; the reference resolver raises before the comparison. `not: {exists: ...}` is the working form.
  - `exists` takes a bare reference string, unlike every other assert operator which uses `of:`.
  - Conv1D `[in, out]` base layout vs Linear adapter layout: `B @ A` is `[2304, 768]` and needs `permute: order [1, 0]` before `add_`.
  - Output alias inference: keeping all intermediates on the `model` alias (and the adapter read-only on `lora`) is what makes the output unambiguous; adapter tensors then never reach the output at all.
  - `100MB` in the plan is binary (104,857,600 B), which matches the task budget exactly; `wte.weight` (154 MB) lands alone in its own shard as required.
- Anything in the task text or documentation that was unclear:
  - The README lists `exists` alongside operators that use `of:`, without showing its actual argument form; the help text was needed.
  - Nothing about how to assert the *absence* of a pattern is documented.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes
