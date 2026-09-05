## Participant self-report

- Final artifact path: `out/T1/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the plan succeeded on the first run.
- Pitfalls or surprises you hit (one line each):
  - `move` requires the destination to not already exist, so blocks had to be
    renumbered in an order where each destination index is vacated (by a
    prior move or the initial delete) before it is reused; ascending
    destination order (2,3,4,...,11) achieves this in one pass.
  - Structured references (`["model","layers","3","*rest"]` -> `["model","layers","2","*rest"]`)
    let one `move` relocate all 7 tensors of a block at once instead of
    writing 7 regex moves per block.
- Anything in the task text or documentation that was unclear: none; the
  README's structured-expression section and the `move`/`delete`/`assert`
  help text were enough to write the plan without trial and error.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: about 10 minutes.
