## Participant self-report

- Final artifact path: `out/T1/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard: ordered the 10 block moves by ascending old
    index so each destination index was already deleted or already vacated by
    an earlier move, avoiding any overwrite.
- Anything in the task text or documentation that was unclear: none; the
  README/interfaces-reference section on structured expressions and capture
  based destination synthesis for `move` was enough to do the renumbering as
  one `move` per surviving block (matching all 15 tensors of that block via a
  `*rest` wildcard) instead of one move per tensor.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes
