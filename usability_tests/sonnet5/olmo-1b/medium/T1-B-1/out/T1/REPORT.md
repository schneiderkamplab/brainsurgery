## Participant self-report

- Final artifact path: `out/T1/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the first run passed all asserts.
- Pitfalls or surprises you hit (one line each):
  - `move` requires the destination to not already exist, so the renumbering
    moves must be ordered so that each destination block index has already
    been vacated (by the initial delete or by an earlier move in the list)
    before it is written to; a naive ascending-by-old-index order works here
    because every destination is either one of the deleted indices (2, 6, 10)
    or was already emptied by a prior move in the same list.
- Anything in the task text or documentation that was unclear: none.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes.
