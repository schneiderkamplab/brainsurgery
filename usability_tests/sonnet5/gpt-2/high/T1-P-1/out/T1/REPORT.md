# Participant self-report

- Final artifact path: `out/T1/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single run succeeded.
- Pitfalls or surprises you hit (one line each):
  - The block-index regex has to be anchored (`^h\.(\d+)\.`) and use a captured integer, not a
    naive string-prefix check, or `h.1.` would also match `h.10.` and `h.11.`.
  - Iterating the old→new block mapping in ascending old-index order and checking each new key
    for a pre-existing collision before writing catches any renumbering-order mistake immediately
    instead of letting one block silently overwrite another.
- Anything in the task text or documentation that was unclear: no, the block list, mapping and
  required checks were fully specified.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes; one script, one execution.
