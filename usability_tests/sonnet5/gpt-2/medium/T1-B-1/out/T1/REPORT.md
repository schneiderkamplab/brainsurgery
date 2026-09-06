## Participant self-report

- Final artifact path: `out/T1/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard: moving blocks in ascending old-index order
    (3->2, 4->3, 6->4, 7->5, 9->6, 10->7, 11->8) guarantees every destination
    slot is already empty (deleted or vacated by the previous move) before
    the move runs, since `move` refuses to overwrite an existing destination.
  - Used anchored regexes (`^h\.N\..*$`) for delete/move targets to avoid the
    pattern `h.1.` accidentally matching `h.10.`/`h.11.` (prefix overreach).
- Anything in the task text or documentation that was unclear: none; the
  README's tensor-reference and move/delete semantics were sufficient.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes, single attempt.
