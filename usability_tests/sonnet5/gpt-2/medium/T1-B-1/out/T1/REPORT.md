# Participant self-report

- Final artifact path: `out/T1/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Renumbering has a collision hazard: moves must go low-to-high old index
    (3->2, 4->3, 6->4, 7->5, 9->6, 10->7, 11->8) so each destination slot is
    already vacated by an earlier delete (blocks 2, 5, 8) or an earlier move
    before it is reused as a target; deleting 2/5/8 first and moving in
    ascending order avoids it entirely, so no actual failure was hit.
  - Used structured path patterns (`["h", "3", "*rest"]` -> `["h", "2",
    "*rest"]`) instead of regex so one `move` per block covers all 13 tensors
    (ln_1/ln_2, attn.c_attn, attn.c_proj, attn.bias, mlp.c_fc, mlp.c_proj)
    without listing them individually.
- Anything in the task text or documentation that was unclear: none; the
  README's structured-expression section (`*xs` variadic capture) and the
  `move`/`delete` help entries were sufficient.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes including verification.
