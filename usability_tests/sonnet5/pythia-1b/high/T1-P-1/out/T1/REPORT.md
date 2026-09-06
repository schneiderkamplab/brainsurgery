# Participant self-report

- Final artifact path: `out/T1/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - None in execution; the main thing to get right up front was building the
    old-index -> new-index map by iterating surviving indices in original
    order (`[i for i in range(16) if i not in {2,6,10,14}]`) rather than
    shifting indices down by counting drops, to avoid any risk of a
    renumbering collision.
- Anything in the task text or documentation that was unclear: none; the
  explicit old->new mapping in the task spec removed any ambiguity about
  block order.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: a few minutes; single-pass script,
  no debugging needed.
