## Participant self-report

- Final artifact path: `out/T4/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None of consequence. Care was needed to compute each fine-tune's task
    vector against the unmodified `base` tensor (not against a
    partially-merged output), and to do the arithmetic in float32 before
    casting back to the original dtype.
- Anything in the task text or documentation that was unclear: no; the spec,
  required checks, and grading criteria were unambiguous.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes (single write + run,
  plus a manual sanity check of the output against an independently computed
  expected value).
