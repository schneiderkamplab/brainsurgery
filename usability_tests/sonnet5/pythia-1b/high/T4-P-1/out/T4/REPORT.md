# Participant self-report

- Final artifact path: `out/T4/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None of substance; the main care point was taking each task vector against
    the unmodified `base` (not against a partially-merged tensor), which the
    task text flags explicitly, so the merge accumulates both deltas onto a
    single `float32` copy of base before casting back to `float16`.
- Anything in the task text or documentation that was unclear:
  - `inputs/` also contains a `lora/` directory not mentioned anywhere in
    TASK.md; ignored it since it's irrelevant to the base/ft1/ft2 merge.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes.
