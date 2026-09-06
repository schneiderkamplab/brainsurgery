# T1 self-report (condition B)

- Final artifact path: `out/T1/plan.yaml` (output `out/T1/model.safetensors`, 86 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard: every surviving block moves to a lower index, so applying the `move` transforms in ascending source order (3->2, 4->3, ... 15->11) guarantees each destination is already free; descending order would hit "destination exists".
  - Dots in the source regex must be escaped (`model\.layers\.3\.(.*)`) or `.` would also match other separators; the `to` side is a plain rewrite where dots are literal and `\1` carries the tail.
  - `delete` with an alternation `(2|6|10|14)` is anchored full-match, so it does not accidentally catch layers 12 or 14 prefixes; I still added a count check before and after the delete.
  - Output path ends in `.safetensors`, so a single file is written and the default 5GB shard budget never kicks in (the result is ~4.0GB).
- Anything in the task text or documentation that was unclear:
  - The README does not state explicitly whether `assert: count` accepts a pattern that matches nothing; I used `not: { exists: ... }` for the "no blocks 12-15" check to avoid depending on that.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: a few minutes; most of it reading `docpack/README.md` and the `move`/`delete` help entries.
