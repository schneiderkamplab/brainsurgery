# Participant self-report (T1, condition B)

- Final artifact path: `out/T1/plan.yaml` (output `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - 1: `PlanLoaderError: each inputs entry must be a non-empty string` — I wrote `inputs` as a `name`/`path` mapping instead of the `alias::path` string form (crash, plan did not parse; no output written).
- Pitfalls or surprises you hit (one line each):
  - Renumbering must go in ascending source order so every destination index is already vacated; `move` refuses existing destinations, which would catch a collision anyway.
  - Dots in names must be escaped in regex references, and matching is full-match, so `layers\.1\.` cannot overreach onto `layers.10`.
- Anything in the task text or documentation that was unclear:
  - The `help` text does not show the `inputs` entry form; it is only in the README plan-format section.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~3 minutes
