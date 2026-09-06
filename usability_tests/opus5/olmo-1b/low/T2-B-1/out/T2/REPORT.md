# T2 self-report (condition B)

- Final artifact path: `out/T2/plan.yaml` (output `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run passed all asserts and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each source to resolve to exactly one tensor and a single literal destination, so the plan cannot be pattern-driven over layers; I emitted the 64 per-tensor copy/delete/concat groups explicitly (270-line plan).
  - Destinations must not already exist, so each original projection has to be `delete`d before the rebuilt tensor is concatenated back under the same name.
  - Tensor references are full-match regexes, so dots in `model.layers.0...` are escaped to avoid accidental overmatching.
  - Slice syntax on the source ref (`::[:640, :]` for rows, `::[:, :640]` for columns) does all the head selection; no reshape into head blocks was needed.
- Anything in the task text or documentation that was unclear: the README does not say whether `concat`/`split` destinations support capture-group rewrites like `copy`'s `to`; the `help` text answered it (they do not), but only after trying.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes, most of it reading `help.txt` and generating the plan text.
