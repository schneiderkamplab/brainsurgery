# T1 self-report (condition B)

- Final artifact path: `out/T1/plan.yaml` -> `out/T1/model.safetensors`
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - #1 `crash` (plan did not compile): `PlanLoaderError: transform #11: exists must be a non-empty string reference or non-empty list of strings` — I wrote `exists: { of: ... }` by analogy with `count`/`shape`, but `exists` takes the reference directly.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collisions: solved by deleting blocks 2/5/8 first and then moving survivors in ascending order, so every `move` destination is already free.
  - `assert` operator argument shapes are not uniform: `count`/`shape`/`dtype` take `{of, is}`, `exists`/`iszero` take a bare reference.
  - Dots in tensor names must be escaped in regex targets (`h\.2\..*`), otherwise `h.2` would also match e.g. `h12`-style names.
- Anything in the task text or documentation that was unclear:
  - The README's assert list does not show the argument form of each operator; I only learned `exists`' form from the error message.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes
