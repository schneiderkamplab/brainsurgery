# T4 participant self-report

- Final artifact path: `out/T4/solution.py` (invoked via `out/T4/run.sh`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None encountered — inputs matched the stated layout exactly (244 tensors,
    same names/shapes/dtypes across base/ft1/ft2).
- Anything in the task text or documentation that was unclear: none; the
  merge formula, MLP tensor list, and required checks were fully specified.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3, used
  directly in a plain Python script rather than through `mergekit`. Reason:
  the task's required checks (identical key sets across all three
  checkpoints, bit-exact match on the 180 non-MLP tensors, exactly 64 merged
  tensors, exactly 244 output tensors) need to fail loudly and precisely;
  writing them as explicit assertions over `dict[str, torch.Tensor]` was more
  direct and auditable than expressing the same preconditions through a
  mergekit task-arithmetic YAML, and it keeps float32 accumulation / dtype
  casting fully explicit and inspectable.
- Approximate time spent, if you can tell: a few minutes (single script,
  single successful run).
