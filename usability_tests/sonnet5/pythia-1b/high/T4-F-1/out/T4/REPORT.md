## Participant self-report

- Final artifact path: `out/T4/solution.py` (invoked via `out/T4/run.sh`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None specific to this task; the tensor-name pattern for the MLP block
    (`gpt_neox.layers.<i>.mlp.{dense_h_to_4h,dense_4h_to_h}.{weight,bias}`)
    matched the spec exactly, so no regex overreach onto other layers was
    a risk.
- Anything in the task text or documentation that was unclear: no — the
  formula, tolerance, and required checks were unambiguous.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0 — tensor arithmetic in float32, cast back to float16.
  - `safetensors` 0.5.3 (`safetensors.torch.load_file`/`save_file`) — reading
    the three input checkpoints and writing the merged output.
  - Chose a plain script over `mergekit`'s task-arithmetic YAML because the
    task's hard precondition (steps must abort loudly if the frozen-backbone
    assumption is violated, or if the merged/output tensor counts are wrong)
    needed explicit, inspectable checks before any tensor is touched, and a
    ~90-line script made the three-way equality check, the exact 64/244
    tensor-count assertions, and the base-vs-ft1/ft2 ordering (each task
    vector taken against the *unmodified* base, not a partially merged one)
    trivial to get right and to verify by inspection. `mergekit`'s
    `task_arithmetic` merge method would also apply lambda-weighted deltas
    but assembles the base state internally, making it less direct to prove
    that the ordering hazard (vector 2 taken against a base already touched
    by vector 1) was avoided, and it does not include a shared-tensor
    precondition check — that would still have had to be a separate script
    step.
- Approximate time spent, if you can tell: a few minutes (exploration,
  writing the script, one successful run, plus an out-of-band negative test
  of the abort path that is not part of the graded artifact).
