# T4 self-report

- Final artifact path: `out/T4/solution.py` (invoked via `out/T4/run.sh`), output at `out/T4/model.safetensors`.
- Number of times you executed the script or plan: 1.
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None specific to this task; the tensor naming (`h.<i>.mlp.c_fc/c_proj.{weight,bias}`) matched the spec exactly, so no regex overreach or renumbering concerns arose.
- Anything in the task text or documentation that was unclear: no, the required checks and merge formula were unambiguous.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3 — direct, dependency-light load/save of the three checkpoints and the output; gives exact control over which tensors are read/written without going through a model config.
  - `torch` 2.14.0 — float32 arithmetic for the task-vector merge and tensor equality checks (`torch.equal`) for the shared-tensor verification.
  - Chose a plain script over `mergekit`'s task-arithmetic YAML because the task's step 1 (three-way exact-equality check on the 112 non-MLP tensors, aborting loudly if it fails) and the exact "40 + 40" scaling of two independent task vectors against the *same* unmodified base are easiest to express and audit directly in code; mergekit's config surface is oriented at HF model directories and would have made the required pre-check and the merge-order guarantee (ft1 and ft2 deltas each computed against `base`, not against an intermediate result) less explicit.
- Approximate time spent, if you can tell: a few minutes (single edit-run cycle, first execution passed).
