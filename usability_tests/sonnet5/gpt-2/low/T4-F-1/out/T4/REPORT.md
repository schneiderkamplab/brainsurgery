# T4 report

## Participant self-report

- Final artifact path: `out/T4/solution.py` (invoked via `out/T4/run.sh`), output at `out/T4/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - None; the tensor names were the standard GPT-2 keys (`h.<i>.mlp.c_fc.weight` etc.) with no HF `transformer.` prefix, matching the task description directly.
- Anything in the task text or documentation that was unclear: none
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3 — direct load (`safe_open`/`get_tensor`) and save (`save_file`) of the three input checkpoints and the output; chosen over mergekit's task-arithmetic config because this task's correctness hinges on an ordering hazard (each task vector must be computed against the unmodified base, not a partially-merged one) and on an upfront three-way equality check outside the 48 MLP tensors — both are easiest to express and audit as a short, explicit script rather than through a merge-config abstraction.
  - `torch` 2.14.0 — tensor arithmetic in float32 and `torch.equal` for the bit-exact non-MLP verification.
- Approximate time spent, if you can tell: a few minutes (single script, single successful run)
