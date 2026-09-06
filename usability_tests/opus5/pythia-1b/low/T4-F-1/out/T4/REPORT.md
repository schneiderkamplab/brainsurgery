# T4 participant self-report

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why: none.
- Pitfalls or surprises you hit (one line each):
  - The ordering hazard is the whole task: both task vectors must be taken against `base[X]`, so I read `base` once and never write back into it while merging.
  - Non-MLP equality has to be bit-exact, so I used `torch.equal` on the float16 tensors rather than an `allclose` tolerance.
  - `dense_h_to_4h` and `dense_4h_to_h` share a prefix, so the key regex is anchored and alternates explicitly to avoid over- or under-matching; the count assertion (64) is what actually guards it.
  - Accumulation in float32 then a single cast back to float16 avoids compounding two float16 roundings.
- Anything in the task text or documentation that was unclear: nothing material. The task lists the MLP tensor names explicitly, which made the 64-tensor selection unambiguous.
- Tools used (condition F):
  - `torch` 2.14.0 — float32 arithmetic and bit-exact `torch.equal` comparisons.
  - `safetensors` 0.5.3 — `load_file` / `save_file` for the three inputs and the single output.
  - Why not mergekit: its task-arithmetic path is a YAML-driven whole-model merge that reads/writes HF model directories and would apply the arithmetic to all 244 tensors. This task requires the opposite — merge exactly 64 tensors, keep the other 180 bit-identical, and abort if the fine-tunes differ anywhere outside the MLP. Encoding that precondition check and the 64/244 assertions in mergekit is not possible without wrapping it in a script anyway, so a direct script is both shorter and the only version where the required checks genuinely fail the run.
- Approximate time spent: ~5 minutes.
