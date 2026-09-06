# Participant self-report — T4 (condition F)

- **Final artifact path:** `out/T4/solution.py` (driven by `out/T4/run.sh`), output at `out/T4/model.safetensors`.
- **Number of times you executed the script or plan:** 1.
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - None specific to this run; the tensor-name regex for MLP tensors
    (`dense_h_to_4h` / `dense_4h_to_h`) needed the `\.` escapes and an
    explicit layer-index bound (`< 16`) to avoid accidentally matching
    something outside the declared 16 layers, but the input checkpoints
    matched the spec exactly so this never triggered in practice.
- **Anything in the task text or documentation that was unclear:** no.
- **Tools used (condition F):** plain Python on top of `torch` (2.14.0) and
  `safetensors` (0.5.3) only. I considered `mergekit`'s task-arithmetic merge
  method, but it operates on full HF model directories/configs and doesn't
  give an easy hook to enforce the required "abort if non-MLP tensors
  differ" precondition check as a hard gate before writing output; a direct
  script over `safetensors`+`torch` made the three required checks (shared
  key set, exactly 64 merged, exactly 244 output tensors) explicit,
  auditable, and guaranteed to abort loudly if violated. The arithmetic
  itself is a straightforward vectorized reimplementation of task-vector
  addition, done in float32 and cast back to float16 as required.
- **Approximate time spent:** ~10 minutes.
