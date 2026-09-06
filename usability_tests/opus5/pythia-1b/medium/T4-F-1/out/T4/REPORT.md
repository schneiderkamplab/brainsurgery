# T4 participant self-report

- **Final artifact path:** `out/T4/solution.py` (output at `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the first execution succeeded.
- **Pitfalls or surprises you hit:**
  - The ordering hazard is easy to write wrong: both task vectors must be taken
    against the original `base[X]`, so I materialised `b32` once and reused it
    rather than accumulating into a running output tensor.
  - Three 2 GB float16 checkpoints do not want to be resident at once; I streamed
    them tensor-by-tensor through `safe_open` instead of `load_file`.
  - `save_file` rejects non-contiguous/shared storage, so every tensor is
    `.clone().contiguous()` before it goes into the dict.
  - `mergekit`'s `task_arithmetic` was the advertised route but would have made
    the required checks harder, not easier (see below).
- **Anything in the task text or documentation that was unclear:** nothing
  material. "Identical" in step 1 I read as bit-exact (`torch.equal`), which is
  consistent with grading demanding bit-exact unchanged tensors.
- **Tools used (condition F):**
  - `torch` 2.14.0 — float32 arithmetic and `torch.equal` for the bit-exact
    frozen-backbone verification.
  - `safetensors` 0.5.3 — lazy `safe_open` reads of the three inputs, `save_file`
    for the single-file output, and a re-open of the written file to confirm the
    key set and count actually on disk.
  - **Considered and rejected: `mergekit` 0.1.4 `task_arithmetic`.** It is the
    natural fit for the arithmetic itself, but it operates on HF model
    directories, re-serialises *every* tensor through its own dtype/sharding
    path (risking the bit-exact requirement on the 180 unchanged tensors and the
    single-file output), and offers no hook to assert the frozen-backbone
    precondition, the "exactly 64 merged" count, or the 244-tensor total. Those
    three assertions are the substance of this task, so a ~100-line script that
    enforces them directly was both shorter and safer than a YAML config plus a
    post-hoc verifier.
- **Approximate time spent:** ~10 minutes, of which the merge run itself was 10 s.
