# T4 participant self-report

- **Final artifact path:** `out/T4/solution.py` (invoked via `out/T4/run.sh`), output at `out/T4/model.safetensors`.
- **Number of times you executed the script or plan:** 1 (the run that produced `out/T4/model.safetensors`). I separately exercised the abort path in a throwaway `/tmp` sandbox with a deliberately corrupted non-MLP tensor to confirm the verification guard fires and exits non-zero; that test did not touch `out/`.
- **Which executions failed, and why:** none of the executions against the real `out/T4` output failed.
- **Pitfalls or surprises you hit:**
  - `h.<i>.attn.bias` is a non-trainable causal-mask buffer, not a parameter — it's outside the 48 MLP tensors but easy to overlook when reasoning about "everything else must be identical"; the shared-tensor verification loop treats it like any other non-MLP tensor and it passed.
  - GPT-2's `Conv1D` layout for `c_fc`/`c_proj` weights is `[in, out]`, not the `[out, in]` of `nn.Linear`; the task spec already states the exact shapes so this only mattered for sanity-checking, not for the arithmetic itself (add/scale is shape-agnostic).
  - Ordering hazard called out in the task text (each task vector must be taken against the unmodified base) is naturally satisfied by loading all three checkpoints up front and never overwriting `base` in memory before both task vectors are computed.
- **Anything in the task text or documentation that was unclear:** no — inputs, formula, and required checks were unambiguous.
- **Tools used (condition F):** `torch` 2.14.0 and `safetensors` 0.5.3 only, via a plain script. I considered `mergekit`'s task-arithmetic YAML merge method, but the task's step-1 precondition (verify all non-MLP tensors are bit-identical across all three checkpoints, abort otherwise) and the required "exactly 48 / exactly 160 tensors" checks are not something mergekit's merge configs express directly — they'd need a wrapper script around it anyway, so a single plain script using `safetensors.safe_open`/`save_file` and `torch` tensor ops was simpler, more auditable, and gave direct control over abort behavior and exact tensor counts.
- **Approximate time spent:** a few minutes — one script written and verified on the first run.
