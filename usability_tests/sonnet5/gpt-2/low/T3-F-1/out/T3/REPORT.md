# Participant self-report — T3

- **Final artifact path:** `out/T3/solution.py`
- **Number of times you executed the script or plan:** 2 (first attempt failed
  due to a path-resolution bug in the script — `IN_PATH`/`OUT_DIR` used the
  wrong number of `parents[]` hops and pointed outside the sandbox; fixed and
  reran successfully).
- **Which executions failed, and why:**
  - Execution 1: `crash` — `FileNotFoundError` for `inputs/base/model.safetensors`,
    caused by an off-by-one in `Path.parents[...]` when computing the sandbox
    root from `out/T3/solution.py`.
- **Pitfalls or surprises you hit:**
  - None related to the actual precision/sharding logic; the exact tensor
    names for the 48 projection matrices and 12 buffers are given explicitly
    in TASK.md, so no regex overreach risk here (avoided pattern matching
    entirely, used explicit name sets instead).
  - Had to double check that `.contiguous()` is applied after `.to(bfloat16)`
    (and on the unchanged float32 tensors) since safetensors requires
    contiguous, non-shared tensors to save cleanly.
- **Anything in the task text or documentation that was unclear:** No.
- **Tools used (condition F):** `torch` 2.14.0 (dtype casts, tensor ops) and
  `safetensors` 0.5.3 (`load_file`/`save_file`) directly, no higher-level
  toolkit (mergekit, torch-state-bridge, transformers export). Chosen because
  the task only requires per-tensor dtype casts, dropping a fixed set of
  keys, and hand-rolled shard packing by a byte budget — a plain script over
  the two lowest-level packages gives exact control over dtype (round-to-
  nearest via `tensor.to(torch.bfloat16)`), the exact key set touched, and
  the shard-size logic (oversized tensor alone in its own shard) without
  fighting a higher-level API's own sharding/export policy.
- **Approximate time spent:** ~5 minutes.
