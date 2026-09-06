# T1 participant self-report (condition F)

- **Final artifact path:** `out/T1/solution.py` (invoked via `out/T1/run.sh`),
  producing `out/T1/model.safetensors`.
- **Number of times you executed the script or plan:** 1.
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - The renumbering shift is exactly "subtract the count of removed block
    indices below this one" (`bisect_left` over the sorted removed set);
    computing it this way instead of hand-writing a lookup table avoids
    transcription errors and generalizes to any removed set.
  - Sharded input: `model.safetensors.index.json`'s `weight_map` spans two
    shard files, so each shard had to be loaded once and indexed by tensor
    name rather than assuming one file holds everything.
  - Needed to `.contiguous()` every tensor before `save_file`, since tensors
    read out of a loaded shard dict can in principle be non-contiguous views;
    `safetensors` refuses to serialize a checkpoint if that's the case, or a
    shared-memory tensor is passed for two keys.
- **Anything in the task text or documentation that was unclear:** no —
  the explicit old->new mapping in TASK.md matched exactly what the
  "subtract count of prior removed blocks" formula produces, which was a
  useful cross-check.
- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — direct `load_file`/`save_file` for sharded
    load and single-file save; no need for a merge-config DSL (mergekit) or
    a rule-based rewriting layer (torch-state-bridge) since the rename rule
    is a single regex substitution and the "delete" rule is a single set
    membership test. A plain script on top of `safetensors` was the smallest
    correct tool for a rename-and-drop task with an explicit spec.
  - `torch` 2.14.0 — only for tensor equality/dtype/shape checks during
    self-verification (not required by the solution itself, only by my
    manual validation pass).
- **Approximate time spent, if you can tell:** single pass, no retries;
  on the order of a few minutes of wall-clock agent time including
  self-verification against the source tensors.
