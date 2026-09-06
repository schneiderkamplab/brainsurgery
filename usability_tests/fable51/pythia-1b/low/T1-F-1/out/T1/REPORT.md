# T1 self-report (condition F)

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`, 184 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - None. Building a fresh dict keyed by new names (rather than renaming in place) avoids the renumbering-collision hazard entirely; an explicit collision assert guards it anyway.
- Anything in the task text or documentation that was unclear:
  - The "Required checks" list says "no tensor of blocks 12, 13, 14, 15 remains" while the removed blocks are 2, 6, 10, 14; I read it as "no index >= 12 after renumbering" and checked both that and the exact index set 0..11.
- Tools used (condition F): name, version, and why:
  - safetensors 0.5.3 (`safe_open` for lazy reads, `save_file` for output): the task is a pure key rename/delete, so a direct safetensors script is the shortest and most auditable route; no need for mergekit or torch-state-bridge.
  - torch 2.14.0: tensor container only (`.contiguous()` before saving).
- Approximate time spent, if you can tell: about 2 minutes.

Checks enforced by the script before any file is written: input has 244 tensors, no name collisions, block indices are exactly 0..11, no index >= 12, exactly 12 `attention.query_key_value.weight` tensors, each block has 15 tensors, total 184 tensors, output path does not already exist. A separate post-hoc comparison confirmed the output matches the source bit-exactly under the specified renumbering.
