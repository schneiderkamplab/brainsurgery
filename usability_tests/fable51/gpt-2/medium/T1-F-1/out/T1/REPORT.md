# T1 (GPT-2 124M, condition F) — participant self-report

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`, 121 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The "no tensor of blocks 9, 10, 11 remains" check must be evaluated on the renumbered result, not the input (old blocks 9–11 survive as new 6–8); the script checks `index >= 9` on output names.
  - Avoided the collision hazard by building a fresh dict from an explicit old→new map instead of renaming in place; a collision check fails the run anyway.
  - The mask buffer `attn.bias` is a legitimate block tensor and must be carried along (13 tensors per block).
  - `save_file` requires contiguous tensors; loaded tensors already are, `.contiguous()` is a no-op guard.
- Anything in the task text or documentation that was unclear: nothing material.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3 (`load_file`/`save_file`): direct, dependency-free read and write of the checkpoint.
  - `torch` 2.14.0: tensor equality in the post-write verification.
  - A plain Python script rather than mergekit or torch-state-bridge: the task is a 9-line rename with an explicit index map, and a script lets the required checks (block ≥ 9 absent, exactly 9 `c_attn.weight`, 121 tensors, no collisions, values carried verbatim, re-read of the written file) fail the run before anything is written.
- Approximate time spent, if you can tell: about 2 minutes.
