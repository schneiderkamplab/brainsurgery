# Participant self-report: T4 (condition P)

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The regex for the 64 MLP tensors is anchored on `mlp.dense_h_to_4h` / `mlp.dense_4h_to_h` so it cannot pick up attention `dense` tensors.
  - Both task vectors are computed from the same float32 copy of the base before either is added, so ordering cannot bias the result.
  - Tensors are made contiguous before `save_file`; the base is a symlink into a read-only tree, which was fine for reading.
- Anything in the task text or documentation that was unclear: nothing; the formula, dtype rule and checks were explicit.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
