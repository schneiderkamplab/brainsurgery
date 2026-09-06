# T4 participant self-report

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each): none; the checkpoints matched the stated layout (160 names, 48 MLP tensors, non-MLP tensors bit-identical across all three)
- Anything in the task text or documentation that was unclear: nothing
- Tools used (condition F): name, version, and why:
  - torch 2.14.0: float32 arithmetic and `torch.equal` for the bit-exact shared-tensor check
  - safetensors 0.5.3: load/save of the checkpoints
  - No mergekit: its task-arithmetic merge does not enforce the "all non-MLP tensors identical" precondition or the exact-48 count, so a short script was simpler and let every required check be an explicit hard failure
- Approximate time spent, if you can tell: about 2 minutes
