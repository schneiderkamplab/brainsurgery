# T4 participant self-report (condition P)

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed all checks and wrote 114 tensors.
- Pitfalls or surprises you hit (one line each):
  - The base is sharded across two files while ft1/ft2 are single files, so I wrote a small lazy reader that resolves shards via the index file and reads each tensor on demand rather than loading three 5 GB checkpoints into memory.
  - Both task vectors must be computed against the same unmodified base tensor; I kept the base tensor `b` in a local and computed `b + lam*(ft1-b) + lam*(ft2-b)` in one expression so no in-place update could leak into the second delta.
  - Shared-tensor check is done with `torch.equal` (bit-exact) on all 66 non-MLP tensors before any merge, plus shape/dtype checks, and the script verifies the written file's key set after saving.
  - Minor: the name-set comparison contains a redundant `kb ==` term (`kb == k1 == kb == k2`); it is still a correct three-way equality check, so I left it rather than spend another execution.
- Anything in the task text or documentation that was unclear: nothing material. The task says "48 MLP tensors" and the grading tolerance (relative Frobenius 1e-5) makes the summation order irrelevant, which was clear.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3 only)
- Approximate time spent, if you can tell: about 3 minutes including one 17 s execution.
