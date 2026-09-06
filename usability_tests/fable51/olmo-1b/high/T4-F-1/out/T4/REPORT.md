# T4 self-report (condition F, olmo-1b)

- Final artifact path: `out/T4/solution.py` (writes `out/T4/model.safetensors`, 114 tensors, float32)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed all checks (about 14 s wall clock).
- Pitfalls or surprises you hit (one line each):
  - The base is sharded (two files plus an index) while ft1/ft2 are single files, so the script needs a small wrapper that maps tensor name to shard; it also cross-checks shard contents against the index.
  - Output must be a single file, so the sharded-export path of `save_pretrained` or mergekit was not appropriate; `safetensors.torch.save_file` writes one file directly.
- Anything in the task text or documentation that was unclear: nothing material. The task does not say whether an existing `out/T4/model.safetensors` should be an error; the script overwrites it.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: `safe_open` for lazy per-tensor reads of all three checkpoints (avoids holding 15 GB in memory), `save_file` for the single-file output; post-write reopen to verify the file on disk.
  - `torch` 2.14.0: float32 arithmetic for the task vectors and `torch.equal` for bit-exact comparison of the 66 shared tensors.
  - Not used: `mergekit` task arithmetic. It can express `base + 0.4*(ft1-base) + 0.4*(ft2-base)`, but it cannot enforce the required step-1 precondition (non-MLP tensors bit-identical across all three checkpoints), nor the "exactly 48 merged" count, and it writes sharded output with its own dtype handling. A ~150-line script that reads each tensor once is simpler to audit than a mergekit YAML plus a separate verification script around it.
- Approximate time spent, if you can tell: roughly 5 minutes including environment inspection.

## What the script enforces (fails with a non-zero exit on violation)

1. Base shard contents match `model.safetensors.index.json`; base, ft1 and ft2 have identical tensor-name sets; there are exactly 114 names and all 48 MLP names are present.
2. Every one of the 66 non-MLP tensors has identical shape, dtype and bit-exact values in base, ft1 and ft2 (checked before any arithmetic).
3. The 48 MLP tensors agree on shape and dtype (F32) across the three checkpoints.
4. Each task vector is computed against the untouched base tensor: `b + 0.4*(ft1-b) + 0.4*(ft2-b)` in float32; exactly 48 tensors merged.
5. The assembled output has exactly 114 tensors with the input names; after writing, the file is reopened and re-verified (114 keys, shared tensors bit-identical to base, a sample of merged tensors equal to the computed values).

An independent float64 recomputation of three merged tensors against the written file gave relative Frobenius errors at float32 rounding level (see console output at the end of the session).
