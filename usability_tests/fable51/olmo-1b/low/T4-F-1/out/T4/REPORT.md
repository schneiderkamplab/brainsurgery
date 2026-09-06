# T4 participant self-report

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; execution 1 passed all checks.
- Pitfalls or surprises you hit (one line each):
  - None. The base is sharded, so I reassembled it from `model.safetensors.index.json` and checked shard keys against the index.
- Anything in the task text or documentation that was unclear:
  - Nothing material. I took "identical" in step 1 to mean bit-exact equality of shape, dtype and values.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: load the three checkpoints and save the single-file output.
  - `torch` 2.14.0: `torch.equal` for the bit-exact precondition, float32 arithmetic for the merge.
  - Not used: `mergekit` task arithmetic. It would have computed the merge, but it does not enforce the shared-tensor precondition, the exact-48 merge count or the 114-tensor output count, so a script wrapping it would still have been needed; a plain script was smaller and enforces all checks in one place.
- Approximate time spent, if you can tell: about 3 minutes, most of it the ~1 minute run over 3x5 GB of float32 weights.

## What the script enforces

1. Same tensor name set in base, ft1, ft2, and the base shards match the index.
2. Exactly 48 tensors match the MLP pattern, covering layers 0..15.
3. Every one of the 66 non-MLP tensors is bit-exact identical in all three (`torch.equal`, plus shape/dtype).
4. Merge `base + 0.4*(ft1-base) + 0.4*(ft2-base)` per MLP tensor in float32, each task vector taken against the unmodified base.
5. Merged count == 48, output count == 114, then the written file is reloaded and re-verified (names, shapes, dtypes, unchanged tensors bit-exact to base).
Any failure raises `SystemExit` with an `ERROR:` message.
