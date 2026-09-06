# T4 run report (condition F, Pythia-1B)

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)

- **Number of times you executed the script or plan:** 2 (1 failure, then success)

- **Which executions failed, and why (one line each):**
  1. `crash` — `RuntimeError: self.dim() cannot be 0 to view Half as Byte (different element sizes)`: my bit-exact comparison bit-cast each tensor to `uint8`, which torch rejects for the 0-dim scalar tensors in the checkpoint. Fixed by `reshape(-1)` before the view.

- **Pitfalls or surprises you hit (one line each):**
  - The checkpoint contains 0-dim scalar tensors, which cannot be bit-cast to a wider dtype; any byte-level identity check has to flatten first.
  - 16 of the 244 tensors are `uint8` (the `attention.bias` causal-mask buffers), not float16, so "cast back to the base dtype" has to mean each tensor's own dtype, not a blanket `.half()`.
  - The ordering hazard is real and not absorbed by the tolerance: computing the second task vector against the already-merged tensor differs from the correct result by ~6e-3 relative Frobenius error, i.e. 6x the 1e-3 grading tolerance. I avoided it by re-reading the base tensor and accumulating both deltas in one float32 expression, with a single cast at the end.
  - `torch.equal` is the wrong identity test for a "these must be bit-identical" precondition (NaN != NaN), so I compared raw bytes instead.

- **Anything in the task text or documentation that was unclear:** Nothing blocking. "cast back to float16 (the base dtype)" reads as if the whole checkpoint were float16; it is not (16 uint8 buffers), but those are all outside the merge set, so the instruction is unambiguous in practice.

- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — streaming per-tensor reads via `safe_open` (never holds three 2 GB checkpoints in memory at once) and `save_file` for the output.
  - `torch` 2.14.0+cu130 — float32 arithmetic, dtype casts, and byte-level tensor comparison.
  - I deliberately did **not** use `mergekit`, the route the condition suggests for T4. Its `task_arithmetic` method would compute the merge, but it does not verify the task's actual precondition (that all 180 non-MLP tensors are bit-identical across the three checkpoints) and it does not let me restrict the merge to exactly 64 named tensors and take everything else bit-exact from the base — it merges every tensor it sees. Since the required checks would have had to be a separate script on top of mergekit anyway, and that script needs to read all three checkpoints tensor-by-tensor regardless, adding mergekit would have meant a YAML config, an HF-model-directory output shape and a second verification pass for no gain. A ~200-line script over safetensors + torch expresses the whole task, including all three required checks, in one artifact.

- **Verification performed:** the script itself aborts (non-zero exit, `MergeError`) on: key-set mismatch between any pair of checkpoints, input tensor count != 244, any missing or unexpected `.mlp.` tensor, per-tensor shape/dtype disagreement, any non-MLP tensor differing bit-wise between base and either fine-tune, merged count != 64, non-finite merged values, output tensor count != 244, and a post-write re-read confirming 180 tensors bit-identical to the base and 64 changed. I additionally ran two out-of-band checks: (a) recomputing three merged tensors independently gives relative error ~2e-4 (fp16 rounding, within the 1e-3 tolerance); (b) a self-test that pointed the script at tiny synthetic checkpoints confirmed the backbone-drift and missing-name guards actually fire rather than passing silently.

- **Approximate time spent, if you can tell:** ~10 minutes, of which the two runs took ~15 s total.
