# T4 participant self-report (condition F, Pythia-1B)

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None. The inputs matched the task description exactly (244 names, 180 shared tensors bit-identical, 64 MLP tensors in layers 0..15).
- Anything in the task text or documentation that was unclear:
  - Nothing material. "Identical" for the shared tensors was interpreted as bit-identical (`torch.equal`, plus shape and dtype), which is also what the grader checks.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: lazy per-tensor reads via `safe_open` so all three 2 GB checkpoints never need to be fully resident at once; `save_file` for the output.
  - `torch` 2.14.0: float32 arithmetic for the merge, `torch.equal` for the bit-exact shared-tensor check.
  - Not used: `mergekit` task arithmetic. It would have computed the merge, but it does not verify the shared-tensor precondition, does not restrict the merge to the 64 MLP tensors (it would apply task vectors to all 244, which is equivalent here only because the others are identical), and does not enforce the required counts. A ~100-line script gives all three checks explicitly and fails loudly.
- Approximate time spent, if you can tell: about 3 minutes. The script runs in under 10 seconds.

## What the script enforces

1. Same tensor name set in base, ft1 and ft2; exactly 244 names.
2. Exactly 64 names match the MLP pattern, covering layers 0..15.
3. Every tensor agrees in shape and dtype across the three files; every non-MLP tensor is bit-identical (`torch.equal`) in all three.
4. Merge: `base + 0.4*(ft1-base) + 0.4*(ft2-base)` in float32, both task vectors taken against the unmodified base, cast back to float16.
5. Merged count == 64, output dict size == 244, and the written file is re-opened to confirm 244 tensors with the same key set. Refuses to run if the output file already exists.

An independent post-hoc check (not part of the artifact) found the worst relative Frobenius error of the merged tensors against a float32 recomputation to be 2.4e-4, from float16 rounding, and all 180 other tensors bit-identical to base.
