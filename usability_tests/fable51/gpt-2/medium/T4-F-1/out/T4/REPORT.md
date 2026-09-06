# Participant self-report: T4 (GPT-2 124M), condition F

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None. Task vectors are computed against the untouched base dict, so ordering is not an issue; tensors are made contiguous before `save_file`.
- Anything in the task text or documentation that was unclear:
  - Nothing material. `inputs/lora/` is present but irrelevant to T4.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: load the three checkpoints and save the merged one with an exact key set.
  - `torch` 2.14.0: `torch.equal` for the bit-exact shared-tensor check and float32 arithmetic for the merge.
  - Plain script rather than mergekit: mergekit's task-arithmetic method cannot express the required precondition (verify all non-MLP tensors identical across three checkpoints, abort otherwise) or the "exactly 48 merged / exactly 160 out" assertions, and it may re-layout or shard the output, which would jeopardize the exact key set and bit-exact unchanged tensors the grader requires.
- Approximate time spent, if you can tell: about 3 minutes.

Checks enforced by `solution.py` (each exits non-zero on failure): identical name sets across base/ft1/ft2; 160 tensors in base; the 48 MLP names match the expected set exactly; every non-MLP tensor bit-equal (`torch.equal`) in all three, with shape/dtype checks; float32 dtype of MLP tensors; exactly 48 merged; exactly 160 output tensors with the base key set; re-read of the written file confirms 160 tensors and bit-exact unchanged tensors.
