# T2 participant self-report (condition F, Pythia-1B)

- Final artifact path: `out/T2/solution.py` (produces `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: failed_assertion. My own extra check "all tensors must stay float16" fired because the 16 `attention.bias` causal-mask buffers are stored as U8 (bool), not float16. The task's required checks were not affected; no output was written. Replaced with a dtype check on the 48 edited projection tensors only.
- Pitfalls or surprises you hit (one line each):
  - The `attention.bias` buffers are U8 in the checkpoint, so a blanket float16 assertion is wrong; the task text says "float16" for the checkpoint as a whole.
- Anything in the task text or documentation that was unclear:
  - Nothing material. The row/column ranges in "Required result" are explicit enough that the interleaved q/k/v layout only matters for deriving the 768-row block size.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: `load_file` / `save_file` for the checkpoint, bit-exact I/O with no model instantiation.
  - `torch` 2.14.0: `index_select` on the head axis and `torch.equal` for the post-write verification.
  - Not used: transformers `prune_heads`. It would require loading the model, changing `num_attention_heads` in the config, and trusting its internal slicing; a direct slice on the state dict is shorter, keeps every untouched tensor byte-identical, and the checks are explicit.
- Approximate time spent, if you can tell: about 3 minutes.

## What the script enforces before writing

- Input has 244 tensors; every edited tensor has its expected pre-prune shape.
- Layer 0 `query_key_value.weight` is `[5376, 2048]`, `query_key_value.bias` is `[5376]`, `dense.weight` is `[2048, 1792]` (the four required checks, plus the same for all 16 layers).
- Output tensor count is 244; edited tensors remain float16; destination does not already exist.

After writing, the script reloads the output and checks every tensor against the source: edited tensors equal `cat(rows 0..3839, 4608..6143)` / `cat(cols 0..1279, 1536..2047)`, everything else is `torch.equal` to the input.
