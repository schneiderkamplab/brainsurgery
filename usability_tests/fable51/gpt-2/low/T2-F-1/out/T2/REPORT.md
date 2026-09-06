# T2 participant self-report

- Final artifact path: `out/T2/solution.py` (output `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The `attn.bias` mask buffer shares the `attn.` prefix with the head-bearing tensors; addressed tensors by exact name so it stays untouched.
  - Conv1D `[in, out]` layout: heads are columns of `c_attn` and rows of `c_proj`, the opposite of `nn.Linear`.
- Anything in the task text or documentation that was unclear: nothing; the explicit column ranges made the layout unambiguous.
- Tools used (condition F): torch 2.14.0 (`index_select` slicing), safetensors 0.5.3 (`load_file`/`save_file`). A plain script was chosen over transformers `prune_heads` because the task specifies exact index ranges and bit-exact output; direct index selection is the shortest path with no risk of `prune_heads` reordering or dtype changes. A post-hoc check compared every output tensor against the spec index ranges of the input (bit-exact); loading via transformers with n_head=11 is rejected by its constructor (768 not divisible by 11), so no HF load test was done.
- Approximate time spent, if you can tell: about 3 minutes.
