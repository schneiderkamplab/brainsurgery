# Participant self-report: T2 (GPT-2 124M, condition P)

- Final artifact path: `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The mask buffer `attn.bias` shares the `attn.` prefix with the head-bearing tensors, so I addressed tensors by exact name per layer instead of a pattern.
  - Conv1D `[in, out]` layout means the fused qkv heads live on dim 1 of `c_attn.weight` but on dim 0 of `c_proj.weight`; built one 704-index keep list and reused it (offset by 768 per q/k/v segment) for all three tensors.
- Anything in the task text or documentation that was unclear: nothing; the explicit column/row ranges made the layout unambiguous.
- Tools used (condition F): not applicable (condition P: torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: about 2 minutes.
