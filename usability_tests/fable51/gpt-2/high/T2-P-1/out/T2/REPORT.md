# Participant self-report: T2 (GPT-2 124M), condition P

- Final artifact path: `out/T2/solution.py` (output checkpoint `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; execution 1 succeeded.
- Pitfalls or surprises you hit (one line each):
  - None blocking. Conv1D `[in, out]` layout means the head axis is the column axis for `c_attn` and the row axis for `c_proj`; the task text already spells out the exact index ranges, so I built one 704-long keep index per 768-wide segment and offset it by 0/768/1536 for the fused q|k|v axis rather than hard-coding ranges.
  - Matched tensor names exactly (`h.<i>.attn.c_attn.*`, `h.<i>.attn.c_proj.weight`) instead of a regex, to avoid touching `mlp.c_proj.weight` and the `attn.bias` mask buffer.
  - Called `.contiguous()` on every tensor before `save_file` since `index_select` results and pass-through views could otherwise be rejected.
- Anything in the task text or documentation that was unclear: nothing; the required index ranges and shapes were explicit.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: about 2 minutes.
