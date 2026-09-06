# T2 self-report (condition P)

- Final artifact path: `out/T2/model.safetensors` (script: `out/T2/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The fused `query_key_value` is GPT-NeoX interleaved (per head `[q|k|v]` in a 768-row block), so head 5 is one contiguous 768-row slice — a `[q|k|v]`-segment assumption would have required three separate slices and produced wrong rows.
  - The output projection prunes the *input* axis (columns), not rows, because of the `nn.Linear` `[out, in]` layout.
  - Sliced tensors must be `.contiguous()` before `save_file`, otherwise safetensors can reject or misserialize non-contiguous views.
- Anything in the task text or documentation that was unclear: nothing; the layout and kept-index ranges were spelled out explicitly.
- Tools used (condition F): n/a (condition P: torch + safetensors only).
- Approximate time spent, if you can tell: ~3 minutes.
