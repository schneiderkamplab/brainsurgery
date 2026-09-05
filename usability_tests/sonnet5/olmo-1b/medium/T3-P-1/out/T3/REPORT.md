## Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the script succeeded on the first run.
- Pitfalls or surprises you hit (one line each):
  - Building the regex for the 112 projection matrices carefully to avoid matching `model.embed_tokens.weight` / `lm_head.weight` (both end in `.weight` too).
  - `embed_tokens.weight` and `lm_head.weight` are each 412 MB (float32), well over the 256 MiB shard cap, so they each need their own shard; the greedy packer had to special-case tensors larger than the cap.
  - Tensors from `safe_open(...).get_tensor(...)` needed `.contiguous()` before `save_file` to avoid a safetensors save error.
- Anything in the task text or documentation that was unclear: none; the per-tensor name list and shard-size rule in TASK.md were sufficient to implement directly.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: ~10 minutes.
