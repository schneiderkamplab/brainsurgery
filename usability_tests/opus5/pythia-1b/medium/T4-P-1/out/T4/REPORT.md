# T4 (condition P) — participant self-report

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The ordering hazard is real: both task vectors have to be computed against the base tensor read once, so I materialise `b32` first and derive `tv1`/`tv2` from it rather than accumulating into a running result.
  - Bit-exactness for the 180 unchanged tensors means the comparison must be `torch.equal` on the raw fp16 values (no tolerance) and the copy must be the base tensor untouched, not a float32 round-trip.
  - `safe_open` lazily per key keeps three 2 GB checkpoints from being resident at once; loading all three state dicts eagerly would be wasteful.
  - safetensors rejects non-contiguous/shared storage on save, so every output tensor is `.clone().contiguous()`.
  - The base file's `__metadata__` is preserved so the output is a drop-in checkpoint.
- **Anything in the task text or documentation that was unclear:**
  - The task says "verify that every tensor outside the 64 MLP tensors is identical in all three" but does not say whether shape/dtype agreement of the MLP tensors themselves must also be checked; I check it, since the subtraction would otherwise fail obscurely.
  - `inputs/` also contains a `lora/` directory that this task does not use; it was not clear it is intentionally irrelevant to T4.
- **Tools used (condition F):** n/a (condition P: torch 2.14.0 + safetensors 0.5.3 only)
- **Approximate time spent, if you can tell:** ~5 minutes.
