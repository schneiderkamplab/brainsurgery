# T1 self-report (condition P)

- **Final artifact path:** `out/T1/solution.py` (output: `out/T1/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The renumbering collision hazard the task warns about disappears if you build a *new* dict keyed by new names instead of mutating in place; I did that and additionally asserted that no new name is ever written twice.
  - Input is sharded, so tensors have to be gathered from both shards via `model.safetensors.index.json` before renaming; I also guard against the same key appearing in two shards.
  - `save_file` rejects non-contiguous/shared storage, so I called `.contiguous()` on every tensor before writing.
  - All checks run before `save_file`, so a failure leaves no partial output.
- **Anything in the task text or documentation that was unclear:** nothing material. The required checks mention "no tensor of blocks 12..15 remains", which for a 0..11 result is equivalent to "no block index >= 12"; I implemented the general form. The task says a single `model.safetensors` with no index file, which I took literally (no `model.safetensors.index.json` and no config/tokenizer copies written).
- **Tools used (condition F):** n/a (condition P: torch 2.14.0 + safetensors 0.5.3 only).
- **Approximate time spent:** ~3 minutes.
