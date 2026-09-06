# Participant self-report: T1 (OLMo-1B-0724-hf, condition P)

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`, 86 tensors, 12 blocks)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Avoided the renumbering collision hazard by building a fresh dict from the source instead of renaming in place, and by asserting on any destination-name collision.
  - Input is sharded; loaded both shards via `model.safetensors.index.json` and asserted no duplicate keys across shards.
  - Called `.contiguous()` on every tensor before `save_file` as a precaution against non-contiguous storage.
- Anything in the task text or documentation that was unclear: nothing material. The "Required checks" list says "no tensor of blocks 12, 13, 14, 15 remains", which is the post-renumbering condition (indices >= 12); I implemented exactly that plus contiguity 0..11 and 7 tensors per block.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes.
