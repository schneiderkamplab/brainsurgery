# T4 — participant self-report

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none — the single execution
  passed all checks and wrote the output.
- **Pitfalls or surprises you hit (one line each):**
  - The ordering hazard is only avoided if both task vectors are taken against the
    original `base[X]`; I built the output into a separate dict and never wrote back
    into the loaded base tensors.
  - Non-MLP tensors include the causal-mask buffers `h.<i>.attn.bias`, whose names
    contain `bias` — a name pattern like `*.bias` would have swept 12 mask buffers
    into the merge, so I enumerated the 48 MLP names explicitly instead of matching.
  - "Identical" for the 112 shared tensors is checked bit-exactly by comparing raw
    bytes rather than with `torch.equal`, which would call two NaNs unequal.
  - Loaded the three checkpoints via `safe_open` rather than `load_file` to keep only
    what is needed resident; three float32 GPT-2 copies is ~1.6 GB either way.
- **Anything in the task text or documentation that was unclear:**
  - The task says the output must have "exactly 160 tensors" but says nothing about
    the safetensors `__metadata__` block, which the base carries. I wrote
    `{"format": "pt"}` so the file loads with HF tooling; metadata is not a tensor,
    so the count is unaffected.
  - `inputs/` also contains a `lora/` directory that this task does not use; I ignored it.
- **Tools used (condition F):** n/a — condition P (plain PyTorch + safetensors).
- **Approximate time spent, if you can tell:** ~5 minutes, most of it reading the
  safetensors header to confirm the 160 names and the `attn.bias` buffers.
