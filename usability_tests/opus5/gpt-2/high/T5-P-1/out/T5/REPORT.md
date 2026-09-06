# T5 — participant self-report (condition P)

- **Final artifact path:** `out/T5/solution.py` (output: `out/T5/model-0000{1..5}-of-00005.safetensors` + `out/T5/model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none — the single execution passed all checks.
- **Pitfalls or surprises you hit (one line each):**
  - Conv1D `[in, out]` vs Linear `[out, in]`: `B @ A` is `[2304, 768]` but the base is `[768, 2304]`, so the product has to be transposed before adding — I asserted the shape relation rather than trusting it, so a wrong-way transpose would fail loudly instead of silently broadcasting.
  - PEFT name prefix: adapter keys carry `base_model.model.` and a `.lora_A/.lora_B` infix; the mapping to base names is prefix strip + infix collapse to `.weight`, and I verified each mapped name actually exists in the base.
  - Scale is `lora_alpha / r = 2`, not `lora_alpha` — read from `adapter_config.json` instead of hardcoding.
  - `wte.weight` (154 MB) exceeds the 100 MiB shard budget on its own; the greedy packer handles it only because an oversized tensor forces the *next* tensor into a new shard, leaving it alone. I made the shard-budget assertion exempt single-tensor shards rather than exempting `wte` by name.
  - `(B @ A).T` is non-contiguous; `base + delta` already returns a contiguous tensor, but I called `.contiguous()` anyway since safetensors rejects non-contiguous/shared storage.
  - The 12 `attn.bias` causal-mask buffers are part of the 160 tensors and are pure passthrough — they must not be confused with `attn.c_attn.bias`, which is also untouched here (LoRA adapts the weight only).
- **Anything in the task text or documentation that was unclear:**
  - The task fixes the shard *budget* but not the shard *ordering* or file-naming convention, so the exact tensor-to-shard partition is not determined by the spec. I preserved the base checkpoint's own key order, packed greedily, and used the HuggingFace `model-{k:05d}-of-{n:05d}.safetensors` convention. If the hidden reference compares the `weight_map` partition byte-for-byte rather than checking the stated rule, a different-but-valid packing could be marked wrong.
  - "No intermediate tensor appears in the output" is unambiguous but only matters for an approach that stages `B @ A` as a named tensor; a plain in-memory script never creates one.
- **Tools used (condition F):** n/a — condition P (torch 2.14.0, safetensors 0.5.3, stdlib only).
- **Approximate time spent, if you can tell:** ~4 minutes: read the two safetensors headers straight out of the files with shell (`od` for the header length, `head`/`tail` to slice the JSON) to confirm names/shapes/dtypes without spending an execution, then wrote and ran the script once.
