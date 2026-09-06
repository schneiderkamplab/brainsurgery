# Participant self-report: T5 (GPT-2 124M), condition P

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`: 5 shards plus `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `fan_in_fan_out = true` means the base weight is Conv1D `[in, out]`, so `scale * (B @ A)` (`[out, in]`) must be transposed before adding; the script reads the flag from `adapter_config.json` rather than hardcoding the transpose.
  - The PEFT prefix `base_model.model.` has to be stripped to map `...lora_A.weight` / `...lora_B.weight` onto `h.<i>.attn.c_attn.weight`.
  - `wte.weight` (154 MB) exceeds the 100 MiB shard budget on its own, so the greedy packer must allow a single oversized tensor in a shard by itself; it landed last in a shard of its own because it is the last key in the base file's order.
  - The merged tensor is made contiguous before saving so safetensors does not reject it.
- Anything in the task text or documentation that was unclear:
  - `adapter_config.json` lists `target_modules = ["c_attn"]` while TASK.md says `["attn.c_attn"]`; this did not matter because the script derives the target names from the adapter tensor names, not from the config.
  - Shard file naming and index `metadata.total_size` are not specified; I used the HF convention `model-0000N-of-0000M.safetensors` and included `total_size`.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 3 minutes
