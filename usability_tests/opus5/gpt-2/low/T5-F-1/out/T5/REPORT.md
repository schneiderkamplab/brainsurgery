# T5 self-report

- **Final artifact path:** `out/T5/solution.py` (output in `out/T5/`, 5 shards + `model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none.
- **Pitfalls or surprises you hit:**
  - The Conv1D `[in, out]` layout vs the adapter's `[out, in]` product: `fan_in_fan_out = true` means `(B @ A)` must be transposed before adding. I read it off the config rather than hardcoding it.
  - PEFT name prefix `base_model.model.` has to be stripped to reach the base names; `adapter_config.json` says `target_modules = ["c_attn"]` (bare) while TASK.md says `attn.c_attn`, so I derived target names from the adapter keys instead of from `target_modules`.
  - `wte.weight` (154 MB) exceeds the 100 MiB shard budget on its own, so the greedy packer needs an explicit "oversized tensor goes alone" branch. It happens to be last in key order, so it landed in its own final shard cleanly.
- **Anything unclear:** the shard budget is stated as tensor data excluding headers, but the shard file naming convention is not specified; I used the HuggingFace `model-000ii-of-000nn.safetensors` convention.
- **Tools used (condition F):** `torch` 2.14.0 and `safetensors` 0.5.3 only, in a plain script. I considered `peft.merge_and_unload`, but that route instantiates a GPT-2 model, and the required output is a *custom* sharding (100 MiB tensor-data budget, oversized tensor alone) that `save_pretrained`'s `max_shard_size` accounting does not match exactly; it would also risk touching dtypes and tied weights. The merge itself is four lines of tensor algebra, so a direct file-level rewrite was both shorter and easier to check.
- **Verification I ran:** a separate reader script re-loaded the output through the index, confirmed the key set equals the base's 160 names, that exactly the 12 `c_attn` weights differ from the base bit-for-bit, that each differs from an independently recomputed `W + 2*(B@A).T` by relative error 0.0, and that every shard is within budget except the lone `wte.weight`.
- **Approximate time spent:** ~5 minutes.
