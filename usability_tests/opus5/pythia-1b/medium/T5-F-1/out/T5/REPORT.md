# T5 self-report (condition F, pythia-1b, medium)

- **Final artifact path:** `out/T5/solution.py` (output checkpoint in `out/T5/`:
  4 shards + `model.safetensors.index.json`).
- **Number of times you executed the script or plan:** 1.
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - PEFT name prefix: adapter keys carry `base_model.model.` and a
    `.lora_A/.lora_B.weight` suffix that must be stripped to recover the base
    name `gpt_neox.layers.<i>.attention.query_key_value.weight`.
  - `fan_in_fan_out = false` with an `nn.Linear` `[out, in]` base means `B @ A`
    is added directly; I still read the flag from `adapter_config.json` and
    transpose if it is ever true, rather than hardcoding.
  - Scale is `lora_alpha / r = 2`, not 1 and not `alpha`.
  - Mixed dtypes: base is float16 (plus uint8 `attention.bias` causal masks),
    adapter is float32; the merge is done in float32 and cast back to the base
    tensor's own dtype.
  - The task text calls `embed_in`/`embed_out` (206 MB each) "larger than
    512 MiB" — they are not, so under the stated 512 MiB budget they legitimately
    share a shard with other tensors. I implemented the stated rule (each shard
    ≤ 512 MiB of tensor data; any single tensor above the budget gets its own
    shard) rather than the parenthetical.
- **Anything unclear:** only the `embed_*` parenthetical above; also the task
  does not say whether shard file naming must follow the HF
  `model-0000k-of-0000N.safetensors` convention, so I used it as the safe default.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — direct load/save of the checkpoint files.
  - `torch` 2.14.0 — the `B @ A` matmul and dtype casts.
  - `huggingface_hub` (pinned) — `split_torch_state_dict_into_shards` for the
    512 MiB shard plan and the `weight_map`, so the sharding follows the
    standard HF convention instead of hand-rolled packing.
  - I deliberately did **not** use `peft.merge_and_unload`: it instantiates the
    full model, upcasts/reloads weights through `transformers`, and would make
    bit-exactness of the 228 untouched tensors depend on the model class's
    load/save path. Operating on the raw state dict keeps untouched tensors
    bit-identical and keeps the required checks explicit.
- **Required checks:** enforced in `solution.py` before anything is written —
  exactly 16 complete adapter pairs merged, no `lora_` key in the output,
  `gpt_neox.layers.0.attention.query_key_value.weight` is `[6144, 2048]` float16,
  and exactly 244 output tensors; plus per-shard budget and weight_map coverage
  checks at write time. Each raises and aborts the run.
- **Approximate time spent:** ~5 minutes.
