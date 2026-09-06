# T5 self-report

- **Final artifact path:** `out/T5/solution.py`
- **Number of times you executed the script or plan:** 2 (the second run was a
  re-run after a cosmetic cleanup of a redundant assertion; both runs succeeded)
- **Which executions failed, and why:** none.
- **Pitfalls or surprises you hit:**
  - PEFT name prefix: adapter keys carry `base_model.model.` and the
    `.lora_A/.lora_B.weight` suffix, so the base name has to be recovered by
    stripping both; I did it with one regex rather than string slicing.
  - `fan_in_fan_out` is false here and the factors are already in `nn.Linear`
    `[out, in]` layout, so `B @ A` is added untransposed; I still read the flag
    from `adapter_config.json` and transpose conditionally rather than
    hardcoding it.
  - Scale is `lora_alpha / r = 2`, not `1` and not `alpha`; the accumulation is
    done in float32 and cast back to the base float16 only at the end.
  - Sharding budget applies to tensor data, not file size on disk: the shard
    files are slightly above 512 MiB because of the safetensors header, so the
    check has to sum `numel * element_size`, not `os.path.getsize`.
  - Both 206 MB embedding tensors fit under the 512 MiB budget, so the
    "oversized tensor alone in its shard" case never triggered here; the
    post-write check still asserts it would hold.
- **Anything unclear:** the task does not fix a shard file naming scheme or the
  index `metadata` block; I used the HuggingFace convention
  (`model-<i>-of-<n>.safetensors` plus `metadata.total_size`). Nor does it fix
  the order tensors are packed in; I kept the base checkpoint's key order and
  filled shards greedily.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — lazy `safe_open` read of the base and the adapter,
    and `save_file` for each shard. Chosen because the task is pure checkpoint
    editing; nothing needs a model to be instantiated.
  - `torch` 2.14.0 — the float32 matmul, the scaling and the dtype cast.
  - I deliberately did **not** use `peft.merge_and_unload`: that route requires
    materialising a `GPTNeoXForCausalLM` through `transformers` and then
    `save_pretrained`, which is far more memory and time for an operation that
    is 16 rank-16 matmuls, and it would hand control of the sharding and of the
    exact key set to the library instead of to the required checks. A ~90-line
    script does it directly and lets every required check fail loudly before a
    byte is written.
- **Approximate time spent:** about 5 minutes.
