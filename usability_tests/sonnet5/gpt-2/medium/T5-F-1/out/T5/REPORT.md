# Participant self-report — T5 (condition F)

- **Final artifact path:** `out/T5/solution.py` (invoked via `out/T5/run.sh`)
- **Number of times you executed the script or plan:** 2
- **Which executions failed, and why (one line each):**
  1. `AssertionError: adapter targets missing base tensor: h.0.attn.c_attn` —
     the regex that strips `.lora_A.weight` / `.lora_B.weight` off the
     adapter tensor name left the base name without its trailing `.weight`,
     so the lookup into the base state dict missed; fixed by appending
     `.weight` back onto the extracted base name.
- **Pitfalls or surprises you hit (one line each):**
  - PEFT adapter keys are prefixed `base_model.model.` and use
    `lora_A.weight` / `lora_B.weight` suffixes, not the base checkpoint's own
    naming — needed a regex to strip both ends correctly.
  - `fan_in_fan_out=true` in `adapter_config.json` is the signal that
    `B @ A` (nn.Linear convention, `[out, in]`) needs a transpose before
    adding to the Conv1D-layout `[in, out]` base weight; got the rel-error
    right (0.0 on the check I ran) only after transposing.
  - `wte.weight` (154 MB) exceeds the 100 MiB shard budget on its own, so it
    has to land alone in its own shard; `huggingface_hub`'s
    `split_torch_state_dict_into_shards` already does this by default when a
    single tensor exceeds `max_shard_size`, no special-casing needed.
- **Anything in the task text or documentation that was unclear:** No —
  the fan_in_fan_out / Conv1D vs Linear layout note and the oversized-tensor
  sharding rule were both spelled out explicitly in TASK.md.
- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — direct load/save of the base and adapter checkpoints
    without instantiating a model.
  - `torch` 2.14.0 — float32 matmul/transpose/add for the merge.
  - `huggingface_hub` (pinned, via `transformers`' dependency) —
    `split_torch_state_dict_into_shards` for the 100 MiB shard-splitting and
    index-file logic, the same helper `transformers` uses internally for
    sharded `save_pretrained`.
  - Did not use `peft.merge_and_unload`: it requires instantiating the full
    `GPT2LMHeadModel` and wrapping it in a `PeftModel` just to get back a
    plain state dict, which is more moving parts than doing the tensor
    arithmetic directly against the two safetensors files. A plain script on
    top of `safetensors`/`torch` was the more direct route and also made the
    required checks (pair count, no `lora_` leakage, shape, tensor count)
    trivial to assert before writing anything.
- **Approximate time spent, if you can tell:** ~10 minutes.
