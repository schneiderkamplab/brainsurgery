# T5 self-report

- **Final artifact path:** `out/T5/solution.py` (output checkpoint in `out/T5/`,
  5 shards + `model.safetensors.index.json`).
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none.
- **Pitfalls or surprises you hit:**
  - Conv1D layout: base `h.<i>.attn.c_attn.weight` is `[in, out] = [768, 2304]`
    while `B @ A` is `[out, in]`, so the delta needs a transpose; this is what
    `fan_in_fan_out: true` in the adapter config signals.
  - Adapter names carry a `base_model.model.` prefix (and PEFT sometimes a
    `.default` sub-module), so the mapping to base names needs stripping, not a
    plain lookup. I matched with a regex tolerant of both.
  - Sharding: `wte.weight` alone is 154 MB, above the 100 MiB budget, so the
    greedy packer has to seal an oversized tensor into its own shard rather
    than refusing it. It happens to be last in the base key order here.
  - GPT-2 checkpoints share storage (`wte`/`lm_head` tying, plus the 12
    `attn.bias` mask buffers), so I `.contiguous().clone()` every tensor before
    `save_file`; safetensors rejects tensors that share memory.
- **Anything unclear:** the task fixes the 100 MiB budget but not the packing
  order or shard naming; I used base checkpoint order, greedy fill, and the
  HF `model-{i:05d}-of-{n:05d}.safetensors` convention. The 100 MiB bound is
  stated as tensor data excluding headers, which is what I measured.
- **Tools used (condition F):** n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- **Approximate time spent:** ~5 minutes.
