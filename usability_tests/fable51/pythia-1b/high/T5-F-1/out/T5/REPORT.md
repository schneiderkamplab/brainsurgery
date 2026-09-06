# T5 (Pythia-1B, condition F) — participant self-report

- Final artifact path: `out/T5/solution.py` (output checkpoint: `out/T5/model-0000{1..6}-of-00006.safetensors` + `out/T5/model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - None failed. Execution 1 produced a valid 4-shard output but packed both
    embeddings into one shard; execution 2 (final) stores each embedding alone,
    as TASK.md states, giving 6 shards.
- Pitfalls or surprises you hit (one line each):
  - TASK.md says the two embedding tensors (206 MB each) are "larger than"
    the 512 MiB budget and stored alone; they are not larger, so I followed the
    stated outcome (standalone shards) rather than the size rule alone.
  - Base checkpoint includes non-float16 buffers (`attention.bias` bool,
    `masked_bias`), so shard byte accounting must use each tensor's element size.
  - PEFT adapter names carry the `base_model.model.` prefix and a
    `.lora_{A,B}.weight` suffix; mapped via one regex to `<base>.weight`.
- Anything in the task text or documentation that was unclear:
  - The sharding paragraph's parenthetical contradicts its own rule (206 MB is
    under 512 MiB); which one the hidden reference follows is not stated.
  - Shard file naming and index `metadata` content are unspecified; I used the
    HuggingFace convention `model-XXXXX-of-XXXXX.safetensors` and
    `{"metadata": {"total_size": ...}, "weight_map": {...}}`.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: float32 `B @ A` and add, cast back to float16.
  - `safetensors` 0.5.3: `safe_open` to read base and adapter, `save_file` per shard.
  - Plain Python script instead of `peft.merge_and_unload` /
    `transformers.save_pretrained`: no need to instantiate the model, full
    control over the shard budget rule (per-tensor bytes, standalone
    embeddings), and it keeps unchanged tensors bit-exact without a
    dtype round trip through a model class.
  - Checks enforced in the script before writing: 16 adapter pairs found and
    merged, no `lora_` key, probe shape `[6144, 2048]` and float16, exactly
    244 tensors, same key set as base, per-shard budget. After writing it
    re-reads the shards and verifies key set, no duplicates across shards,
    bit-exact equality for the 228 untouched tensors and relative Frobenius
    error <= 1e-3 for the 16 merged weights.
- Approximate time spent, if you can tell: ~3 minutes wall clock.
