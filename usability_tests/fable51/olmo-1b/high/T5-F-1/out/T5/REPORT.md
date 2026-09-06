# T5 (OLMo-1B-0724-hf), condition F: participant self-report

- Final artifact path: `out/T5/solution.py` (wrapper: `out/T5/run.sh`). Output
  checkpoint: `out/T5/model-0000{1..10}-of-00010.safetensors` plus
  `out/T5/model.safetensors.index.json`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; execution 1 passed
  all checks and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - The adapter config's `target_modules` are `q_proj`/`v_proj` (module suffixes),
    not the `self_attn.q_proj` strings TASK.md quotes; I mapped names by
    stripping the `base_model.model.` prefix and the `.lora_{A,B}.weight`
    suffix instead of relying on `target_modules`.
  - TASK.md says `embed_tokens` and `lm_head` are "larger than" the shard
    budget, but at 412,090,368 bytes they are below 512 MiB. A plain greedy
    packer would co-locate them with layer tensors, so I added an explicit
    rule: tensors above half the budget get their own shard, as the text asks.
  - Each layer is exactly 256 MiB of float32, so two layers fill a shard to
    exactly 536,870,912 bytes; the check is "at most", so this is valid.
  - The base index is alphabetical (`layers.10` before `layers.2`); I used a
    natural sort so shards hold consecutive layers.
- Anything in the task text or documentation that was unclear:
  - Whether the grader expects a specific shard partition or only the size and
    index constraints ("sharding rules"). I followed the text literally: the
    two big tensors alone, everything else greedily packed at <= 512 MiB.
  - The `target_modules` wording mismatch noted above.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: the float32 `B @ A` matmul and the add.
  - `safetensors` 0.5.3: `safe_open` to stream tensors from the base shards and
    the adapter (headers first for planning, data per output shard), and
    `save_file` for each output shard.
  - `json` / `re` (stdlib): adapter config, base index, name mapping, output index.
  - `peft` 0.20.0 was used only as a read-only cross-check after the run
    (a single `nn.Linear` wrapped with `LoraConfig`, `merge_and_unload`);
    its result is bit-identical to the merged tensor in the output. I did not
    use `peft.merge_and_unload` for the solution itself because TASK.md asks
    for the merge directly on the checkpoint files, and instantiating the
    5 GB model through transformers would also have handed shard packing and
    tie-weight handling to `save_pretrained`, whose `max_shard_size` is
    decimal MB and would not have honoured the "big tensors alone" rule.
  - `mergekit`, `torch-state-bridge`, `transformers` not used.
- Approximate time spent, if you can tell: about 5 minutes wall clock,
  including inspection of the inputs; the script itself runs in ~10 s.

## What the script enforces before writing

- adapter type is LoRA, `bias == "none"`, scale = alpha / r (or alpha / sqrt(r) if `use_rslora`);
- every adapter tensor parses as `base_model.model.<base>.lora_{A,B}.weight`,
  every pair is complete, targets an existing base tensor, and A/B shapes fit
  the base weight under the recorded `fan_in_fan_out`;
- exactly 32 pairs, and no leftover adapter tensors;
- no output name contains `lora_`; `model.layers.0.self_attn.q_proj.weight` is `[2048, 2048]`;
- the output has exactly 114 names, identical to the base;
- every planned shard is <= 512 MiB of tensor data; destination has no
  pre-existing checkpoint files.

After writing, the script reopens the output and checks index/shard
consistency, per-shard data size, shapes and dtypes, bit-exactness of the 82
unchanged tensors against the base, and a relative error <= 1e-6 of the 32
merged tensors against a recomputation.
