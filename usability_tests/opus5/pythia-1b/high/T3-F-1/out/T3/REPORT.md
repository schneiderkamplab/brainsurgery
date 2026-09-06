# T3 self-report (condition F, Pythia-1B)

- **Final artifact path:** `out/T3/solution.py` (entry point `out/T3/run.sh`;
  output checkpoint `out/T3/model-0000{1..9}-of-00009.safetensors` +
  `out/T3/model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single
  execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The task quotes the embedding tensors as "206 MB each", which is their
    float16 size in the input; in the float32 output they are 412 MB each, so
    they exceed the 256 MiB shard budget by even more and go into their own
    shards either way.
  - `gpt_neox.layers.<i>.attention.masked_bias` is a float16 scalar (shape
    `[]`) and `...attention.bias` is uint8, so a dtype-based filter would not
    separate buffers from parameters; only the names do.
  - The obvious regex `.*weight` would have hit the embeddings, the layer
    norms and `final_layer_norm`, so I did not use a regex at all.
  - The shard budget applies to tensor payload, not file size; safetensors
    headers add a few kB per shard, so I measured `numel * element_size` sums
    rather than `os.path.getsize`.
- **Anything in the task text or documentation that was unclear:**
  - The shard file naming convention is not specified. I used HuggingFace's
    canonical `model-{n:05d}-of-{total:05d}.safetensors`, which is what the
    index-file convention implies and what serving stacks expect.
  - It is not stated whether `config.json`/tokenizer files should be copied
    into `out/T3/`. The required result lists only shards plus the index, so I
    wrote only those.
- **Tools used (condition F):**
  - `torch` 2.14.0 — dtype casting (`.to(torch.bfloat16)` / `.to(torch.float32)`)
    and the bit-exactness comparison in the verification pass.
  - `safetensors` 0.5.3 — `safe_open` to stream the input key by key (so the
    float16 source and the converted copies are never both fully resident) and
    `save_file` to write each shard.
  - `huggingface_hub` 1.16.1 — `split_torch_state_dict_into_shards` with
    `max_shard_size=268435456`. I used the library splitter rather than my own
    greedy loop because it is the reference implementation of the exact
    sharding rule the task states, including "a tensor larger than the budget
    gets its own shard", and it produces the canonical file naming and index
    layout.
  - **Why not the suggested route:** the hint points at `transformers`
    dtype export (`save_pretrained(dtype=...)`), but that applies *one* dtype
    to the whole model. This task needs per-tensor dtypes, and round-tripping
    through `GPTNeoXForCausalLM` would also re-materialise the causal-mask and
    `inv_freq` buffers that must be deleted, and could re-tie or re-order keys.
    A direct state-dict script avoids all of that. `mergekit`, `peft` and
    `torch-state-bridge` solve merging / adapters / key renaming, none of
    which this task involves — key names must not change here.
- **How the required checks are enforced:** `required_checks()` runs on the
  in-memory state dict *before* any file is written and raises `SystemExit` on
  failure — exactly 64 bfloat16 tensors and they are exactly the 64 enumerated
  projection matrices, `gpt_neox.layers.0.attention.query_key_value.weight` is
  bfloat16, `gpt_neox.embed_in.weight` is float32, and the tensor count is 196.
  Pre-flight assertions also confirm all 64 projection and 48 buffer names
  exist in the input with the expected shapes, so a typo fails as a missing
  key instead of silently under-matching. After writing, `verify_on_disk()`
  reloads every shard and re-runs the same checks plus the per-shard byte
  budget, weight_map/shard agreement, no duplicate keys across shards, and
  bit-identical values.
- **Approximate time spent, if you can tell:** roughly 5 minutes.
