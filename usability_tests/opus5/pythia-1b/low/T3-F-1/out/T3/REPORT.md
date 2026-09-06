# T3 self-report

- **Final artifact path:** `out/T3/solution.py` (invoked by `out/T3/run.sh`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the first execution succeeded.
- **Pitfalls or surprises you hit:**
  - The obvious trap is an over-broad dtype pattern: `.*weight` would also hit
    `embed_in`/`embed_out` and every layer-norm weight, so I anchored the
    bfloat16 regex to the four named projection matrices and asserted the count
    is exactly 64.
  - `attention.bias` and `attention.masked_bias` look like ordinary biases but
    are buffers to be dropped, while the *projection* biases must be kept and
    upcast — the drop regex is therefore name-exact, not suffix-based.
  - Sharding budget: `embed_in.weight` and `embed_out.weight` are 206 MB each in
    float32 and exceed the 256 MiB shard budget only in combination, so a
    single-tensor shard has to be permitted; I let the standard HF splitter
    handle it and asserted the per-shard byte total explicitly.
  - `.contiguous()` on every tensor to avoid safetensors rejecting shared or
    non-contiguous storage.
- **Anything unclear:** the shard *file naming* is not specified in TASK.md
  (only the index and the byte budget), so I used the HuggingFace convention
  `model-0000k-of-0000N.safetensors`, which is what the index format implies.
  Whether shards must be packed greedily in original key order was likewise
  implied rather than stated; greedy in key order is the standard behaviour.
- **Tools used (condition F):**
  - `torch` 2.14.0 — dtype casts (`.to(torch.bfloat16)` / `.to(torch.float32)`).
  - `safetensors` 0.5.3 — `load_file` / `save_file` for the shards.
  - `huggingface_hub` (pinned) — `split_torch_state_dict_into_shards`, the same
    splitter `save_pretrained` uses, so the shard layout and index match what a
    serving stack expects instead of my own hand-rolled packing.
  - I deliberately did **not** go through `transformers` `save_pretrained`: it
    exports one dtype for the whole model, so per-tensor mixed precision would
    have required post-hoc rewriting anyway, and loading into a `GPTNeoXModel`
    would re-materialise the very buffers the task asks me to drop.
  - `mergekit` / `peft` / `torch-state-bridge` are not relevant here (no merge,
    no adapter, no key renaming — names must not change).
- **Approximate time spent:** ~4 minutes.
