# T3 — Participant self-report (condition P)

- **Final artifact path:** `out/T3/solution.py` (output checkpoint in `out/T3/`:
  9 shards `model-0000N-of-00009.safetensors` + `model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none — first execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - `gpt_neox.layers.<i>.attention.bias` is the causal-mask buffer while
    `attention.dense.bias` / `attention.query_key_value.bias` are real parameters, so
    the delete pattern has to be anchored with `$` or it eats the projection biases.
  - A `.*weight` pattern would sweep in `embed_in`/`embed_out` and every layer norm;
    I listed the four projection names explicitly and anchored both ends instead.
  - Shard packing must be planned on **output** byte sizes, not input ones: the
    embeddings are 206 MB as float16 in the input but 412 MB once upcast to float32,
    which is what pushes them over the 256 MiB budget into shards of their own.
  - TASK.md quotes the embeddings as "206 MB each" (the float16 input size); the
    float32 output tensors are 412,090,368 bytes each. Same outcome — both are over
    the budget and get a solo shard — but the number in the task text is the input's.
  - safetensors' `keys()` is sorted, so `embed_out.weight` sorts before
    `gpt_neox.embed_in.weight` and lands in shard 1; layer indices sort as strings
    (0, 1, 10, 11, ... 15, 2, 3, ...). Deterministic, just not numeric order.
  - Wrote each shard by reopening the input with `safe_open` and pulling only that
    shard's tensors, to avoid holding the whole ~2.4 GB float32 state dict in RAM.
- **Anything in the task text or documentation that was unclear:**
  - The shard **naming scheme** is not specified. I used the HuggingFace convention
    `model-{k:05d}-of-{n:05d}.safetensors`; if the hidden reference names shards
    differently, the file names will not match even though the layout does.
  - The **packing order and algorithm** are not specified either — "at most 256 MiB
    per shard" admits many valid packings. I used the obvious one: greedy first-fit
    in sorted-key order, flushing the current shard when the next tensor would
    exceed the budget (this is what `split_torch_state_dict_into_shards` does).
  - Whether the index needs a `metadata.total_size` field is not stated; I included
    it (2,436,513,792 bytes) since serving stacks expect it.
  - Whether `config.json` / tokenizer files should be copied into `out/T3/` is not
    stated. I wrote only shards + index, per "Output:" in the task.
- **Tools used (condition F):** n/a — condition P (torch 2.14.0, safetensors 0.5.3).
- **Approximate time spent, if you can tell:** ~5 minutes, one execution (~40 s of it
  reading, casting and writing 2.4 GB).
