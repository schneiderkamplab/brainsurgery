# T3 participant self-report

- **Final artifact path:** `out/T3/solution.py` (output checkpoint in `out/T3/`:
  9 shards `model-0000N-of-00009.safetensors` + `model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - The obvious `.*weight` regex would hit `embed_in`/`embed_out`, all layer
    norms and `attention.bias`, so I built the 64 projection names and the 48
    buffer names literally from the layer index instead of pattern matching.
  - `gpt_neox.layers.<i>.attention.bias` is a uint8 causal-mask buffer whose
    name looks exactly like a parameter bias; deleting by suffix `bias` alone
    would also delete the real projection biases.
  - Both embedding tensors (412 MB each in fp32) exceed the 256 MiB shard
    budget on their own, so the shard-budget check must exempt single-tensor
    shards; I verified afterwards that each is alone in its shard.
  - `huggingface_hub.split_torch_state_dict_into_shards` defaults to the
    `model{suffix}.safetensors` pattern only if you pass it explicitly when
    calling it outside `save_pretrained`; I set it explicitly.
- **Anything in the task text or documentation that was unclear:** the exact
  shard *naming* convention and tensor ordering are not specified, only the
  size rule and the index; I assumed the standard HuggingFace layout
  (`model-0000N-of-0000M.safetensors`, insertion-order greedy packing).
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — `load_file` / `save_file`; direct, dtype-exact
    checkpoint I/O without instantiating a model.
  - `torch` 2.14.0 — `.to(torch.bfloat16)` / `.to(torch.float32)` casts, which
    is exactly the rounding the task specifies.
  - `huggingface_hub` 1.16.1 — `split_torch_state_dict_into_shards`, the same
    function `transformers.save_pretrained` uses, so the shard packing and the
    index `weight_map`/`total_size` match what serving stacks expect.
  - Considered and rejected: `transformers` `save_pretrained(dtype=...)`,
    because it applies one dtype to the whole model and re-creates buffers such
    as `attention.bias`, which this task requires to be absent. `mergekit`
    targets merges, not per-tensor mixed precision. A plain script over
    safetensors + the hub splitter was smaller and let the required checks run
    before anything was written.
- **Approximate time spent:** ~5 minutes.

## Checks enforced by `solution.py` (all before writing)

Each raises `SystemExit` (non-zero exit) if violated: exactly 64 bfloat16
tensors; `gpt_neox.layers.0.attention.query_key_value.weight` is bfloat16;
`gpt_neox.embed_in.weight` is float32; exactly 196 output tensors. Plus:
every non-projection tensor is float32, all expected input names present,
names unchanged, no multi-tensor shard over 268,435,456 bytes of tensor data,
and `weight_map` covers every tensor.
