# T3 participant self-report

- **Final artifact path:** `out/T3/solution.py` (output checkpoint in `out/T3/`:
  4 shards + `model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the first execution
  passed all checks and wrote the output.
- **Pitfalls or surprises you hit (one line each):**
  - `h.<i>.attn.bias` is the causal-mask buffer, not a projection bias, so any
    name rule keyed on "bias" or on `.*weight` would have hit the wrong tensors;
    I avoided regex entirely and built the 48 projection names and the 12 buffer
    names explicitly from the layer index.
  - `mlp.c_proj.weight` and `attn.c_proj.weight` share a suffix, so a suffix-only
    match would be ambiguous about which one it means; the explicit set makes it moot.
  - The tensor order inside the input safetensors file is arbitrary (hash order,
    e.g. `h.3.ln_2.bias` first, `wte.weight` in the middle), so greedy packing in
    file order would be non-reproducible; I shard over `sorted(keys)` instead.
  - `wte.weight` is 154 MB, above the 64 MiB shard budget, so the packer needs an
    explicit "oversized tensor goes alone" branch rather than only a running-sum
    check, otherwise it either overflows a shard or drags neighbours along.
  - safetensors rejects non-contiguous tensors, so I called `.contiguous()` after
    the dtype cast.
- **Anything in the task text or documentation that was unclear:**
  - The shard file naming scheme is not specified; I used the HuggingFace
    convention `model-000NN-of-000NN.safetensors`.
  - "at most 64 MiB of tensor data" is stated per shard but a single oversized
    tensor is explicitly exempt; I read that as: the budget applies to multi-tensor
    shards, and a lone oversized tensor is allowed to exceed it.
  - Whether the index should carry a `metadata.total_size` field is not stated;
    I included it, as `save_pretrained` does.
  - Whether the tokenizer/config files from `inputs/base/` should be copied into
    `out/T3/` is not stated; I wrote only the weights and the index.
- **Tools used (condition F):** n/a (condition P): torch 2.14.0, safetensors 0.5.3.
- **Approximate time spent, if you can tell:** ~5 minutes.
