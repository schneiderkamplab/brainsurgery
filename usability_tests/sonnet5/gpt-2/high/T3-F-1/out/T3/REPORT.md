# T3 participant self-report

- **Final artifact path:** `out/T3/solution.py` (produces `out/T3/model-*.safetensors`
  and `out/T3/model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - `h.<i>.attn.bias` is the only non-parameter buffer per layer and is easy
    to confuse with `h.<i>.attn.c_attn.bias` / `h.<i>.attn.c_proj.bias`
    (real parameter biases that must stay float32) — matched by exact key,
    not by a `*bias*` substring, to avoid dropping the wrong thing.
  - The projection weight list must be exact (`attn.c_attn.weight`,
    `attn.c_proj.weight`, `mlp.c_fc.weight`, `mlp.c_proj.weight`); a looser
    `.*weight` pattern would also catch `wte.weight`, `wpe.weight`, and the
    `ln_*.weight` norms, which must stay float32.
  - `wte.weight` (154 MB) exceeds the 64 MiB shard budget on its own, so the
    packer has to special-case any tensor larger than the budget into a
    lone shard rather than trying to split it or silently overflowing a
    shard.
- **Anything in the task text or documentation that was unclear:** No.
  Tensor names, shapes, sizes and shard budget were fully specified, so the
  same exact-key-set approach used for the cast/drop steps was used for
  shard planning too (deterministic greedy bin-packing in original key
  order).
- **Tools used (condition F): name, version, and why:**
  - `torch` 2.14.0 — `.to(torch.bfloat16)` for the round-to-nearest-even
    cast, `torch.dtype`/`element_size()` for shard-size accounting.
  - `safetensors` 0.5.3 — `safe_open`/`get_tensor` to read the input
    checkpoint tensor-by-tensor (no need to materialize a full state dict
    twice) and `safetensors.torch.save_file` to write each shard.
  - No merge/adapter tooling (`mergekit`, `peft`, `torch-state-bridge`) was
    needed: this task is a per-tensor dtype cast, a buffer drop, and a
    manual shard layout — a plain safetensors read/write script is the most
    direct route and gives full control over the exact key set and the
    sharding rule the task specifies (single-oversized-tensor-gets-its-own
    shard), which is easier to express directly than to configure through a
    merge-config-oriented tool.
- **Approximate time spent, if you can tell:** Under 5 minutes of wall time
  (single script, single execution, no debugging needed).
