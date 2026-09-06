# T3 self-report

- **Final artifact path:** `out/T3/solution.py` (output in `out/T3/`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - `h.<i>.attn.bias` is a causal-mask buffer, so a name-based "bias" rule would
    have kept it; I matched it explicitly by anchored regex and dropped it.
  - A `.*weight` pattern would hit `wte`/`wpe`/layer norms, so the projection
    regex is fully anchored to the four `attn.c_attn|attn.c_proj|mlp.c_fc|mlp.c_proj`
    weights per layer.
  - `wte.weight` (154 MB) exceeds the 64 MiB shard budget on its own; the
    splitter puts it alone in a shard, and my post-write check exempts only
    single-tensor shards.
- **Anything unclear:** the shard *file naming* is not specified; I used the
  HuggingFace convention `model-0000k-of-0000n.safetensors` produced by
  `split_torch_state_dict_into_shards`, which is what serving stacks expect.
- **Tools used (condition F):**
  - `torch` 2.14.0 — dtype casts (`.to(torch.bfloat16)`, RNE as specified).
  - `safetensors` 0.5.3 — `load_file` / `save_file` for exact tensor I/O.
  - `huggingface_hub` 1.16.1 — `split_torch_state_dict_into_shards` for the
    standard shard layout and `weight_map`, rather than hand-rolling packing.
  - I deliberately did *not* use `transformers.save_pretrained`: it applies one
    global dtype, which cannot express a per-tensor mixed-precision split, and
    it would re-materialize the mask buffers via the model class.
  - `mergekit` / `peft` are not applicable here (no merging, no adapters).
- **Approximate time spent:** ~5 minutes.
