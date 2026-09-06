# T3 participant self-report

- **Final artifact path:** `out/T3/solution.py` (output checkpoint in `out/T3/`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the first execution passed all
  checks and wrote the checkpoint.
- **Pitfalls or surprises you hit:**
  - `h.<i>.attn.bias` (the causal-mask buffer) and `h.<i>.attn.c_attn.bias` /
    `c_proj.bias` (real parameters) differ only by one path segment, so the
    drop pattern has to be anchored at both ends (`^h\.\d+\.attn\.bias$`);
    a substring match on `attn.bias` would have deleted parameters.
  - A `.*weight` style pattern would have swept up `wte.weight`,
    `wpe.weight` and every layer-norm weight, so I enumerated the four
    projection suffixes explicitly and asserted the cast set equals the
    expected 48 names rather than just counting them.
  - `wte.weight` is 154 MB, well over the 64 MiB shard budget; a greedy
    packer has to tolerate an oversized tensor instead of erroring. The HF
    splitter handles it by giving that tensor its own shard.
  - GPT-2 Conv1D matrices are stored `[in, out]` (e.g. `c_fc.weight` is
    `[768, 3072]`), the opposite of `nn.Linear`; irrelevant for a pure dtype
    cast but worth confirming before touching shapes.
- **Anything in the task text or documentation that was unclear:** the shard
  file naming and the packing order are not specified, only the 64 MiB
  budget and the "oversized tensor alone" rule. I used the canonical
  HuggingFace layout (`model-0000k-of-0000N.safetensors`, greedy packing in
  state-dict order) on the assumption the reference does the same.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — load/save; the input and required output format.
  - `torch` 2.14.0 — `tensor.to(torch.bfloat16)` for the exact
    round-to-nearest-even cast the task specifies, and `torch.equal` for the
    bit-exactness checks.
  - `huggingface_hub` (pinned) — `split_torch_state_dict_into_shards`, the
    same splitter `save_pretrained` uses, so the shard layout and index file
    match what serving stacks expect without me reimplementing the packer.
  - I did *not* use `transformers` `save_pretrained`: it applies a single
    dtype to the whole model, which cannot express a per-tensor mix of
    bfloat16 and float32, and loading through `GPT2LMHeadModel` would have
    renamed keys (`transformer.` prefix, `lm_head`) and re-materialised the
    mask buffers. `mergekit` and `peft` are for merging/adapters and have no
    role in a dtype-and-sharding export.
- **Approximate time spent:** ~5 minutes.

## What the script enforces before writing

Required checks (each raises `SystemExit` and aborts before any file is
written): exactly 48 bfloat16 tensors; `h.0.attn.c_attn.weight` is bfloat16;
`wte.weight` is float32; exactly 148 output tensors. Additional guards:
exactly 12 buffers dropped, the cast set equals the intended 48 names,
`wpe.weight`/`ln_f.*` are float32, and all non-cast tensors compare
bit-equal to the input. After writing, the checkpoint is re-read from disk
and every tensor is compared against the expected dtype and bits, the index
`weight_map` is checked against the tensors actually present, and no
multi-tensor shard is allowed to exceed 64 MiB of tensor data.
