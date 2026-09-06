# T5 run record — participant self-report

- **Final artifact path:** `out/T5/solution.py` (run as
  `.venv/bin/python out/T5/solution.py` from the sandbox root); output in
  `out/T5/model-0000{1..10}-of-00010.safetensors` + `model.safetensors.index.json`.
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the first execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - PEFT name prefix: adapter keys carry `base_model.model.` on top of the
    already-`model.`-prefixed base names, so the base target is
    `key[len("base_model.model."):]` with `.lora_{A,B}.weight` -> `.weight`,
    not a naive strip of one `model.` level.
  - `adapter_config.json` lists `target_modules` as `["q_proj", "v_proj"]`,
    not the dotted `["self_attn.q_proj", ...]` that TASK.md quotes; I derived
    targets from the adapter tensor names instead of from the config so the
    discrepancy could not silently drop pairs.
  - Scale is `lora_alpha / r = 2`, not `1`, and it multiplies `B @ A`
    (`[out, r] @ [r, in]`); with `fan_in_fan_out = false` no transpose applies,
    but I still branch on the config flag rather than hard-coding the layout.
  - The task text says `model.embed_tokens.weight` and `lm_head.weight`
    (412 MB each) are "larger than" the 512 MiB budget and must sit alone;
    they are not (393 MiB each), so that clause is vacuous here. I applied the
    stated hard rule — every shard's tensor data <= 536,870,912 bytes, and any
    single oversized tensor alone — via the canonical greedy splitter. As it
    happens `lm_head.weight` does end up alone in shard 1.
  - The base checkpoint's own sharding (2 files, 5.1 GB total) is unrelated to
    the required output sharding, so the output had to be re-split from scratch.
- **Anything in the task text or documentation that was unclear:**
  - The 512 MiB clause about oversized tensors (see above) is inconsistent with
    the stated 412 MB sizes; it was not clear whether an exact shard-to-tensor
    assignment is graded or only the budget property.
  - Whether `out/T5/` should also carry `config.json` / tokenizer files. The
    "Required result" lists only shard files plus the index, so I wrote only those.
- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — direct `load_file`/`save_file` on the checkpoint
    files; the whole point of the task is to merge without instantiating a model.
  - `torch` 2.14.0 — float32 matmul for `scale * B @ A` and the bit-exact copy
    of untouched tensors.
  - `huggingface_hub` 0.36.x `split_torch_state_dict_into_shards` — the same
    splitter `transformers.save_pretrained` uses, so the 512 MiB budget and the
    "oversized tensor alone" rule come from the canonical implementation rather
    than from a hand-rolled packer. I still re-verify each shard's byte total
    before writing.
  - **Considered and rejected:** `peft.merge_and_unload` (the route F-allowed.md
    suggests) would require instantiating a full OLMo-1B `AutoModelForCausalLM`
    and a `PeftModel`, i.e. ~5 GB of allocation plus a dtype/`device_map` round
    trip, to do 32 rank-16 matmuls; and `save_pretrained` gives no direct hook to
    assert the required checks before writing. `mergekit` targets model-level
    merges, not folding an adapter into named tensors. A ~150-line script over
    safetensors is smaller, has no model-class dependency, and lets every
    required check fail loudly *before* any file is written.
- **Approximate time spent, if you can tell:** ~5 minutes.

## Required checks, as enforced

All raise `CheckFailed` and exit non-zero *before* any output file is created:

1. exactly 32 complete `(lora_A, lora_B)` pairs found, and `merged == 32`;
2. no output tensor name contains `lora_`;
3. `model.layers.0.self_attn.q_proj.weight` present with shape `(2048, 2048)`;
4. the output holds exactly 114 tensors.

Additional guards: every adapter key carries the PEFT prefix and is one of
`lora_A`/`lora_B`; no pair is half-present; rank matches `r`; every target
exists in the base; delta shape matches the base weight; all outputs are
float32; each written shard is within the 512 MiB budget (or is a lone tensor);
the index `weight_map` covers exactly the output key set.

## Independent verification of the produced output

Re-read `out/T5` from disk and compared against `inputs/`:
114 tensors, key set identical to the base, all float32, no `lora_` names,
`q_proj` layer 0 shape `(2048, 2048)`, all 10 shards <= 536,870,912 bytes of
tensor data, `weight_map` complete; the 82 untouched tensors bit-exact against
the base and the 32 merged weights at relative Frobenius error 0.0 against an
independently recomputed `W + 2 * B @ A`.
