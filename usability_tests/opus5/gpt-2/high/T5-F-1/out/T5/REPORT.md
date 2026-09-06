# T5 — LoRA adapter merge with sharded export (GPT-2 124M) — participant self-report

- **Final artifact path:** `out/T5/solution.py` (run as
  `.venv/bin/python out/T5/solution.py` from the sandbox root). Output:
  `out/T5/model-0000{1..5}-of-00005.safetensors` + `out/T5/model.safetensors.index.json`.

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why:** none — the first execution succeeded.

- **Pitfalls or surprises you hit:**
  - The obvious "supported" route (PEFT `merge_and_unload` + `save_pretrained`) is a trap
    for this task: it instantiates `GPT2LMHeadModel`, which renames every key with a
    `transformer.` prefix, adds a tied `lm_head.weight`, and drops the 12
    `h.<i>.attn.bias` causal-mask buffers as non-persistent. That alone breaks the
    "exactly 160 tensors, same names as the base" and bit-exactness requirements, so I
    did the merge directly on the tensors instead.
  - Conv1D layout: the base `h.<i>.attn.c_attn.weight` is `[in, out] = [768, 2304]` while
    `B @ A` is `[out, in] = [2304, 768]`. `fan_in_fan_out: true` in `adapter_config.json`
    is exactly the flag that says the product must be transposed before adding; I read the
    flag from the config rather than hardcoding the transpose.
  - `adapter_config.json` lists `target_modules: ["c_attn"]`, not `"attn.c_attn"` as the
    task text says. I match target modules by suffix so both spellings validate.
  - Adapter key prefix is `base_model.model.` (no `default.` sub-key in this dump), and
    the base names carry no `transformer.` prefix, so the mapping is a plain prefix strip
    plus `.lora_A.weight` -> `.weight`.
  - Shard budget vs. a single oversized tensor: `wte.weight` is 154,389,504 bytes, over the
    100 MiB budget, so the size check has to special-case a shard that holds exactly one
    tensor. It landed alone in `model-00004-of-00005.safetensors` as required.
  - `(B @ A).T` is non-contiguous; safetensors rejects non-contiguous tensors, so the merged
    result is `.contiguous()` before saving.

- **Anything in the task text or documentation that was unclear:**
  - The task says `target_modules = ["attn.c_attn"]` but the actual file says `["c_attn"]`.
  - The shard-file naming scheme is not specified (only the index file name is). I used the
    HuggingFace convention `model-<k>-of-<n>.safetensors` produced by
    `split_torch_state_dict_into_shards`.
  - "at most 100 MiB of tensor data, not counting file headers" is stated clearly, and the
    exception for a single oversized tensor is stated — but it is left implicit whether the
    packing must match HF's greedy algorithm exactly. I used HF's own splitter to be safe.

- **Tools used (condition F):**
  - `torch` 2.14.0+cu130 — tensor math for the low-rank product, transpose and add, in float32.
  - `safetensors` 0.5.3 — `load_file` / `save_file`; direct file-level access is what keeps the
    148 untouched tensors bit-exact and preserves the exact 160-key set.
  - `huggingface_hub` (pinned) — `split_torch_state_dict_into_shards` for the canonical
    sharding + `weight_map`, so the layout matches what a HF-produced reference would look
    like instead of a hand-rolled packing.
  - **Considered and rejected:** `peft` 0.20.0 `merge_and_unload` and `transformers` 5.12.1
    `save_pretrained` — correct for the *math*, wrong for the *checkpoint contract* (key
    renaming, tied `lm_head`, dropped mask buffers), as described under Pitfalls.
    `mergekit` 0.1.4 has no LoRA-fold-into-base path that preserves a raw non-prefixed
    GPT-2 key set. `torch-state-bridge` only rewrites keys, which is not the hard part here.

- **How the required checks are enforced:** all four are hard assertions in `solution.py`
  that raise `CheckFailed` *before* anything is written — 12 adapter pairs found and merged,
  no `lora_` in the output key set, `h.0.attn.c_attn.weight` still `[768, 2304]`, exactly 160
  tensors. On top of that the script asserts the key set / shapes / dtypes equal the base's,
  that each shard is within the 100 MiB budget (or is a lone oversized tensor), and then
  re-reads every shard from disk to confirm the `weight_map` is consistent, the 148 unchanged
  tensors are bit-identical to the base, and each of the 12 merged weights is within 1e-6
  relative Frobenius error of an independently recomputed reference.

- **Approximate time spent:** ~5 minutes.
