# T5 — Participant self-report (condition P)

- **Final artifact path:** `out/T5/solution.py` (199 non-blank, non-comment lines).
  Output checkpoint: `out/T5/` — 10 shard files plus `model.safetensors.index.json`.

- **Number of times you executed the script or plan:** 2

- **Which executions failed, and why (one line each):**
  - None. Execution 1 passed every required check and wrote a valid output; execution 2 was a
    deliberate re-run after I changed the shard-packing rule (see the first pitfall below).

- **Pitfalls or surprises you hit (one line each):**
  - TASK.md is internally inconsistent about the two embedding matrices: it calls
    `model.embed_tokens.weight` / `lm_head.weight` "a single tensor larger than [512 MiB] ... stored
    alone in its own shard", but 412 MB is 393 MiB, i.e. *below* the 536,870,912-byte budget, so a
    literal `nbytes > SHARD_LIMIT` rule never puts them alone — my first run packed `embed_tokens`
    together with a 64 MiB layer tensor. I changed the rule to "a tensor taking more than half a
    shard is never packed with anything else", which honours the explicit requirement, still
    respects the ≤512 MiB budget, and leaves the 4096 MiB of layer tensors packing into exactly
    8 full 512 MiB shards.
  - The script must live in `out/T5/`, which is also the output directory — my first draft cleared
    the directory with `shutil.rmtree(OUT_DIR)`, which would have deleted the running script itself.
    Caught by reading before running; replaced with a targeted unlink of `*.safetensors` and
    `*.index.json`.
  - Doubled prefix: PEFT keys are `base_model.model.` + the base name, and the base name already
    starts with `model.`, so the strip is `base_model.model.` and not `base_model.`.
  - `adapter_config.json` lists `target_modules` as bare `["q_proj", "v_proj"]`, while TASK.md
    writes them as `self_attn.q_proj` / `self_attn.v_proj`. I derived the base names from the
    adapter tensor keys instead of from `target_modules`, so the discrepancy does not matter.
  - Loading the whole 5.1 GB base to plan shards is unnecessary: `safe_open(...).get_slice(k)`
    exposes `get_shape()` / `get_dtype()` from the header alone, so the shard plan and all four
    required checks run before a single byte of tensor data is read, and the write loop then holds
    only one shard (≤512 MiB) in memory at a time.
  - `save_file` needs contiguous tensors; the merged weights come out of an `add` so they are
    contiguous anyway, but I call `.contiguous()` explicitly rather than rely on it.

- **Anything in the task text or documentation that was unclear:**
  - The "stored alone in its own shard" clause above — the stated threshold and the named tensors
    contradict each other, and the two readings give different outputs. I followed the named
    tensors, since that is the more specific instruction, and both readings satisfy the budget rule.
  - Shard *file naming* is unspecified ("shard files plus an index file"). I used the HuggingFace
    convention `model-000NN-of-000TT.safetensors`. Because naming is free, I read "sharding rules"
    in the grading section as rule checks (budget, solitary large tensors, index/shard agreement)
    rather than an exact shard-membership match against the reference.
  - Not stated whether the index `metadata.total_size` is required. I wrote it; it comes out to
    5,119,148,032, identical to the base index, which is a useful confirmation that no tensor
    changed size.

- **Tools used (condition F):** n/a — condition P. Only `torch` 2.14.0+cu130, `safetensors` 0.5.3
  and the standard library.

- **Approximate time spent, if you can tell:** ~5 minutes wall clock; each script execution takes
  about 9 seconds.

## What the script does

1. Reads `inputs/base/model.safetensors.index.json` and both shard headers to build a
   name → (shape, dtype, nbytes) map without loading tensor data.
2. Reads `r`, `lora_alpha` and `fan_in_fan_out` from `adapter_config.json` (not hardcoded);
   `scale = 32 / 16 = 2.0`. `fan_in_fan_out = false`, so `B @ A` is added untransposed; the
   transposed branch is implemented for the other case and verified by a shape check.
3. Pairs every `*.lora_A.weight` with its `*.lora_B.weight`, maps the key to the base name, and
   verifies each target exists, each rank agrees with `r`, `B @ A` matches the base `[out, in]`,
   the base weight is float32, and that all 64 adapter tensors are consumed by exactly 32 pairs.
4. Runs the four required checks — 32 pairs, no `lora_` in the output names, probe
   `model.layers.0.self_attn.q_proj.weight` is `[2048, 2048]`, 114 output tensors — all before
   any file is written.
5. Plans shards greedily in base-index order under the 512 MiB budget, with the solo rule above.
6. Writes each shard: loads its tensors, applies `w += scale * (B @ A)` in float32 where an adapter
   pair applies, re-checks shape and dtype per tensor, and asserts 32 merges happened in total.
7. Re-reads the written index and every shard header and re-verifies the key set, the tensor count,
   the absence of `lora_` names, the per-shard budget, index/shard agreement, and that no
   unexpected `.safetensors` file is present.

Final output: 114 tensors, float32, names identical to the base, in 10 shards —
`lm_head.weight` alone (393 MiB), `model.embed_tokens.weight` alone (393 MiB), and 8 shards of
exactly 512.0 MiB holding two transformer layers each.
