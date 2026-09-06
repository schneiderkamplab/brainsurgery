# T5 self-report (condition B, Pythia-1B)

- **Final artifact path:** `out/T5/plan.yaml` (output checkpoint in `out/T5/`,
  4 shards + `model.safetensors.index.json`, 244 tensors).
- **Number of times you executed the script or plan:** 2.
- **Which executions failed, and why (one line each):**
  1. `matmul source_b missing: lora::base_model\.model\.gpt_neox.layers.0...lora_A\.weight`
     — I wrote `from_b` as a regex (escaped dots), but `from_b`/`to` are *rewrite*
     templates applied to the `from_a` captures, so the backslashes were taken
     literally. Fixed by writing `from_b` as a plain replacement string with `\1`.
  2. (second execution succeeded)
- **Pitfalls or surprises you hit (one line each):**
  - Ternary transforms (`matmul`, `add`) match on `from_a` only; `from_b` and `to`
    are capture rewrites, not independent patterns — the doc example
    `add: {from_a: '.*.weight', from_b: '.*.delta'}` reads like both are patterns.
  - The intermediate `B @ A` product has to be created on the **base** alias, not the
    adapter alias, or the run fails with `cannot infer output model uniquely`.
  - The intermediate must not be named with `lora_` in it, since the required check
    asserts no such name exists before writing; I used `<module>.mergedelta`.
  - Adding a float32 delta to a float16 base needs an explicit `cast_` up and back;
    the transforms require matching dtypes for in-place `add_`.
  - Shard units are binary (`512MB` = 512 MiB), which is exactly the task budget.
- **Anything in the task text or documentation that was unclear:**
  - TASK.md says `gpt_neox.embed_in.weight` / `embed_out.weight` (206 MB each) are
    "a single tensor larger than" 512 MiB and get their own shard — they are not
    larger than 512 MiB, and the default packing put them together with other
    tensors in shard 1 (529,608,772 bytes of tensor data, under budget).
  - TASK.md lists `target_modules = ["attention.query_key_value"]`, the actual
    `adapter_config.json` has `["query_key_value"]`; immaterial here.
- **Tools used (condition F):** n/a.
- **Approximate time spent, if you can tell:** ~10 minutes.

## Verification done outside the plan (read-only)

Reading the output back: 244 tensors over 4 shards, every shard ≤ 512 MiB of
tensor data, no `lora_` names, the 228 untouched tensors bit-identical to the
base, and the 16 merged weights matching `(W.float() + 2 * B @ A).half()` exactly
(relative Frobenius error 0.0), dtype float16, shape [6144, 2048].
