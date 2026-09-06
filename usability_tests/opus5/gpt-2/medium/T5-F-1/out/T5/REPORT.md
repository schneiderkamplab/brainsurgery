# T5 self-report

- **Final artifact path:** `out/T5/solution.py` (run: `.venv/bin/python out/T5/solution.py`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the first execution succeeded.
- **Pitfalls or surprises you hit:**
  - Conv1D layout: the base `h.<i>.attn.c_attn.weight` is `[in, out] = [768, 2304]`
    while `B @ A` is `[out, in]`, so `fan_in_fan_out = true` means the delta is
    transposed before adding. Getting this backwards would silently fail only at grading.
  - PEFT key prefix `base_model.model.` has to be stripped to reach the base names.
  - `adapter_config.json` lists `target_modules: ["c_attn"]` (bare module name), not
    `"attn.c_attn"` as TASK.md states; I derived target names from the adapter keys
    themselves rather than from `target_modules`, which sidesteps the discrepancy.
  - Shard budget vs. an oversized tensor: `wte.weight` is 154 MB, over the 100 MiB
    budget, so the packing loop has to seal a single oversized tensor into its own
    shard instead of trying to fit it.
- **Anything unclear:** the `target_modules` mismatch noted above; also the task does
  not fix a shard file naming scheme, so I used the HuggingFace convention
  `model-000NN-of-000NN.safetensors` with a greedy first-fit packing in base key order.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — direct `load_file` / `save_file` on the checkpoint files.
  - `torch` 2.14.0 — float32 matmul, transpose, add.
  - No `peft` / `mergekit`. `peft.merge_and_unload` would require instantiating a
    GPT-2 model and would then re-serialize through `save_pretrained`, giving me no
    direct control over the exact 100 MiB shard budget or the guarantee that the key
    set stays exactly the base's 160 names; the task is explicitly checkpoint-level
    surgery, and a ~90-line script does it with all four required checks enforced
    in-process before anything is written.
- **Approximate time spent:** ~5 minutes.

## Verification performed

An independent re-read of `out/T5` confirmed: 160 tensors across 5 shards, index
`weight_map` covering all of them, every shard ≤ 100 MiB except the single-tensor
`wte.weight` shard, exactly 12 tensors differing from the base, and relative error
0.0 against an independently recomputed `W + 2*(B@A).T` for a spot-checked layer.
