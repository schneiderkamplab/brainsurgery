# T2 self-report (condition F)

- **Final artifact path:** `out/T2/solution.py` -> `out/T2/model.safetensors`

- **Number of times you executed the script or plan:** 2

- **Which executions failed, and why (one line each):**
  - Execution 1: `failed_assertion` — my own post-condition `assert t.dtype == torch.float16` fired on `gpt_neox.layers.0.attention.bias`, which is `uint8`; the checkpoint is not uniformly float16. No output was written (the check runs before `save_file`). Fixed by asserting each tensor's dtype equals *its own source dtype* instead of a hardcoded one.

- **Pitfalls or surprises you hit (one line each):**
  - The three attention buffers (`attention.bias`, `masked_bias`, `rotary_emb.inv_freq`) are not float16 — `attention.bias` is `uint8` and `masked_bias`/`inv_freq` are their own dtypes; "the checkpoint is stored in float16" describes the weights, not every tensor.
  - The two head axes differ: heads are **rows** (dim 0) of `query_key_value.{weight,bias}` but **columns** (dim 1) of `dense.weight`. Slicing dense on dim 0 would have produced a file with plausible-looking `[1792, 2048]` shapes that silently fails the spec.
  - GPT-NeoX interleaves q/k/v *inside* each head, so a head is one contiguous 768-row block; treating the tensor as `[q | k | v]` segments (the layout most other architectures use) would need three separate slices and give a different, wrong result with identical shapes. The required-checks shape list cannot distinguish the two, so shapes alone are not sufficient evidence of correctness.
  - `attention.dense.bias` stays `[2048]`: it is applied after the output projection, so it is not per head even though it lives on the same module.
  - A 7-head Pythia is **not loadable in HF Transformers 5.12.1 as-is**: `GPTNeoXAttention` hardcodes `query_key_value = Linear(hidden, 3*hidden)` and `dense = Linear(hidden, hidden)` (`modeling_gpt_neox.py:200-201`), so no config makes the module expect `[5376, 2048]`/`[2048, 1792]`. That is a limitation of the reference modeling code, not of the checkpoint.
  - Consequently the advertised route "T2 via transformers `prune_heads`" does not apply: `GPTNeoXForCausalLM` does not implement `_prune_heads` (the base `prune_heads` needs per-model support and NeoX's fused, interleaved qkv has none), so I did not pursue it.

- **Anything in the task text or documentation that was unclear:**
  - "The result must be loadable as the same architecture with 7 heads per layer" is not achievable with the installed Transformers, per the point above; I read it as a statement of intent about the layout and graded myself on the explicit row/column index lists instead.
  - Nothing else — TASK.md gave the exact keep-ranges, so the layout facts were specified rather than something I had to infer. I verified them independently anyway (below) rather than trusting the prose.

- **Tools used (condition F):**
  - `safetensors` 0.5.3 — load and save. The only thing actually needed: the task is a pure tensor-slicing rewrite with unchanged keys, so streaming tensors in and writing them back out is the whole job.
  - `torch` 2.14.0 — `index_select` for the keep-index slicing and `torch.equal` for verification.
  - `transformers` 5.12.1 — only to *check* the `prune_heads` route and read `modeling_gpt_neox.py`; not used to produce the output.
  - I deliberately did **not** use mergekit or torch-state-bridge: both operate on keys/whole tensors (layer slicing, key rewriting, merge arithmetic) and neither can express "drop a sub-range of one axis of a tensor", which is the entire task. A ~60-line script with explicit assertions was the shortest correct route and the easiest to make bit-exact.

- **How I verified (beyond the required checks):**
  - The required checks plus per-layer shape checks for all 16 layers, an edited-tensor count (16/16/16 = 48), an unchanged key set, and per-tensor dtype preservation, all asserted *before* writing.
  - After writing, an independent re-derivation: reloaded both files and confirmed every one of the 244 tensors is `torch.equal` to the expected slice of the source (bit-exact, dtypes matched).
  - A functional check: hand-rolled NeoX attention on layer 0 in float64 shows the 7-head pruned checkpoint's output is *exactly* equal (max abs diff 0.0) to the original 8-head layer with head 5's contribution zeroed, and differs from the unmodified layer (max abs diff 1.73). This is what rules out the `[q|k|v]`-segment misreading, which shapes alone cannot catch.

- **Approximate time spent:** ~5 minutes.
