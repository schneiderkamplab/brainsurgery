# T2 self-report (condition F, Pythia-1B)

- **Final artifact path:** `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution
  succeeded and all required checks passed.
- **Pitfalls or surprises you hit (one line each):**
  - `transformers` 5.12.1 has no head-pruning path for GPT-NeoX at all
    (`modeling_gpt_neox.py` contains no `prune` code), so the route suggested in
    `F-allowed.md` for this task does not exist; I checked this before writing code.
  - The fused `query_key_value` layout is head-major with q/k/v interleaved
    *inside* each 768-row head block, not `[q | k | v]` segments; treating it as
    segments would drop three wrong 256-row stripes while still yielding the
    correct `[5376, 2048]` shape, so shape checks alone do not catch that error.
  - The two head axes differ: rows for `query_key_value.{weight,bias}`, columns
    (dim 1) for `attention.dense.weight`.
  - `attention.bias` is a `uint8` buffer (not float16) and `attention.dense.bias`
    is not head-bearing; both must be copied through untouched, so a blanket
    "all tensors are float16" or "every attention.* tensor is head-bearing"
    assumption would be wrong.
  - Slicing the stored fp16 tensors keeps values bit-exact; loading through a
    model and re-saving would have risked a dtype round-trip.
- **Anything in the task text or documentation that was unclear:** nothing
  blocking — the task text specifies the row/column ranges explicitly. One
  observation: the result is not loadable by a stock `GPTNeoXConfig` with
  `num_attention_heads=7`, because that config derives `head_size =
  hidden_size // num_attention_heads = 2048 // 7 = 292`, not 256; the pruned
  shapes follow the task's explicit spec, which is what grading compares.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — memory-mapped read of the input header and tensors and
    the output write; keeps the `{"format": "pt"}` metadata and the fp16 bits.
  - `torch` 2.14.0 — `index_select` on dim 0 / dim 1 to drop head 5's block.
  - `transformers` 5.12.1 — inspected only, to confirm GPT-NeoX has no
    `prune_heads` implementation; not used to produce the output.
  - Not used: `mergekit` and `torch-state-bridge` operate on whole tensors and
    key names (layer slicing, key rewriting, merging); neither can slice *inside*
    a tensor, which is the entire task here. `peft` is irrelevant (no adapters).
- **Approximate time spent, if you can tell:** ~5 minutes.

## What the script enforces before writing

Written by `safetensors.torch.save_file` only after all of these hold:

- the input has 244 tensors and contains all 48 head-bearing tensors;
- the keep-indices literally equal rows `0..3839` + `4608..6143` and columns
  `0..1279` + `1536..2047`;
- `gpt_neox.layers.0.attention.query_key_value.weight` is `[5376, 2048]`;
- `gpt_neox.layers.0.attention.query_key_value.bias` is `[5376]`;
- `gpt_neox.layers.0.attention.dense.weight` is `[2048, 1792]`;
- the same three shape checks for *every* layer 0..15 (48 tensors edited);
- every non-buffer tensor is still float16;
- the output has exactly 244 tensors.

Any failure raises `CheckFailed` and no file is written.

## Post-run verification (separate from the script)

Compared `out/T2/model.safetensors` against the input tensor by tensor: key sets
identical (244), dtypes identical, the 48 edited tensors bit-equal to the
expected `torch.cat` of the kept slices, and all 196 other tensors bit-equal to
the input.
